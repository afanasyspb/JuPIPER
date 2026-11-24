"""
    JuPIPER

A from-scratch, allocation-free Julia core for FFT-based rigid-body protein-protein
docking, following the PIPER algorithm of Kozakov, Brenke, Comeau & Vajda
(Proteins 65:392–406, 2006, https://doi.org/10.1002/prot.21117).

The core defines the memory model, backend interface, and rotation loop.
The scoring functions document their assumptions about radii, charges, and
deposition on the grid.

# FFT correlation

PIPER's energy for a ligand translated by `(a,b,c)` is a sum of P correlations

    E(a,b,c) = sum_p sum_ijk R_p[i,j,k] * L_p[i+a, j+b, k+c]

and the reference implementation evaluates it with **P forward and P inverse**
FFTs (paper, PDF p.8). Because the inverse transform is linear, the sum can be
accumulated in the Fourier domain and inverted **once**:

    E = irfft( sum_p conj(rfft(R_p)).* rfft(L_p) )

That is `P + 1` transforms instead of `2P`. With the paper's own P in 2..4 and two extra
terms (shape, electrostatics), this is a 1.5-1.6x reduction in transform count before
any tuning, threading, or GPU work.

# Backend story

There is exactly **one** implementation of the hot loop. It is generic over the
array type. `Array` -> FFTW + Julia threads; `CuArray` -> cuFFT, batched over
rotations. Only allocation, plan construction, and the voxelisation kernel are
backend-aware, and `KernelAbstractions.jl` unifies the last of those.
"""
module JuPIPER

using FFTW
using LinearAlgebra: LinearAlgebra, mul!, dot, tr, eigen, Symmetric, I, kron, norm
using StaticArrays: SVector, SMatrix, MVector
using KernelAbstractions
using KernelAbstractions: get_backend, synchronize, @kernel, @index, @Const
using Atomix: Atomix
using Adapt: Adapt, adapt
using Random: Random
using Printf: @sprintf
using PrecompileTools: @setup_workload, @compile_workload

export GridSpec, Structure, Potential, Workspace, ReceptorSpectra
export gridspec_for, workspace, correlate!, scan_best!, dock!
export pairwise_rmsd!, greedy_cluster
export nextfftsize, load_dars, truncation_error

export RotationSet, rotations_fibonacci, rotations_hopf, rotations_random
export rotations_hopf_balanced, hopf_split
export fiber, rotmatrix, rotangle
export covering_radius, covering_bound, covering_efficiency, nn_stats
export covering_spread

export shape_receptor_grid!, occupancy_grid!, shape_spectra, shape_structure
export spectra_from_grids, vdw_radius

export MemoryModel, memory_model, bytes_per_batch, bytes_fixed, job_bytes
export max_batch, autotune_batch, calibrate_fft_factor, safe_fft_factor, memory_report
export MemoryGuard, memory_guard, guard_interval, maybe_collect!
export device_free_bytes, device_total_bytes, device_collect!, device_synchronize!


export PDBSource, BioStructuresPDB, AtomTable, IngestReport
export read_atoms, center!, uncenter!, extent, infer_element
export TailReport, tail_report, warn_on_tail, TAIL_INFLATION_WARN

export ACPTypeTable, load_acp_types, acp_type, acp_types!, ACP18_LABELS
export ChargeModel, ZeroCharges, FormalCharges, atom_charge
export PiperWeights, PIPER_WEIGHTS_ENZYME_INHIBITOR, PIPER_WEIGHTS_ANTIBODY_ANTIGEN
export Parametrization, parametrize, piper_structure, type_histogram, has_vdw_radius

export GridFit, grid_fit, grid_fit_rotated, stencil_leak, mass_defect, mass_tolerance

export GBTruncation, GBCapped, GBTruncated
export gb_receptor_grid!, dars_receptor_grid!, box_smooth!, box_taps
export BoxFilter, box_filter
export dars_rank, type_abundance, receptor_weights
export piper_receptor_grids!, piper_receptor_spectra
export PIPER_D, PIPER_DPAD, PIPER_SMOOTH_BOX, PIPER_SPACING

export ClusProProfile, load_cluspro_profiles, cluspro_profile, shape_ratio
export assumed_piper_weights, rescale_terms!, term_ratio!, retarget_spectra!

export minimal_image, pose_transform, pose_transforms, pose_coords!, pose_coords
export native_ligand_coords, ca_mask, residue_ids, interface_masks
export ligand_rmsd, kabsch, superposed_rmsd, interface_rmsd
export SearchFloor, search_floor, hit_rank, subset_frame

export FibreBound, fibre_bound, envelopes!, bound_landscape!, fibre_bound_min
export fibre_core!, EnvelopeMode, SweptSet, PerCellExact

export dock_pruned!, PruneReport, prune_rate

export SpatialMask, no_mask, is_identity, mask_crossover
export residue_mask, spatial_mask, spatial_mask!, masked_residue_count

export Restraint, RestraintGroup, RestraintSet
export restraint, restraint_group, restraint_set, nrestraints
export RestraintWorkspace, restraint_workspace, feasible_translations!
export dock_restrained!, RestraintReport, skip_rate, feasible_fraction

export CyclicSymmetry, cyclic_symmetry, symmetry_axes, hopf_axes, axis_angle
export SymmetryFit, symmetry_fit, plane_cells, plane_cells!, symmetric_search_floor
export dock_symmetric!, SymmetryReport, rotation_reduction, translation_reduction
export CLUSPRO_SYM_TOL, sym_tol_floor


"""
    Backend

Trait selecting where buffers live and which FFT library runs. `CPUBackend` is
always available; `CUDABackend` becomes functional when the CUDA package
extension is loaded (see `ext/`). Dispatch on the *array type* does the real
work -- this trait only exists to pick the allocator.
"""
abstract type Backend end

struct CPUBackend <: Backend end

"""GPU backend marker. Requires loading `CUDA` for the extension to define its methods."""
struct CUDABackend <: Backend end

"""
    _backend_unavailable(backend, what)

Uniform error for a seam whose implementation lives in a package extension that
has not been loaded.

!!! warning "Never annotate this on a concrete backend the extension also claims"
    The obvious way to write a helpful stub is `allocate(::CUDABackend,...) =
    error("load CUDA")`, and it is wrong. The extension defines `allocate(::CUDABackend,
    ::Type{T}, dims::Integer...)` -- the *same signature* -- so loading it is a method
    overwrite, which Julia refuses during precompilation of the extension module:
    `ERROR: Method overwriting is not permitted during Module precompilation`. The
    message is emitted against the extension, which is the file that did nothing wrong.

    Every seam below is therefore declared at `::Backend`, strictly less specific than
    any concrete backend, so an extension method always wins by specificity and can
    never collide.
"""
@noinline function _backend_unavailable(backend::Backend, what::AbstractString)
    error("""
          `$(what)` has no method for $(nameof(typeof(backend))).

          If this is CUDABackend, the extension is not loaded. Activate it with
          `using CUDA, JuPIPER` in an environment that has CUDA -- that is
          `--project=gpu`, never `--project=.` or `--project=bench` (both are
          deliberately CUDA-free; see gpu/Project.toml).

          If this is a new backend, it must supply: allocate, adapt_rots,
          adapt_backend, device_free_bytes, device_total_bytes, device_collect!,
          device_synchronize!.
          """)
end

"""
    allocate(backend, T, dims...) -> AbstractArray{T}

The single seam between CPU and GPU memory. The CUDA extension adds
`allocate(::CUDABackend, ...) = CuArray{T}(undef, dims...)`; nothing else in the
hot path needs to know which backend it is on.
"""
allocate(::CPUBackend, ::Type{T}, dims::Integer...) where {T} = Array{T}(undef, dims...)

allocate(b::Backend, ::Type{T}, dims::Integer...) where {T} =
    _backend_unavailable(b, "allocate")


"""
    isfftfriendly(n) -> Bool

FFTW has hand-written codelets for sizes of the form `2^a 3^b 5^c 7^d 11^e 13^f`
with `e + f <= 1`; other sizes fall back to a slower generic (Bluestein / Rader)
path. We additionally require `n` to be even so that the r2c output length
`n >> 1 + 1` is exact and the c2r inverse is unambiguous.
"""
function isfftfriendly(n::Integer)
    n > 0 || return false
    iseven(n) || return false
    m = n
    for p in (2, 3, 5, 7)
        while m % p == 0
            m ÷= p
        end
    end
    return m == 1 || m == 11 || m == 13
end

"""
    nextfftsize(n) -> Int

Smallest FFT-friendly size `>= n`. This is the `r(N)` of the PIPER paper
("moves N to the nearest product of small prime numbers", PDF p.13).
"""
function nextfftsize(n::Integer)
    m = max(2, Int(n))
    iseven(m) || (m += 1)
    while !isfftfriendly(m)
        m += 2
    end
    return m
end

"""
    GridSpec{T}

Rectilinear grid the correlation lives on. `n` is the cell count per axis,
`spacing` the cell size in angstrom (PIPER uses 1.2 A), `origin` the Cartesian
coordinate of cell `(1,1,1)`.
"""
struct GridSpec{T<:AbstractFloat}
    n::NTuple{3,Int}
    spacing::T
    origin::SVector{3,T}
end

Base.length(g::GridSpec) = prod(g.n)

"""Number of complex rows in the r2c spectrum (first transformed axis is halved)."""
speclen(g::GridSpec) = (g.n[1] >> 1) + 1

"""
    gridspec_for(receptor, ligand; spacing=1.2, pad=2) -> GridSpec

Grid sizing rule from the PIPER paper (PDF p.13):

    S_x = extent_x(receptor) + max(extent_x, extent_y, extent_z)(ligand)
    N_x = r(S_x / spacing + pad)

The ligand contributes its **maximum extent over all three axes** because it is
rotated arbitrarily, so its footprint along any given axis is not known ahead of
time. Note the paper deliberately picks a grid slightly too small to hold the
ligand at the far corners: those placements wrap around under the FFT's periodic
boundary, but they correspond to large separations where the energy is ~0, so
the aliasing is harmless. We reproduce that choice; `pad` is the knob that
buys margin if a future term is not short-ranged.
"""
function gridspec_for(receptor::AbstractVector{SVector{3,T}},
                      ligand::AbstractVector{SVector{3,T}};
                      spacing::Real = 1.2, pad::Real = 2) where {T}
    ext(pts, ax) = begin
        lo, hi = typemax(T), typemin(T)
        for p in pts
            lo = min(lo, p[ax]); hi = max(hi, p[ax])
        end
        hi - lo
    end
    ligmax = maximum(ax -> ext(ligand, ax), 1:3)
    s = T(spacing)
    n = ntuple(ax -> nextfftsize(ceil(Int, (ext(receptor, ax) + ligmax) / s + pad)), 3)

    c = sum(receptor) / length(receptor)
    origin = c - SVector{3,T}(n[1] * s / 2, n[2] * s / 2, n[3] * s / 2)
    return GridSpec{T}(n, s, origin)
end


"""
    Structure{T}

Atoms in struct-of-arrays layout. `coords` are **centred on the centroid** so
that a rotation is a pure `R * x` with no translation bookkeeping, and so that
the closed-form RMSD in `pairwise_rmsd!` is valid.

`termw` is the per-atom weight for each correlation term, precomputed once and
reused for all ~70,000 rotations: column `p` holds the scalar this atom
contributes to grid `p`. Layout is `(natoms, nterms)` so a term is contiguous.
"""
struct Structure{T,VC<:AbstractVector{SVector{3,T}},MW<:AbstractMatrix{T}}
    coords::VC
    termw::MW
end


natoms(s::Structure) = length(s.coords)
nterms(s::Structure) = size(s.termw, 2)

Adapt.adapt_structure(to, s::Structure) =
    Structure(adapt(to, s.coords), adapt(to, s.termw))

"""
    build_termweights(typeidx, charge, U) -> Matrix

Assemble the `(natoms, nterms)` weight matrix in PIPER's term order:

  1. shape complementarity  -- ligand occupancy, weight 1 for every atom
  2. electrostatics         -- ligand partial charge
  3..2+P  pairwise (DARS)   -- `u_pJ`, the p-th eigenvector component for the
                               atom's type J (paper, PDF p.13)

The receptor-side counterparts carry the weights `w1, w2, w3` and the
eigenvalues `lambda_p`; folding them there (see [`ReceptorSpectra`](@ref))
keeps the ligand side rotation-independent and the hot loop free of scaling.
"""
function build_termweights(typeidx::AbstractVector{<:Integer},
                           charge::AbstractVector{T},
                           U::AbstractMatrix{T}) where {T}
    na = length(typeidx)
    P = size(U, 2)
    W = Matrix{T}(undef, na, 2 + P)
    @inbounds for i in 1:na
        W[i, 1] = one(T)
        W[i, 2] = charge[i]
        for p in 1:P
            W[i, 2+p] = U[typeidx[i], p]
        end
    end
    return W
end


"""
    Potential{T}

A `K x K` symmetric contact-energy matrix and its eigen-decomposition.
PIPER's key trick (PDF p.12): `eps_IJ = sum_p lambda_p u_pI u_pJ`, so a
K-atom-type pairwise potential -- which would naively need K correlations --
collapses to `P` correlations once the spectrum is truncated to the `P` largest
`|lambda_p|`.

The paper reports that `2 <= P <= 4` suffices, justified by the ~10% error the grid
discretisation already introduces -- but read that sentence carefully before applying it
here. It is stated about **ACP and RPScore** (PDF p.13), two potentials this repository
does not hold; **on the DARS matrix rank 4 leaves 14.5%**, and the first rank under 10%
is 5.
"""
struct Potential{T<:AbstractFloat}
    eps::Matrix{T}
    labels::Vector{String}
    lambda::Vector{T}
    U::Matrix{T}
end

function Potential(eps::AbstractMatrix{T}, labels::AbstractVector{<:AbstractString}) where {T}
    size(eps, 1) == size(eps, 2) || throw(ArgumentError("potential matrix must be square"))
    maximum(abs, eps .- eps') < sqrt(eps_tol(T)) ||
        throw(ArgumentError("potential matrix is not symmetric"))
    F = eigen(Symmetric(Matrix{T}(eps)))
    ord = sortperm(F.values; by = abs, rev = true)
    return Potential{T}(Matrix{T}(eps), String.(labels), F.values[ord], F.vectors[:, ord])
end

"""
    Potential{T}(p::Potential) -> Potential{T}

Narrow (or widen) a potential's element type **without recomputing its
eigen-decomposition**.

The distinction is the whole reason this exists rather than `load_dars(; T = Float32)`.
Both give a `Potential{Float32}`, but that one eigendecomposes an 18x18 symmetric matrix
*in single precision*, and every downstream quantity -- the rank `dars_rank` selects,
the truncation error that justifies it, the eigenvector each atom type is paired with --
inherits the result. This one does the decomposition in whatever `p` was built at
(Float64 by default) and narrows only the stored values.

Needed because `piper_receptor_spectra` requires the potential, the atom table and the
grid to share `T`, so a Float32 device run cannot use `load_dars` directly.
`bench/realdock.jl` is that caller.
"""
Potential{T}(p::Potential) where {T} =
    Potential{T}(Matrix{T}(p.eps), copy(p.labels), Vector{T}(p.lambda), Matrix{T}(p.U))


Base.convert(::Type{Potential{T}}, p::Potential) where {T} =
    p isa Potential{T} ? p : Potential{T}(p)

eps_tol(::Type{T}) where {T} = eps(T)

"""
    truncation_error(pot, P; weights=nothing) -> (abs_fro, rel_fro)

Frobenius error of approximating `eps` by its `P` dominant eigenpairs. This is
the number that decides `P`; run it before committing to a rank. See
[`dars_rank`](@ref), which inverts it.

`weights`, if given, is a per-type importance vector `w`, and entry `(I,J)` is weighted
by `w_I w_J` in both the numerator and the denominator. Unweighted -- the default, and
what the paper's own reasoning implies -- every type pair counts equally, which
over-weights types that barely occur: on the DARS matrix the `CSg` row and column alone
carry **50%** of `||eps||_F^2`, and the paper itself calls that entry an artifact.
[`type_abundance`](@ref) supplies a composition-based `w`; it moves the answer by only
1-2 points, which is itself worth knowing.
"""
function truncation_error(pot::Potential{T}, P::Integer;
                          weights::Union{Nothing,AbstractVector} = nothing) where {T}
    K = size(pot.eps, 1)
    approx = zeros(T, K, K)
    @inbounds for p in 1:P
        lp = pot.lambda[p]
        up = @view pot.U[:, p]
        for j in 1:K, i in 1:K
            approx[i, j] += lp * up[i] * up[j]
        end
    end
    if weights === nothing
        e = sqrt(sum(abs2, pot.eps .- approx))
        return (e, e / sqrt(sum(abs2, pot.eps)))
    end
    length(weights) == K ||
        throw(DimensionMismatch("weights must have $K entries, got $(length(weights))"))
    num = zero(T); den = zero(T)
    @inbounds for j in 1:K, i in 1:K
        wij = T(weights[i]) * T(weights[j])
        num += wij * abs2(pot.eps[i, j] - approx[i, j])
        den += wij * abs2(pot.eps[i, j])
    end
    e = sqrt(num)
    return (e, e / sqrt(den))
end

"""
    load_dars(path=...) -> Potential

Read the 18x18 DARS matrix transcribed from Table I of the PIPER preprint.
See `data/potentials/dars_acp18.tsv` for provenance and transcription caveats.
"""
function load_dars(path::AbstractString = joinpath(@__DIR__, "..", "data",
                                                   "potentials", "dars_acp18.tsv");
                   T::Type = Float64)
    labels = String[]
    rows = Vector{Vector{T}}()
    header = String[]
    for line in eachline(path)
        isempty(strip(line)) && continue
        startswith(strip(line), "#") && continue
        fields = split(rstrip(line), '\t')
        if isempty(header)
            header = String.(fields[2:end])
            continue
        end
        push!(labels, String(fields[1]))
        push!(rows, parse.(T, fields[2:end]))
    end
    K = length(labels)
    M = Matrix{T}(undef, K, K)
    for i in 1:K
        M[i, :] .= rows[i]
    end
    labels == header || @warn "DARS row labels and column headers disagree" labels header
    return Potential(M, labels)
end


"""
    Trilinear

Deposition kernel: each atom spreads its weight over the 8 surrounding cells
with trilinear weights.

The reference uses indicator shells with `d = r_vdw + 2 A` and `D = 6 A`.
"""
struct Trilinear end

@kernel function _deposit_kernel!(dst, @Const(coords), @Const(termw),
                                  @Const(rots), term::Int,
                                  origin::SVector{3}, invspacing, nx::Int, ny::Int, nz::Int)
    a, b = @index(Global, NTuple)
    R = rots[b]
    w = termw[a, term]

    p = R * coords[a]
    g = (p - origin) * invspacing
    i0 = unsafe_trunc(Int, floor(g[1]))
    j0 = unsafe_trunc(Int, floor(g[2]))
    k0 = unsafe_trunc(Int, floor(g[3]))
    fx = g[1] - i0
    fy = g[2] - j0
    fz = g[3] - k0
    @inbounds for dk in 0:1, dj in 0:1, di in 0:1
        i = i0 + di; j = j0 + dj; k = k0 + dk
        if 1 <= i+1 <= nx && 1 <= j+1 <= ny && 1 <= k+1 <= nz
            wx = di == 0 ? (1 - fx) : fx
            wy = dj == 0 ? (1 - fy) : fy
            wz = dk == 0 ? (1 - fz) : fz

            Atomix.@atomic dst[i+1, j+1, k+1, b] += w * wx * wy * wz
        end
    end
end

"""
    voxelize!(dst, structure, rots, term, spec)

Zero `dst` and deposit correlation term `term` of `structure` under each
rotation in `rots`, one batch slice per rotation. `dst` is `(nx, ny, nz, B)`.

Two methods, selected by dispatch on the destination array type:

  * `::Array` -- a plain serial loop. Over 70,000 rotations x 5 terms that is ~78 MB of
    garbage and a steady GC drip in the hot loop, which is exactly what this project
    exists to avoid. Since each CPU task owns its Workspace exclusively, the scatter
    needs no atomics either, so the hand-written loop is both allocation-free and
    faster.
  * `::AbstractArray` -- the KernelAbstractions path, which is what runs on GPU.

The two must stay numerically identical; `test/test_backends.jl` asserts it.
"""
function voxelize!(dst::Array{T,4}, s::Structure, rots::AbstractVector,
                   term::Integer, spec::GridSpec{T}, ::Trilinear = Trilinear()) where {T}
    fill!(dst, zero(T))
    nx, ny, nz, B = size(dst)
    length(rots) == B || throw(DimensionMismatch("rots ($(length(rots))) vs batch ($B)"))
    coords = s.coords
    termw = s.termw
    origin = spec.origin
    inv_h = inv(spec.spacing)
    t = Int(term)
    @inbounds for b in 1:B
        R = rots[b]
        for a in eachindex(coords)
            w = termw[a, t]
            g = (R * coords[a] - origin) * inv_h
            i0 = unsafe_trunc(Int, floor(g[1]))
            j0 = unsafe_trunc(Int, floor(g[2]))
            k0 = unsafe_trunc(Int, floor(g[3]))
            fx = g[1] - i0; fy = g[2] - j0; fz = g[3] - k0
            if 0 <= i0 && i0 + 1 < nx && 0 <= j0 && j0 + 1 < ny && 0 <= k0 && k0 + 1 < nz

                for dk in 0:1
                    wz = dk == 0 ? (1 - fz) : fz
                    for dj in 0:1
                        wy = dj == 0 ? (1 - fy) : fy
                        wyz = w * wy * wz
                        dst[i0+1, j0+dj+1, k0+dk+1, b] += wyz * (1 - fx)
                        dst[i0+2, j0+dj+1, k0+dk+1, b] += wyz * fx
                    end
                end
            else

                for dk in 0:1, dj in 0:1, di in 0:1
                    i = i0 + di + 1; j = j0 + dj + 1; k = k0 + dk + 1
                    (1 <= i <= nx && 1 <= j <= ny && 1 <= k <= nz) || continue
                    wx = di == 0 ? (1 - fx) : fx
                    wy = dj == 0 ? (1 - fy) : fy
                    wz = dk == 0 ? (1 - fz) : fz
                    dst[i, j, k, b] += w * wx * wy * wz
                end
            end
        end
    end
    return dst
end

function voxelize!(dst::AbstractArray{T,4}, s::Structure, rots::AbstractVector,
                   term::Integer, spec::GridSpec{T}, ::Trilinear = Trilinear()) where {T}
    fill!(dst, zero(T))
    backend = get_backend(dst)
    nx, ny, nz, B = size(dst)
    length(rots) == B || throw(DimensionMismatch("rots ($(length(rots))) vs batch ($B)"))
    kern = _deposit_kernel!(backend)
    kern(dst, s.coords, s.termw, rots, Int(term),
         spec.origin, inv(spec.spacing), nx, ny, nz;
         ndrange = (natoms(s), B))
    synchronize(backend)
    return dst
end


"""
    ReceptorSpectra{CA}

`conj(rfft(R_p))`, pre-scaled by the term weight *and* by `1/(nx*ny*nz)`.

Folding both the physical weights (`w1, w2, w3, lambda_p`) and the inverse-FFT
normalisation into the receptor spectra at setup time means the 70,000-iteration
hot loop performs **no scaling passes at all** -- the unnormalised `brfft` comes
out correctly scaled for free.

Stored as a `Vector` of dense 3-D arrays, one per term, rather than one 5-D
array. The reason is [`_accumulate!`](@ref): its fast path reinterprets the
complex buffers as pairs of reals, which requires a dense array rather than a
`SubArray` slice.
"""
struct ReceptorSpectra{T,CA<:AbstractArray{Complex{T},3}}
    data::Vector{CA}
    spec::GridSpec{T}
end

nterms(rs::ReceptorSpectra) = length(rs.data)

"""
    receptor_spectra(backend, receptor, spec, weights; flags) -> ReceptorSpectra

This function deposits receptor and ligand weights by trilinear interpolation.
For the PIPER scoring fields, use [`piper_receptor_spectra`](@ref). The reference builds

  * shape: `R[l,m,n] = -c[l,m,n] + w1 * r[l,m,n]` where `c` counts receptor atoms
    at `d < r < D` from the cell and `r` counts those at `r < d`, with `D = 6 A`
    and `d = r_vdw + 2 A` (PDF p.10);
  * electrostatics: a simplified constant-Born-radii GB field, truncated at `2D`
    and smoothed by convolution with a 3 A box (PDF p.11);
  * pairwise: `R_p[l,m,n] = sum_i u_pI * delta_i` with `delta_i = 1` when the cell
    is within `D` of receptor atom `i` (PDF p.13).
"""
function receptor_spectra(backend::Backend, rec::Structure{T}, spec::GridSpec{T},
                          weights::AbstractVector{T};
                          flags = FFTW.MEASURE) where {T}
    P = nterms(rec)
    length(weights) == P || throw(DimensionMismatch("one weight per term required"))
    nx, ny, nz = spec.n
    nxr = speclen(spec)
    scale = one(T) / T(nx * ny * nz)

    grid = allocate(backend, T, nx, ny, nz, 1)
    plan = FFTW_plan_rfft(grid, flags)
    ident = adapt_rots(backend, [one(SMatrix{3,3,T,9})])

    tmp = allocate(backend, Complex{T}, nxr, ny, nz, 1)
    terms = [allocate(backend, Complex{T}, nxr, ny, nz) for _ in 1:P]
    for p in 1:P
        voxelize!(grid, rec, ident, p, spec)
        mul!(tmp, plan, grid)
        t3 = reshape(tmp, nxr, ny, nz)
        terms[p] .= conj.(t3) .* (weights[p] * scale)
    end
    return ReceptorSpectra{T,eltype(terms)}(terms, spec)
end


FFTW_plan_rfft(A::AbstractArray, flags) = plan_rfft(A, 1:3; flags = flags)
FFTW_plan_brfft(A::AbstractArray, d::Integer, flags) = plan_brfft(A, d, 1:3; flags = flags)
adapt_rots(::CPUBackend, v::AbstractVector) = v


adapt_rots(b::Backend, ::AbstractVector) = _backend_unavailable(b, "adapt_rots")

"""
    adapt_backend(backend, x) -> x'

Move `x` to wherever `backend` computes. Identity on the CPU.

The general form of [`adapt_rots`](@ref), which predates it and stays because
rotations are the case the hot loop needs. This one exists for `Structure`, whose
coordinate vector and term-weight matrix are both read *inside* the voxelisation
kernel -- so a host-resident `Structure` handed to a device workspace fails
rather than silently copying, and a caller that only has a `Backend` (not the
`CuArray` type itself) previously had no way to move one. `bench/realdock.jl` is
that caller.

Annotated `::Backend` for the fallback, never `::CUDABackend`: an identical signature in
the base module and the extension is a method overwrite, which Julia 1.12 turns into a
precompilation error reported against the extension. `test_vram.jl` counts methods per
defining module to pin it.
"""
adapt_backend(::CPUBackend, x) = x
adapt_backend(b::Backend, x) = _backend_unavailable(b, "adapt_backend")


"""
    Workspace{T,RA,CA,FP,BP}

All scratch for one *independent stream* of rotations. On CPU there is one
Workspace per task (`batch == 1`); on GPU there is one Workspace with
`batch == 32..64` so cuFFT sees a batched transform and kernel-launch overhead
amortises.

Critically, the P ligand grids are **not** materialised simultaneously. Term `p` is
voxelised, transformed, and accumulated before term `p+1` reuses the same two buffers.
That holds the per-stream working set at four grid-sized arrays instead of `2P + 2`,
which is the difference between fitting in cache and not.

Buffer inventory (Float32, 128^3, batch B):
    lig_real  nx*ny*nz*B*4       8.0 MiB * B
    lig_spec  nxr*ny*nz*B*8      8.1 MiB * B
    accum     nxr*ny*nz*B*8      8.1 MiB * B
    score     nx*ny*nz*B*4       8.0 MiB * B
                                ~32 MiB * B
"""
struct Workspace{T,RA<:AbstractArray{T,4},CA<:AbstractArray{Complex{T},4},FP,BP}
    spec::GridSpec{T}
    nterms::Int
    batch::Int
    lig_real::RA
    lig_spec::CA
    accum::CA
    score::RA
    fwd::FP
    bwd::BP
end

"""
    workspace(backend, spec, nterms; T=Float32, batch=1, flags=FFTW.MEASURE)

Allocate one stream's buffers and build its FFT plans.

`flags = FFTW.MEASURE` is the right default for production: the plan is reused
~70,000 times, so seconds of planning are free. Use `FFTW.ESTIMATE` in tests.
Persist planner effort across runs with [`save_wisdom`](@ref) /
[`load_wisdom!`](@ref).

!!! warning "Plan once, single-threaded"
    FFTW's planner mutates global state. Build every Workspace up front on one
    thread, then hand them to tasks. Executing a plan is thread-safe; creating
    one is not.
"""
function workspace(backend::Backend, spec::GridSpec{T}, nterms::Integer;
                   batch::Integer = 1, flags = FFTW.MEASURE) where {T}
    nx, ny, nz = spec.n
    nxr = speclen(spec)
    B = Int(batch)

    lig_real = allocate(backend, T, nx, ny, nz, B)
    lig_spec = allocate(backend, Complex{T}, nxr, ny, nz, B)
    accum    = allocate(backend, Complex{T}, nxr, ny, nz, B)
    score    = allocate(backend, T, nx, ny, nz, B)

    fwd = FFTW_plan_rfft(lig_real, flags)
    bwd = FFTW_plan_brfft(accum, nx, flags)

    return Workspace{T,typeof(lig_real),typeof(lig_spec),typeof(fwd),typeof(bwd)}(
        spec, Int(nterms), B, lig_real, lig_spec, accum, score, fwd, bwd)
end

"""Bytes of device/host memory held by one Workspace."""
function workspace_bytes(ws::Workspace{T}) where {T}
    sizeof(T) * (length(ws.lig_real) + length(ws.score)) +
    2 * sizeof(T) * (length(ws.lig_spec) + length(ws.accum))
end

save_wisdom(path::AbstractString) = FFTW.export_wisdom(path)
load_wisdom!(path::AbstractString) = isfile(path) ? (FFTW.import_wisdom(path); true) : false


include("vram.jl")


"""
    _accumulate!(acc, Rp, lig, ngrid)

`acc[off+i] += Rp[i] * lig[off+i]` for every batch slice, where `Rp` is a single
3-D receptor spectrum broadcast across the batch.

Four array touches per element (read `acc`, read `Rp`, read `lig`, write `acc`)
and one complex fused multiply-add: pure memory bandwidth.

**Measured:** this stage is **30-38% of per-rotation runtime** -- comparable to
the forward FFTs and far more than voxelisation. It is therefore worth a hand-
written kernel. Julia's `Complex{T}` broadcast (`acc .+= Rp .* lig`) achieves
only ~15 GB/s here; an explicit loop doing the complex multiply-add in real
components reaches 31-49 GB/s:

    Float32 N=96   15.9 -> 48.7 GB/s
    Float32 N=128  15.5 -> 37.9 GB/s
    Float64 N=128  21.5 -> 30.8 GB/s

The broadcast is presumably failing to vectorise complex arithmetic; the loop
below is the same arithmetic written so that LLVM can see it.

The generic method keeps the broadcast, which is correct everywhere and is the
right choice on GPU (where it becomes a single fused kernel and a scalar loop
would be catastrophic). Both paths are covered by `test_backends.jl`.
"""
function _accumulate!(acc::Array{Complex{T},4}, Rp::Array{Complex{T},3},
                      lig::Array{Complex{T},4}, ngrid::Int) where {T}
    B = length(acc) ÷ ngrid
    @inbounds for b in 0:B-1
        off = b * ngrid
        @simd for i in 1:ngrid
            rc = Rp[i]; lc = lig[off+i]; ac = acc[off+i]
            acc[off+i] = Complex(
                muladd(real(rc), real(lc), muladd(-imag(rc), imag(lc), real(ac))),
                muladd(real(rc), imag(lc), muladd( imag(rc), real(lc), imag(ac))))
        end
    end
    return acc
end

function _accumulate!(acc::AbstractArray{Complex{T},4}, Rp::AbstractArray{Complex{T},3},
                      lig::AbstractArray{Complex{T},4}, ngrid::Int) where {T}
    acc .+= reshape(Rp, size(Rp)..., 1) .* lig
    return acc
end

"""
    _store!(acc, Rp, lig, ngrid)

`acc[off+i] = Rp[i] * lig[off+i]` -- the first term of the accumulation, written
rather than accumulated.

Using this for `p == 1` removes both the `fill!(acc, 0)` pass and the read of
`acc` in the first `_accumulate!`. At 128^3 that is ~16 MiB of memory traffic
saved per rotation, on a stage that is already bandwidth-bound.
"""
function _store!(acc::Array{Complex{T},4}, Rp::Array{Complex{T},3},
                 lig::Array{Complex{T},4}, ngrid::Int) where {T}
    B = length(acc) ÷ ngrid
    @inbounds for b in 0:B-1
        off = b * ngrid
        @simd for i in 1:ngrid
            rc = Rp[i]; lc = lig[off+i]
            acc[off+i] = Complex(muladd(real(rc), real(lc), -imag(rc) * imag(lc)),
                                 muladd(real(rc), imag(lc),  imag(rc) * real(lc)))
        end
    end
    return acc
end

function _store!(acc::AbstractArray{Complex{T},4}, Rp::AbstractArray{Complex{T},3},
                 lig::AbstractArray{Complex{T},4}, ngrid::Int) where {T}
    acc .= reshape(Rp, size(Rp)..., 1) .* lig
    return acc
end

"""
    correlate!(ws, ligand, rots, recspec) -> ws.score

Evaluate the full PIPER energy landscape for every rotation in `rots`, for all
translations simultaneously. Returns `(nx, ny, nz, B)`; entry `[a,b,c,r]` is the
energy of rotation `rots[r]` translated by `(a-1, b-1, c-1)` cells.

**This function is identical for CPU and GPU.** `Array` in, FFTW runs; `CuArray`
in, cuFFT runs. That is the whole point of the design -- multiple dispatch
resolves `mul!` and `_accumulate!` to the right implementation, and no branch,
wrapper, or `#ifdef` appears anywhere in the numerical code.

Transform count is `nterms + 1`, not `2 * nterms`: the per-term products are
summed in the Fourier domain and inverted once.

Allocation-free after warm-up; `test/test_allocations.jl` asserts it rather than
assuming it.
"""
function correlate!(ws::Workspace{T}, lig::Structure, rots::AbstractVector,
                    rs::ReceptorSpectra{T}) where {T}
    ws.nterms == nterms(rs) ||
        throw(DimensionMismatch("workspace has $(ws.nterms) terms, receptor has $(nterms(rs))"))
    ngrid = length(ws.accum) ÷ ws.batch
    @inbounds for p in 1:ws.nterms
        voxelize!(ws.lig_real, lig, rots, p, ws.spec)
        mul!(ws.lig_spec, ws.fwd, ws.lig_real)

        if p == 1
            _store!(ws.accum, rs.data[p], ws.lig_spec, ngrid)
        else
            _accumulate!(ws.accum, rs.data[p], ws.lig_spec, ngrid)
        end
    end

    mul!(ws.score, ws.bwd, ws.accum)
    return ws.score
end


const SCAN_GROUPSIZES = (32, 64, 128, 256)

"""Workgroup size for a slab of `nxy` cells: the smallest specialisation that
keeps every lane fed, capped at the largest one we compile."""
function _scan_groupsize(nxy::Integer)
    gs = SCAN_GROUPSIZES[1]
    for g in SCAN_GROUPSIZES
        g <= nxy && (gs = g)
    end
    return gs
end


@inline function _better(v1, i1, v2, i2)
    return v1 < v2 || (v1 == v2 && i1 < i2)
end

@kernel function _slab_min_kernel!(vals, idxs, @Const(score), nx::Int, ny::Int,
                                   nz::Int, ::Val{GS}) where {GS}
    k, b = @index(Group, NTuple)
    lane = @index(Local, Linear)

    T = @uniform eltype(score)
    sv = @localmem T (GS,)
    si = @localmem Int32 (GS,)

    nxy = nx * ny
    slabbase = nxy * (k - 1)
    base = slabbase + nxy * nz * (b - 1)

    best = typemax(T)
    besti = Int32(1)
    q = lane
    @inbounds while q <= nxy
        v = score[base+q]
        if v < best
            best = v
            besti = Int32(slabbase + q)
        end
        q += GS
    end
    @inbounds sv[lane] = best
    @inbounds si[lane] = besti
    @synchronize()

    s = @uniform GS >> 1
    while s > 0
        @inbounds if lane <= s && _better(sv[lane+s], si[lane+s], sv[lane], si[lane])
            sv[lane] = sv[lane+s]
            si[lane] = si[lane+s]
        end
        @synchronize()
        s = @uniform s >> 1
    end

    if lane == 1
        @inbounds vals[k, b] = sv[1]
        @inbounds idxs[k, b] = si[1]
    end
end

"""
    scan_best!(vals, idxs, score) -> (vals, idxs)

Lowest-energy translation per rotation. `vals`/`idxs` are `(nz, B)` scratch;
the caller reduces the `nz` partials.

The generic method is a **workgroup per slab** with a shared-memory tree reduction: `nz
* B * GS` work items rather than `nz * B`, a 32-256x increase in exposed parallelism for
the same output.

Both stages are ordinary KernelAbstractions, so this is still one kernel for two
devices: `@localmem` lowers to `CuStaticSharedArray` on CUDA and to a plain
array on the CPU backend, which is what lets `test_backends.jl` check it here
against the serial method. Note `@cuStaticSharedMem` -- the spelling a CUDA-only
version would reach for -- is deprecated in CUDA.jl 6.x in favour of
`CuStaticSharedArray`; going through KA sidesteps the question entirely.

Each rotation retains one minimum-energy translation. PIPER groups translations
within 10 Å cubes and may retain several minima per rotation (preprint p.8).
"""
function scan_best!(vals::Matrix{T}, idxs::Matrix{<:Integer},
                    score::Array{T,4}) where {T}
    nx, ny, nz, B = size(score)
    size(vals) == (nz, B) || throw(DimensionMismatch("vals must be (nz, B)"))
    @inbounds for b in 1:B, k in 1:nz
        best = typemax(T); bi = 1
        for j in 1:ny, i in 1:nx
            v = score[i, j, k, b]
            if v < best
                best = v
                bi = i + nx * (j - 1) + nx * ny * (k - 1)
            end
        end
        vals[k, b] = best
        idxs[k, b] = bi
    end
    return (vals, idxs)
end

function scan_best!(vals::AbstractMatrix{T}, idxs::AbstractMatrix{<:Integer},
                    score::AbstractArray{T,4}) where {T}
    nx, ny, nz, B = size(score)
    size(vals) == (nz, B) || throw(DimensionMismatch("vals must be (nz, B)"))
    backend = get_backend(score)

    gs = _scan_groupsize(nx * ny)
    _launch_slab_min!(backend, Val(gs), vals, idxs, score, nx, ny, nz, B)
    synchronize(backend)
    return (vals, idxs)
end


function _launch_slab_min!(backend, ::Val{GS}, vals, idxs, score,
                           nx, ny, nz, B) where {GS}
    kern = _slab_min_kernel!(backend, (GS, 1))
    kern(vals, idxs, score, nx, ny, nz, Val(GS); ndrange = (nz * GS, B))
    return nothing
end

"""Cartesian cell offset (0-based) of a linear grid index."""
function unravel(spec::GridSpec, lin::Integer)
    nx, ny, _ = spec.n
    z, r = divrem(lin - 1, nx * ny)
    y, x = divrem(r, nx)
    return (x, y, z)
end


"""
    Pose{T}

One scored rigid placement: rotation index, integer cell translation, energy.
Kept deliberately small and `isbitstype` so a `Vector{Pose}` is a flat buffer.
"""
struct Pose{T}
    energy::T
    rot::Int32
    tx::Int16
    ty::Int16
    tz::Int16
end

"""
    dock!(poses, ligand, rots, recspec, pool) -> poses

Run every rotation and record the best translation of each. `pool` is a vector
of Workspaces, one per concurrent task -- **never** index a buffer pool by
`Threads.threadid()`. Since Julia 1.7 tasks migrate between threads at any yield
point, so `threadid()` is not stable for the lifetime of a task and two tasks can
observe the same id. Binding one workspace per *task* before the spawn, as here,
is the only pattern that is correct by construction.

Set `FFTW.set_num_threads(1)` and `LinearAlgebra.BLAS.set_num_threads(1)` before
calling: parallelism belongs to the rotation loop, and nested FFTW/BLAS threads
oversubscribe the machine.

`guards` is an optional vector of [`MemoryGuard`](@ref)s, **one per pool entry**
for exactly the reason the workspaces are: a `MemoryGuard` is mutable, so a
single shared guard would be written by every task concurrently -- the same race
the pool discipline above exists to prevent. The default `nothing` is not a
degraded mode: the rotation loop allocates zero bytes, so the derived check
interval is infinite and a guard would never fire (see `src/vram.jl`). Passing
`nothing` lets Julia specialise the call away entirely, so the default path
carries no branch at all.
"""
function dock!(poses::AbstractVector{Pose{T}}, lig::Structure, rots::AbstractVector,
               rs::ReceptorSpectra{T}, pool::AbstractVector{<:Workspace{T}};
               guards::Union{Nothing,AbstractVector{<:MemoryGuard}} = nothing) where {T}
    nrot = length(rots)
    length(poses) == nrot || throw(DimensionMismatch("one pose slot per rotation"))
    nchunks = length(pool)
    B = first(pool).batch
    all(w -> w.batch == B, pool) || throw(ArgumentError("pool must be uniform in batch size"))

    nrot % B == 0 ||
        throw(ArgumentError("length(rots)=$nrot must be a multiple of batch=$B"))
    guards === nothing || length(guards) == nchunks ||
        throw(ArgumentError("guards must be one per pool entry ($(nchunks)), " *
                            "got $(length(guards)) -- a shared guard is a data race"))

    csize = B * cld(cld(nrot, nchunks), B)
    chunks = collect(Iterators.partition(1:nrot, csize))
    @assert length(chunks) <= nchunks "chunking invariant violated: would share workspaces"

    @sync for (ic, chunk) in enumerate(chunks)
        ws = pool[ic]
        gd = guards === nothing ? nothing : guards[ic]
        Threads.@spawn _dock_chunk!(poses, lig, rots, rs, ws, chunk, gd)
    end
    return poses
end

"""
    _host_mirror(a) -> Array

`a` itself when it already lives on the host, a host copy of it otherwise.

The `nz`-way reduction at the end of [`_dock_chunk!`](@ref) is a scalar loop over
`nz * batch` partials, so it has to run somewhere it can index. Returning the
argument unchanged on the CPU is what keeps that path copy-free -- the caller
tests `!==` rather than branching on a backend.
"""
_host_mirror(a::Array) = a
_host_mirror(a::AbstractArray) = Array(a)

function _dock_chunk!(poses::AbstractVector{Pose{T}}, lig, rots, rs,
                      ws::Workspace{T}, chunk, guard) where {T}
    nz = ws.spec.n[3]

    vals = similar(ws.score, T, nz, ws.batch)
    idxs = similar(ws.score, Int32, nz, ws.batch)

    hvals = _host_mirror(vals)
    hidxs = _host_mirror(idxs)
    for base in Iterators.partition(chunk, ws.batch)
        rb = @view rots[base]
        nb = length(rb)
        @assert nb == ws.batch "ragged batch reached the inner loop"

        maybe_collect!(guard)
        correlate!(ws, lig, rb, rs)
        scan_best!(vals, idxs, ws.score)
        if hvals !== vals
            copyto!(hvals, vals)
            copyto!(hidxs, idxs)
        end

        @inbounds for b in 1:nb
            best = typemax(T); bi = 1
            for k in 1:nz
                if hvals[k, b] < best
                    best = hvals[k, b]; bi = Int(hidxs[k, b])
                end
            end
            x, y, z = unravel(ws.spec, bi)
            poses[base[b]] = Pose{T}(best, Int32(base[b]), Int16(x), Int16(y), Int16(z))
        end
    end
    return nothing
end


"""
    pairwise_rmsd!(D, rots, trans, S, M) -> D

Full `N x N` matrix of ligand RMSDs between rigid poses, in **O(N^2)** flops
total rather than O(N^2 * natoms).

Every pose is the same rigid body under a different `(R_i, t_i)`, and docking
RMSD is computed without re-superposition (the receptor frame is fixed -- the
PIPER paper defines it exactly this way, PDF p.22). So with ligand coordinates
`y_k` centred (`sum_k y_k = 0`) and `S = sum_k y_k y_k'`:

    RMSD^2(i,j) = ||t_i - t_j||^2 + (2/M) * ( tr(S) - tr(R_i' R_j S) )

Expanding `sum_k ||(R_i - R_j) y_k + dt||^2`: the cross term vanishes by
centring, and `(R_i - R_j)'(R_i - R_j) = 2I - Q - Q'` with `Q = R_i' R_j`.

The rotation term is bilinear in `vec(R)`, so the whole matrix is two GEMMs:
`G = A * kron(S, I3) * A'` with `A[i, :] = vec(R_i)'`. For N = 1000 poses of a
2000-atom ligand that is ~18 MFLOP instead of ~16 GFLOP.

`M` is the atom count the RMSD is defined over; `S` must be built from the same
atom subset. Interface-RMSD works too **provided the subset is fixed across
poses** -- if the interface is redefined per pose, this identity does not hold
and you must fall back to the direct computation.

!!! note "Accuracy floor"
    The identity evaluates `RMSD^2` as a difference of quantities of order
    `tr(S) ~ M*Rg^2`, so it loses absolute precision to cancellation:

        err(RMSD) ~ sqrt(2*eps(T)*tr(S)/M) = Rg * sqrt(2*eps(T))

    This is **independent of the RMSD value**, so it is worst where the RMSD is
    smallest. Measured: ~1.2e-6 A for a ligand with `Rg ~ 31 A` in Float64,
    concentrated entirely on near-identical pose pairs. Seven orders of
    magnitude below a 9 A clustering radius, so irrelevant for clustering -- but
    do not use this function to resolve sub-micro-angstrom differences, and do
    not run it in Float32 (where the floor would be ~0.5 A and useless).
"""
function pairwise_rmsd!(D::AbstractMatrix{T}, rots::AbstractVector{<:SMatrix{3,3,T}},
                        trans::AbstractVector{<:SVector{3,T}},
                        S::SMatrix{3,3,T}, M::Integer) where {T}
    Base.require_one_based_indexing(D, rots, trans)
    M > 0 || throw(ArgumentError("M must be a positive atom count"))
    N = length(rots)
    length(trans) == N || throw(DimensionMismatch("one translation per rotation required"))
    size(D) == (N, N) || throw(DimensionMismatch("D must be N x N"))
    A = Matrix{T}(undef, N, 9)
    @inbounds for i in 1:N
        R = rots[i]
        for c in 1:3, r in 1:3
            A[i, r+3*(c-1)] = R[r, c]
        end
    end
    Kmat = kron(Matrix{T}(S), Matrix{T}(LinearAlgebra.I, 3, 3))
    G = A * Kmat * A'
    trS = tr(S)
    invM = one(T) / T(M)
    @inbounds for j in 1:N, i in 1:N
        dt = trans[i] - trans[j]
        r2 = dot(dt, dt) + 2 * invM * (trS - G[i, j])
        D[i, j] = sqrt(max(zero(T), r2))
    end
    return D
end

"""
    second_moment(coords) -> SMatrix{3,3}

`S = sum_k y_k y_k'` for centred coordinates -- the only ligand-dependent input
`pairwise_rmsd!` needs.
"""
function second_moment(coords::AbstractVector{<:SVector{3,T}}) where {T}
    S = zero(SMatrix{3,3,T,9})
    for y in coords
        S += y * y'
    end
    return S
end

"""
    greedy_cluster(D, radius; maxclusters=30) -> Vector{Vector{Int}}

ClusPro-style greedy clustering: repeatedly take the pose with the most
unassigned neighbours within `radius`, emit it and its neighbours as a cluster,
remove them, repeat. Cluster *population*, not energy, is the ranking signal --
that is the whole premise of the ClusPro discrimination step.

**The cluster centre is `members[1]`.** Its neighbour count is measured among
the poses still available when the cluster is selected. The centre is included
in its cluster even when a caller-supplied dissimilarity has `D[i,i] > radius`.
"""
function greedy_cluster(D::AbstractMatrix{T}, radius::Real; maxclusters::Integer = 30) where {T}
    Base.require_one_based_indexing(D)
    N = size(D, 1)
    size(D, 2) == N || throw(DimensionMismatch("D must be square"))
    alive = trues(N)
    clusters = Vector{Vector{Int}}()
    r = T(radius)
    while length(clusters) < maxclusters && any(alive)
        bestc, bestn = 0, -1
        @inbounds for i in 1:N
            alive[i] || continue
            c = 0
            for j in 1:N
                (alive[j] && D[i, j] <= r) && (c += 1)
            end
            c > bestn && (bestn = c; bestc = i)
        end
        bestc == 0 && break
        members = Int[bestc]
        alive[bestc] = false
        @inbounds for j in 1:N
            if alive[j] && D[bestc, j] <= r
                push!(members, j)
                alive[j] = false
            end
        end
        push!(clusters, members)
    end
    return clusters
end


include("rotations.jl")
include("shape.jl")


include("pdb.jl")
include("acp.jl")


include("voxel.jl")


include("mask.jl")
include("physics.jl")


include("scoring.jl")


include("validate.jl")


include("fibre.jl")


include("restraints.jl")


include("multimer.jl")


@setup_workload begin
    T = Float32
    coords = [SVector{3,T}(randn(), randn(), randn()) .* 4 for _ in 1:32]
    W = rand(T, 32, 3)
    @compile_workload begin
        s = Structure(coords, W)
        spec = GridSpec{T}((16, 16, 16), T(1.2), SVector{3,T}(-9.6, -9.6, -9.6))
        rs = receptor_spectra(CPUBackend(), s, spec, ones(T, 3); flags = FFTW.ESTIMATE)
        ws = workspace(CPUBackend(), spec, 3; batch = 1, flags = FFTW.ESTIMATE)
        rots = [one(SMatrix{3,3,T,9})]
        correlate!(ws, s, rots, rs)
        vals = Matrix{T}(undef, 16, 1)
        idxs = Matrix{Int32}(undef, 16, 1)
        scan_best!(vals, idxs, ws.score)

        radii = fill(T(1.7), 32)
        sspec = shape_spectra(CPUBackend(), coords, radii, spec; flags = FFTW.ESTIMATE)
        sws = workspace(CPUBackend(), spec, 1; batch = 1, flags = FFTW.ESTIMATE)
        correlate!(sws, shape_structure(coords), rots, sspec)

        rset = rotations_hopf(T, 4, 4)
        covering_bound(length(rset))
        rotangle(rset.quats[1], rset.quats[2])
    end
end

end
