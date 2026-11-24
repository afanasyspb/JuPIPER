"""Attractive-shell cutoff `D`, in angstrom (PDF p.10). Also the pairwise
contact radius (p.13) and, doubled, the electrostatic truncation (p.11)."""
const PIPER_D = 6.0

"""Repulsive shell is `r_vdw + PIPER_DPAD` (PDF p.10)."""
const PIPER_DPAD = 2.0

"""Edge of the cube the electrostatic field is smoothed with, in angstrom.
The paper calls this smoothing "very important" (PDF p.11)."""
const PIPER_SMOOTH_BOX = 3.0

"""Grid spacing the paper uses, in angstrom (PDF p.13)."""
const PIPER_SPACING = 1.2


"""
    box_taps(T, box, spacing) -> (offsets, weights)

Separable normalised taps for convolution with a cube of edge `box` angstrom on
a grid of the given `spacing`. Returns integer cell offsets and matching
weights that sum to 1.

The taps are **area-weighted**, not rounded: a 3 A box on a 1.2 A grid is 2.5
cells wide, which no odd cell count represents, so the tap at offset `k` gets
the fraction of cell `k` the box actually covers. That gives `[0.3, 0.4, 0.3]`
at PIPER's own parameters -- exact, and free of the arbitrary choice between a
1-cell box (too sharp) and a 3-cell one (3.6 A, 20% too wide).

A box narrower than one cell yields a single unit tap: a filter that cannot be
resolved by the grid is the identity, which is the honest answer rather than an
error.
"""
function box_taps(::Type{T}, box::Real, spacing::Real) where {T<:AbstractFloat}
    lam = T(box) / T(spacing)
    lam > 0 || throw(ArgumentError("box edge must be positive, got $box"))
    K = max(0, ceil(Int, (lam - 1) / 2))
    offs = collect(-K:K)
    half = lam / 2
    wts = T[max(zero(T), min(T(k) + T(0.5), half) - max(T(k) - T(0.5), -half)) / lam
            for k in offs]
    return (offs, wts)
end

"""
    BoxFilter{T}

Precomputed taps plus the line scratch for [`box_smooth!`](@ref). Built once per
grid by [`box_filter`](@ref) and reused, for the same reason an FFT plan is:
`box_taps` allocates two small vectors, and a kernel that allocates cannot be
asserted allocation-free by `test_physics.jl` no matter how small the constant.

`edge` is retained so a smoothed field can say what smoothed it.
"""
struct BoxFilter{T<:AbstractFloat}
    edge::T
    offs::Vector{Int}
    wts::Vector{T}
    line::Vector{T}
end

Base.show(io::IO, f::BoxFilter{T}) where {T} =
    print(io, "BoxFilter{$T}(", f.edge, " A, ", length(f.offs), " taps)")

"""
    box_filter(T, spec; edge=PIPER_SMOOTH_BOX) -> BoxFilter

Build the smoothing filter for `spec`. Allocates; setup-time. Pass the result to
[`gb_receptor_grid!`](@ref) or [`box_smooth!`](@ref); pass `nothing` instead to
skip smoothing, which the paper does not do and calls "very important" not to
(PDF p.11).
"""
function box_filter(::Type{T}, spec::GridSpec{T};
                    edge::Real = PIPER_SMOOTH_BOX) where {T}
    offs, wts = box_taps(T, edge, spec.spacing)
    return BoxFilter{T}(T(edge), offs, wts, Vector{T}(undef, maximum(spec.n)))
end
box_filter(spec::GridSpec{T}; kwargs...) where {T} = box_filter(T, spec; kwargs...)

"""
    box_smooth!(A, f::BoxFilter) -> A
    box_smooth!(A, spec; edge=PIPER_SMOOTH_BOX) -> A   # convenience: allocates

Convolve `A` in place with a cube of edge `f.edge` angstrom, separably, one axis
at a time. Mutates `A`. The `BoxFilter` method allocates nothing.

Outside the grid the field is taken as **zero**, so a nonzero field at a face
is attenuated within one tap of it. That is the correct reading for a truncated,
decaying potential; it is also the reason [`grid_fit`](@ref)'s `margin_angstrom`
is worth looking at before trusting values near the boundary.
"""
function box_smooth!(A::AbstractArray{T,3}, f::BoxFilter{T}) where {T}
    offs, wts, line = f.offs, f.wts, f.line
    length(offs) == 1 && return A
    nx, ny, nz = size(A)
    length(line) >= max(nx, ny, nz) ||
        throw(DimensionMismatch("line buffer holds $(length(line)), need $(max(nx,ny,nz))"))
    nt = length(offs)
    @inbounds begin
        for k in 1:nz, j in 1:ny
            for i in 1:nx; line[i] = A[i, j, k]; end
            for i in 1:nx
                acc = zero(T)
                for t in 1:nt
                    s = i + offs[t]
                    1 <= s <= nx && (acc += wts[t] * line[s])
                end
                A[i, j, k] = acc
            end
        end
        for k in 1:nz, i in 1:nx
            for j in 1:ny; line[j] = A[i, j, k]; end
            for j in 1:ny
                acc = zero(T)
                for t in 1:nt
                    s = j + offs[t]
                    1 <= s <= ny && (acc += wts[t] * line[s])
                end
                A[i, j, k] = acc
            end
        end
        for j in 1:ny, i in 1:nx
            for k in 1:nz; line[k] = A[i, j, k]; end
            for k in 1:nz
                acc = zero(T)
                for t in 1:nt
                    s = k + offs[t]
                    1 <= s <= nz && (acc += wts[t] * line[s])
                end
                A[i, j, k] = acc
            end
        end
    end
    return A
end


box_smooth!(A::AbstractArray{T,3}, spec::GridSpec{T};
            edge::Real = PIPER_SMOOTH_BOX) where {T} =
    box_smooth!(A, box_filter(T, spec; edge = edge))


"""
    GBTruncation

Which reading of the preprint's electrostatic cutoff to apply. **The preprint is
ambiguous here and the two readings do not agree**, so the choice is an argument rather
than a constant.

The printed formula (PDF p.11) is

    R(l,m,n) = sum_i  q_i / ( rhat^2 + D^2 exp(-rhat^2 / 4D^2) )^(1/2)
    rhat     = max(r, 2D)

and the prose beside it says "the potential is truncated at the distance 2D".
Taken literally, `max` makes `rhat` constant for every atom *closer* than 12 A
and lets the field decay only *beyond* it -- the radial dependence inverted, the
near field flattened, and the term rendered meaningless. That cannot be what was
computed, so one of the two statements is a typo and neither of the surviving
readings is licensed over the other:

  * [`GBCapped`](@ref) -- `rhat = min(r, 2D)`, i.e. the *distance* is truncated.
    Continuous at `r = 2D`, which is the only reason to write a clamp at all,
    and it is the printed formula with one character changed. **Default.**
  * [`GBTruncated`](@ref) -- atoms beyond `2D` contribute nothing, the ordinary
    meaning of a truncated potential. Discontinuous at the cutoff by
    `q_i / den(2D)`, about `0.08 q_i`.

They are not interchangeable, and the difference is **not** a constant offset:

    R_capped(cell) - R_trunc(cell) = ( Q_total - sum_{r<2D} q_i ) / den(2D)

whose second term varies from cell to cell. A translation-invariant difference would
cancel out of every ranking; this one does not.
"""
abstract type GBTruncation end

"""`rhat = min(r, 2D)`: the distance is clamped, the field is continuous. Default."""
struct GBCapped <: GBTruncation end

"""Atoms beyond `2D` are dropped: the ordinary truncated potential, discontinuous."""
struct GBTruncated <: GBTruncation end


@inline _gb_shift(::GBCapped, inv_den_cut::T) where {T} = inv_den_cut
@inline _gb_shift(::GBTruncated, ::T) where {T} = zero(T)
@inline _gb_offset(::GBCapped, q_total::T, inv_den_cut::T) where {T} = q_total * inv_den_cut
@inline _gb_offset(::GBTruncated, ::T, ::T) where {T} = zero(T)

"""
    gb_receptor_grid!(R, coords, charges, spec; D, truncation, smooth) -> R

Fill `R` with the receptor's Generalized Born electrostatic potential field
(PDF p.11), then smooth it with `smooth`. Mutates and fully overwrites `R`,
which must be `(nx, ny, nz)`.

    R(l,m,n) = sum_i  q_i / ( rhat^2 + D^2 exp(-rhat^2 / 4D^2) )^(1/2)

with `D` the constant Born radius -- the paper reuses the shape term's 6 A for
it -- and `rhat` set by `truncation`, whose docstring is where the ambiguity in
the source is written down. Pass `smooth = nothing` to skip smoothing; the paper
does not, and calls it "very important" because the unsmoothed field's minima
move with the atomic coordinates.

Allocates nothing when handed a [`BoxFilter`](@ref); the default builds one per
call. Setup-time: `O(natoms * (4D/h)^3)`, about 2e7 cell visits for a 2000-atom
receptor at 1.2 A.
"""
function gb_receptor_grid!(R::AbstractArray{T,3},
                           coords::AbstractVector{SVector{3,T}},
                           charges::AbstractVector{T},
                           spec::GridSpec{T};
                           D::Real = PIPER_D,
                           truncation::GBTruncation = GBCapped(),
                           smooth::Union{Nothing,BoxFilter{T}} =
                               box_filter(T, spec)) where {T}
    length(coords) == length(charges) ||
        throw(DimensionMismatch("one charge per atom required"))
    nx, ny, nz = spec.n
    size(R) == (nx, ny, nz) || throw(DimensionMismatch("R must be $(spec.n)"))
    Dt = T(D)
    D2 = Dt * Dt
    inv4D2 = inv(4 * D2)
    rcut = 2 * Dt
    rcut2 = rcut * rcut
    inv_den_cut = inv(sqrt(rcut2 + D2 * exp(-rcut2 * inv4D2)))
    shift = _gb_shift(truncation, inv_den_cut)
    q_total = sum(charges; init = zero(T))
    fill!(R, _gb_offset(truncation, q_total, inv_den_cut))

    h = spec.spacing
    inv_h = inv(h)
    rc = rcut * inv_h
    origin = spec.origin
    @inbounds for a in eachindex(coords)
        q = charges[a]
        iszero(q) && continue
        g = (coords[a] - origin) * inv_h
        lo1 = max(1, ceil(Int, g[1] - rc) + 1); hi1 = min(nx, floor(Int, g[1] + rc) + 1)
        lo2 = max(1, ceil(Int, g[2] - rc) + 1); hi2 = min(ny, floor(Int, g[2] + rc) + 1)
        lo3 = max(1, ceil(Int, g[3] - rc) + 1); hi3 = min(nz, floor(Int, g[3] + rc) + 1)
        for n in lo3:hi3
            dz = ((n - 1) - g[3]) * h
            dz2 = dz * dz
            dz2 > rcut2 && continue
            for m in lo2:hi2
                dy = ((m - 1) - g[2]) * h
                dyz2 = dz2 + dy * dy
                dyz2 > rcut2 && continue
                @simd for l in lo1:hi1
                    dx = ((l - 1) - g[1]) * h
                    r2 = dyz2 + dx * dx

                    inv_den = inv(sqrt(r2 + D2 * exp(-r2 * inv4D2)))
                    R[l, m, n] += ifelse(r2 < rcut2, q * (inv_den - shift), zero(T))
                end
            end
        end
    end
    smooth === nothing || box_smooth!(R, smooth)
    return R
end


"""
    dars_receptor_grid!(R, coords, u, spec; D=PIPER_D) -> R

Fill `R` with the `p`-th pairwise receptor field (PDF p.13)

    R_p(l,m,n) = sum_i u_pI * delta_i,   delta_i = 1 if |cell - atom_i| < D

where `u[i]` is the `p`-th eigenvector component for the *type* of atom `i` --
already resolved per atom by the caller, which keeps this kernel free of any
typing concern. Mutates and fully overwrites `R`. Allocation-free; setup-time.

Note the asymmetry with the definition it discretises. `E_pair` sums `eps_ij`
over pairs with `d < r_ij < D` (PDF p.12), an annulus; the grid form on p.13
uses `r < D`, a ball, dropping the inner exclusion. The exclusion is partly
recovered on the ligand side, where `L_p` is nonzero only where a *ligand* atom
sits -- but only partly. This reproduces p.13 as printed, because the grid form
is the thing the reference actually evaluates.
"""
function dars_receptor_grid!(R::AbstractArray{T,3},
                             coords::AbstractVector{SVector{3,T}},
                             u::AbstractVector{T},
                             spec::GridSpec{T}; D::Real = PIPER_D) where {T}
    length(coords) == length(u) ||
        throw(DimensionMismatch("one eigenvector component per atom required"))
    nx, ny, nz = spec.n
    size(R) == (nx, ny, nz) || throw(DimensionMismatch("R must be $(spec.n)"))
    fill!(R, zero(T))
    Dt = T(D)
    D2 = Dt * Dt
    h = spec.spacing
    inv_h = inv(h)
    Dc = Dt * inv_h
    origin = spec.origin
    @inbounds for a in eachindex(coords)
        ua = u[a]
        g = (coords[a] - origin) * inv_h
        lo1 = max(1, ceil(Int, g[1] - Dc) + 1); hi1 = min(nx, floor(Int, g[1] + Dc) + 1)
        lo2 = max(1, ceil(Int, g[2] - Dc) + 1); hi2 = min(ny, floor(Int, g[2] + Dc) + 1)
        lo3 = max(1, ceil(Int, g[3] - Dc) + 1); hi3 = min(nz, floor(Int, g[3] + Dc) + 1)
        for n in lo3:hi3
            dz = ((n - 1) - g[3]) * h
            dz2 = dz * dz
            dz2 > D2 && continue
            for m in lo2:hi2
                dy = ((m - 1) - g[2]) * h
                dyz2 = dz2 + dy * dy
                dyz2 > D2 && continue
                @simd for l in lo1:hi1
                    dx = ((l - 1) - g[1]) * h
                    r2 = dyz2 + dx * dx
                    R[l, m, n] += ifelse(r2 < D2, ua, zero(T))
                end
            end
        end
    end
    return R
end


"""
    type_abundance(tbl; T=Float64) -> Vector{T}

Normalised frequency of each ACP type over the 167 standard heavy atoms, i.e.
under **uniform residue composition**. `MSE` is excluded so the vector counts
each chemical position once.

Its use is as the `weights` argument of [`truncation_error`](@ref): an unweighted
Frobenius norm treats a type that occurs twice in the 20 residues as equal in importance
to one that occurs twenty-eight times. This is a better model and still not the right
one -- the right one weights by *interface contact* frequency, which is surface-biased
and which this repository does not have.
"""
function type_abundance(tbl::ACPTypeTable; T::Type = Float64)
    w = zeros(T, length(tbl.labels))
    for (res, d) in tbl.byres
        res == "MSE" && continue
        for (_, t) in d
            w[Int(t)] += one(T)
        end
    end
    s = sum(w)
    s > 0 && (w ./= s)
    return w
end

"""
    dars_rank(pot; tol=0.10, weights=nothing, Pmax=size(pot.eps,1)) -> Int

Smallest `P` whose rank-`P` truncation of `pot` has relative Frobenius error at
most `tol`; `Pmax` if none does.

`tol` defaults to 0.10 because that is the figure the preprint itself uses to
justify truncating at all: "restricting consideration to a grid may yield up to
10 % error in the energy values, and hence it is well justified to truncate the
summation when the energy contributions of the neglected terms are comparable
to this error" (PDF p.13).

**On the DARS matrix this returns 5, not the 4 a reader of the preprint would expect.**
The "only 2 to 4 eigenvalues are needed" sentence on the same page is stated about *ACP
and RPScore* -- two potentials this repository does not hold -- and not about DARS,
which is the paper's own new matrix. Measured here, rank 4 leaves 14.5% (16.4%
abundance-weighted).
"""
function dars_rank(pot::Potential{T}; tol::Real = 0.10,
                   weights::Union{Nothing,AbstractVector} = nothing,
                   Pmax::Integer = size(pot.eps, 1)) where {T}
    for P in 1:Pmax
        _, rel = truncation_error(pot, P; weights = weights)
        rel <= tol && return P
    end
    return Int(Pmax)
end


"""
    receptor_weights(pot, w; P) -> Vector

The scalar folded into each receptor spectrum, in `build_termweights`' term
order: `[1, w2, w3*lambda_1, …, w3*lambda_P]`.

Term 1 carries 1, not `w1`: `w1` is already inside the shape *field*
(`R = -c + w1 r`), so folding it again would square it. Pairing this with a
ligand `Structure` built by [`piper_structure`](@ref) reproduces
`E = E_shape + w2 E_elec + w3 E_pair` exactly.
"""
function receptor_weights(pot::Potential{T}, w::PiperWeights{T}; P::Integer) where {T}
    1 <= P <= length(pot.lambda) ||
        throw(ArgumentError("P must be in 1:$(length(pot.lambda)), got $P"))
    v = Vector{T}(undef, 2 + P)
    v[1] = one(T)
    v[2] = w.w2
    @inbounds for p in 1:P
        v[2+p] = w.w3 * pot.lambda[p]
    end
    return v
end

"""
    piper_receptor_grids!(grids, at, par, pot, spec; kwargs...) -> grids

Fill `2 + P` real receptor fields in PIPER's term order. `grids` must already
hold `2 + P` arrays of `spec.n`; they are fully overwritten. Mutates `grids`,
allocates only a line buffer and the per-atom eigenvector column.

Keywords: `P`, `w_rep` (the shape term's `w1`), `D`, `dpad`, `truncation`,
`box`. The physics defaults are the published enzyme-inhibitor ones.

The pairwise term here is **DARS alone**. Scaling DARS by 3/4 without its partner would
be arbitrary, so it is used unscaled and this deviation is named rather than absorbed.
"""
function piper_receptor_grids!(grids::AbstractVector{<:AbstractArray{T,3}},
                               at::AtomTable{T}, par::Parametrization{T},
                               pot::Potential{T}, spec::GridSpec{T};
                               P::Integer = dars_rank(pot),
                               w_rep::Real = DEFAULT_W_REP,
                               D::Real = PIPER_D, dpad::Real = PIPER_DPAD,
                               truncation::GBTruncation = GBCapped(),
                               smooth::Union{Nothing,BoxFilter{T}} =
                                   box_filter(T, spec),
                               mask::Union{Nothing,SpatialMask{T}} = nothing) where {T}
    length(par) == length(at) ||
        throw(DimensionMismatch("parametrisation and atom table disagree in length"))
    length(grids) == 2 + P ||
        throw(DimensionMismatch("need $(2 + P) grids for P = $P, got $(length(grids))"))
    isempty(par.untyped) || throw(ArgumentError(
        "$(length(par.untyped)) atom(s) are untyped; a type index of 0 cannot " *
        "index the eigenvector matrix"))
    pot.labels == par.labels || throw(ArgumentError(
        "potential labels and type-table labels disagree; the type index would " *
        "select the wrong eigenvector"))
    1 <= P <= length(pot.lambda) ||
        throw(ArgumentError("P must be in 1:$(length(pot.lambda)), got $P"))

    mask === nothing || length(mask) == length(at) ||
        throw(DimensionMismatch("mask is for $(length(mask)) atoms, receptor " *
                                "has $(length(at))"))
    shape_receptor_grid!(grids[1], at.coords, par.radius, spec;
                         w_rep = w_rep, D = D, dpad = dpad,
                         rep_scale  = mask === nothing ? nothing : mask.rep,
                         attr_scale = mask === nothing ? nothing : mask.attr)
    gb_receptor_grid!(grids[2], at.coords, par.charge, spec;
                      D = D, truncation = truncation, smooth = smooth)
    u = Vector{T}(undef, length(at))
    for p in 1:P
        @inbounds for a in eachindex(u)
            u[a] = pot.U[par.acptype[a], p]
        end
        dars_receptor_grid!(grids[2+p], at.coords, u, spec; D = D)
    end
    return grids
end

"""
    piper_receptor_spectra(backend, at, par, pot, spec; kwargs...) -> ReceptorSpectra

Build all `2 + P` receptor fields and transform them into the conjugated,
pre-scaled spectra the hot loop consumes. Allocates `2 + P` real grids plus the
spectra; setup-time.

`P` defaults to [`dars_rank`](@ref)`(pot)` -- the smallest rank meeting the
preprint's own 10% justification -- rather than to a literal. Overriding it
below that rank emits a warning naming the measured error, once per call, which
is setup-time and therefore not a hot-loop cost.

Pass `flags = FFTW.ESTIMATE` in tests; the default `MEASURE` spends seconds
planning, which is free when a plan is reused 70,000 times and is not free in a
test suite.
"""
function piper_receptor_spectra(backend::Backend, at::AtomTable{T},
                                par::Parametrization{T}, pot::Potential{T},
                                spec::GridSpec{T};
                                P::Integer = dars_rank(pot),
                                w::PiperWeights{T} = PiperWeights{T}(
                                    T(PIPER_WEIGHTS_ENZYME_INHIBITOR.w1),
                                    T(PIPER_WEIGHTS_ENZYME_INHIBITOR.w2),
                                    T(PIPER_WEIGHTS_ENZYME_INHIBITOR.w3)),
                                D::Real = PIPER_D, dpad::Real = PIPER_DPAD,
                                truncation::GBTruncation = GBCapped(),
                                smooth::Union{Nothing,BoxFilter{T}} =
                                    box_filter(T, spec),
                                mask::Union{Nothing,SpatialMask{T}} = nothing,
                                tol::Real = 0.10,
                                flags = FFTW.MEASURE) where {T}
    _, rel = truncation_error(pot, P)
    rel > tol && @warn """
        Rank-$P truncation of the pairwise potential leaves $(round(100rel; digits=1))% \
        relative Frobenius error, above the $(round(100tol; digits=1))% the preprint \
        uses to justify truncating (PDF p.13). `dars_rank(pot; tol=$tol)` returns \
        $(dars_rank(pot; tol = tol))."""
    grids = [Array{T,3}(undef, spec.n...) for _ in 1:(2+P)]
    piper_receptor_grids!(grids, at, par, pot, spec; P = P, w_rep = w.w1,
                          D = D, dpad = dpad, truncation = truncation,
                          smooth = smooth, mask = mask)
    return spectra_from_grids(backend, grids, spec, receptor_weights(pot, w; P = P);
                              flags = flags)
end
