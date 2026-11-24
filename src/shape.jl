"""
    VDW_RADII

Van der Waals radii in angstrom, by element symbol (uppercase). Bondi's values,
which are the set almost universally used for this purpose.

The model uses uniform radii by element. The reference adjusts radii by
surface exposure to account for conformational differences (preprint p.10).
"""
const VDW_RADII = Dict{String,Float64}(
    "H" => 1.20, "C" => 1.70, "N" => 1.55, "O" => 1.52,
    "S" => 1.80, "P" => 1.80, "F" => 1.47, "CL" => 1.75,
    "BR" => 1.85, "I" => 1.98, "SE" => 1.90,
    "FE" => 2.00, "ZN" => 1.39, "MG" => 1.73, "CA" => 2.31, "NA" => 2.27,
)

"""Radius for an element symbol, defaulting to carbon for anything unlisted."""
vdw_radius(::Type{T}, element::AbstractString) where {T} =
    T(get(VDW_RADII, uppercase(strip(element)), 1.70))

"""
    DEFAULT_W_REP

The paper's `w1`, the weight of the repulsive half of `E_shape`.

The reference fitted a value of 2.1 for enzyme/inhibitor complexes and 2.0 for
antibody/antigen complexes by logistic regression (preprint p.15).
See [`PIPER_WEIGHTS_ENZYME_INHIBITOR`](@ref) for the enzyme/inhibitor weights.
"""
const DEFAULT_W_REP = 2.1

"""
    shape_receptor_grid!(R, coords, radii, spec; w_rep=DEFAULT_W_REP, D=6, dpad=2) -> R

Fill `R` with PIPER's receptor shape field `-c + w_rep * r` (PDF p.10).

Mutates and fully overwrites `R`, which must be `(nx, ny, nz)`. Allocates
nothing beyond what `R` already is. Setup-time: `O(natoms * (2D/h)^3)`, about a
million operations for a 2000-atom receptor at 1.2 A spacing -- irrelevant next
to 70,000 rotations of FFT work.

`w_rep` is the paper's `w1`, defaulting to the published enzyme-inhibitor value
(see [`DEFAULT_W_REP`](@ref)). It stays an argument rather than a constant
because the paper fits a *different* value for antibody-antigen, and because
p.9 makes clear it is a tolerance knob: "selected to avoid substantial steric
clashes, but to allow for some atomic overlaps that occur due the differences
between bound and unbound structures".

`nothing` means "all ones" and is the only path with no per-atom multiply. They are
plain vectors rather than a [`SpatialMask`](@ref) so that this file keeps no dependency
on the ingestion layer; build them with [`spatial_mask`](@ref).

The published `w1` was fitted with surface- and core-specific radii. Its
calibration therefore need not transfer to shells built with uniform radii.
"""
function shape_receptor_grid!(R::AbstractArray{T,3},
                              coords::AbstractVector{SVector{3,T}},
                              radii::AbstractVector{T},
                              spec::GridSpec{T};
                              w_rep::Real = DEFAULT_W_REP, D::Real = 6,
                              dpad::Real = 2,
                              rep_scale::Union{Nothing,AbstractVector{T}} = nothing,
                              attr_scale::Union{Nothing,AbstractVector{T}} = nothing) where {T}
    length(coords) == length(radii) ||
        throw(DimensionMismatch("one radius per atom required"))
    rep_scale === nothing || length(rep_scale) == length(coords) ||
        throw(DimensionMismatch("one repulsion multiplier per atom required"))
    attr_scale === nothing || length(attr_scale) == length(coords) ||
        throw(DimensionMismatch("one attraction multiplier per atom required"))
    nx, ny, nz = spec.n
    size(R) == (nx, ny, nz) || throw(DimensionMismatch("R must be $(spec.n)"))
    fill!(R, zero(T))
    h = spec.spacing
    inv_h = inv(h)
    Dc = T(D) * inv_h
    D2 = T(D)^2
    wr = T(w_rep)
    origin = spec.origin
    @inbounds for a in eachindex(coords)
        g = (coords[a] - origin) * inv_h
        d_a = radii[a] + T(dpad)
        d2 = d_a * d_a

        wa = rep_scale  === nothing ? wr     : wr * rep_scale[a]
        ca = attr_scale === nothing ? one(T) : attr_scale[a]

        lo1 = max(1, ceil(Int, g[1] - Dc) + 1); hi1 = min(nx, floor(Int, g[1] + Dc) + 1)
        lo2 = max(1, ceil(Int, g[2] - Dc) + 1); hi2 = min(ny, floor(Int, g[2] + Dc) + 1)
        lo3 = max(1, ceil(Int, g[3] - Dc) + 1); hi3 = min(nz, floor(Int, g[3] + Dc) + 1)
        for n in lo3:hi3
            dz = ((n - 1) - g[3]) * h
            dz2 = dz * dz
            for m in lo2:hi2
                dy = ((m - 1) - g[2]) * h
                dyz2 = dz2 + dy * dy
                dyz2 > D2 && continue
                for l in lo1:hi1
                    dx = ((l - 1) - g[1]) * h
                    r2 = dyz2 + dx * dx
                    r2 > D2 && continue

                    R[l, m, n] += r2 < d2 ? wa : -ca
                end
            end
        end
    end
    return R
end

"""
    occupancy_grid!(L, coords, radii, spec) -> L

Fill `L` with PIPER's hard ligand indicator: 1 where a grid point lies inside a
ligand atom's van der Waals sphere, 0 elsewhere (PDF p.10). Mutates `L`.

This is the reference definition. The production hot loop uses trilinear
deposition instead (see [`Trilinear`](@ref)) because a hard indicator has zero
gradient with respect to atom coordinates almost everywhere, which would end
the differentiable-docking direction before it starts. Keeping the exact form
here means the deviation can be *measured* rather than argued about --
`test_shape.jl` does exactly that.

Not for the hot path: it is `O(natoms * (2 r_vdw / h)^3)` per rotation with no
reuse.
"""
function occupancy_grid!(L::AbstractArray{T,3},
                         coords::AbstractVector{SVector{3,T}},
                         radii::AbstractVector{T},
                         spec::GridSpec{T}) where {T}
    nx, ny, nz = spec.n
    size(L) == (nx, ny, nz) || throw(DimensionMismatch("L must be $(spec.n)"))
    fill!(L, zero(T))
    h = spec.spacing
    inv_h = inv(h)
    origin = spec.origin
    @inbounds for a in eachindex(coords)
        g = (coords[a] - origin) * inv_h
        ra = radii[a]
        rc = ra * inv_h
        r2max = ra * ra
        lo1 = max(1, ceil(Int, g[1] - rc) + 1); hi1 = min(nx, floor(Int, g[1] + rc) + 1)
        lo2 = max(1, ceil(Int, g[2] - rc) + 1); hi2 = min(ny, floor(Int, g[2] + rc) + 1)
        lo3 = max(1, ceil(Int, g[3] - rc) + 1); hi3 = min(nz, floor(Int, g[3] + rc) + 1)
        for n in lo3:hi3
            dz = ((n - 1) - g[3]) * h
            for m in lo2:hi2
                dy = ((m - 1) - g[2]) * h
                for l in lo1:hi1
                    dx = ((l - 1) - g[1]) * h
                    if dx * dx + dy * dy + dz * dz <= r2max
                        L[l, m, n] = one(T)
                    end
                end
            end
        end
    end
    return L
end

"""
    spectra_from_grids(backend, grids, spec, weights; flags) -> ReceptorSpectra

Turn explicitly-built real receptor grids into the conjugated, pre-scaled
spectra the hot loop consumes: `conj(rfft(G_p)) * weights[p] / (nx*ny*nz)`.

This is the seam that lets real physics enter the pipeline. `receptor_spectra`
builds its grids by trilinear deposition, which is a placeholder; this takes
whatever field you computed -- the shape field above, a Generalized Born
electrostatic field, anything -- and does only the transform and the scaling.

Allocates: `P` spectra plus a plan. Setup-time.
"""
function spectra_from_grids(backend::Backend, grids::AbstractVector{<:AbstractArray{T,3}},
                            spec::GridSpec{T}, weights::AbstractVector{T};
                            flags = FFTW.MEASURE) where {T}
    P = length(grids)
    length(weights) == P || throw(DimensionMismatch("one weight per grid required"))
    nx, ny, nz = spec.n
    nxr = speclen(spec)
    scale = one(T) / T(nx * ny * nz)

    buf = allocate(backend, T, nx, ny, nz, 1)
    plan = FFTW_plan_rfft(buf, flags)
    tmp = allocate(backend, Complex{T}, nxr, ny, nz, 1)
    terms = [allocate(backend, Complex{T}, nxr, ny, nz) for _ in 1:P]
    for p in 1:P
        size(grids[p]) == (nx, ny, nz) ||
            throw(DimensionMismatch("grid $p is $(size(grids[p])), expected $(spec.n)"))

        copyto!(reshape(buf, nx, ny, nz), grids[p])
        mul!(tmp, plan, buf)
        terms[p] .= conj.(reshape(tmp, nxr, ny, nz)) .* (weights[p] * scale)
    end
    return ReceptorSpectra{T,eltype(terms)}(terms, spec)
end

"""
    shape_spectra(backend, coords, radii, spec; w_rep, D, dpad, flags) -> ReceptorSpectra

One-term receptor spectra for shape-only ("lock and key") docking. Convenience
wrapper over [`shape_receptor_grid!`](@ref) and [`spectra_from_grids`](@ref).

The resulting `ReceptorSpectra` has `nterms == 1`, so `correlate!` runs **two**
transforms per rotation instead of `P + 1`. Allocates; setup-time.
"""
function shape_spectra(backend::Backend, coords::AbstractVector{SVector{3,T}},
                       radii::AbstractVector{T}, spec::GridSpec{T};
                       w_rep::Real = DEFAULT_W_REP, D::Real = 6, dpad::Real = 2,
                       flags = FFTW.MEASURE) where {T}
    R = Array{T,3}(undef, spec.n...)
    shape_receptor_grid!(R, coords, radii, spec; w_rep = w_rep, D = D, dpad = dpad)
    return spectra_from_grids(backend, [R], spec, ones(T, 1); flags = flags)
end

"""
    shape_structure(coords) -> Structure

A `Structure` carrying exactly one correlation term with unit weight per atom:
the ligand side of the shape term. Allocates.
"""
shape_structure(coords::AbstractVector{SVector{3,T}}) where {T} =
    Structure(coords, ones(T, length(coords), 1))
