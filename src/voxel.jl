"""
    gridspec_for(receptor::AtomTable, ligand::AtomTable; spacing=1.2, pad=2)

Grid sizing (PDF p.13-14) driven by two parsed structures rather than by raw
coordinate vectors. Delegates to the coordinate method; see it for the rule.

Both tables must share an element type. Neither needs to be centred -- the rule
is stated in terms of extents and a centroid, both of which are translation
covariant -- but note the returned `origin` is in whatever frame the tables are
in, so mixing a centred receptor with an uncentred ligand is a bug this cannot
catch. Allocates the way the coordinate method does; setup-time.
"""
function gridspec_for(receptor::AtomTable{T}, ligand::AtomTable{T};
                      spacing::Real = 1.2, pad::Real = 2) where {T}
    return gridspec_for(receptor.coords, ligand.coords; spacing = spacing, pad = pad)
end

"""
    GridFit{T}

What a set of coordinates does to a [`GridSpec`](@ref) under trilinear
deposition. Produced by [`grid_fit`](@ref); every field is a count or a
fraction, never a verdict.

  * `natoms`          -- atoms examined
  * `n_leaking`       -- atoms whose 8-cell stencil is not wholly inside the grid
                         **and** which therefore lose weight. An atom sitting
                         exactly on the last grid plane straddles the boundary
                         but leaks nothing, and is not counted here.
  * `n_outside`       -- atoms with no stencil cell inside at all: 100% lost
  * `leak_total`      -- summed lost *fraction* over all atoms, in units of
                         atoms. `leak_total / natoms` is the mass defect the
                         deposition will show for a term with uniform weights.
  * `leak_worst`      -- largest single-atom lost fraction, in `[0, 1]`
  * `worst_atom`      -- index attaining it (`0` if nothing leaks)
  * `margin_cells`    -- smallest distance from any atom to the nearest grid
                         face, in cells. Negative means an atom is outside.
  * `margin_angstrom` -- the same in angstrom, which is the number to compare
                         against an interaction cutoff

The last one is the field to read when asking whether the *receptor field* fits,
as opposed to the receptor atoms: `shape_receptor_grid!` writes out to `D = 6` A
beyond every atom, so a `margin_angstrom` below `D` means the field itself is
clipped even though every atom landed.
"""
struct GridFit{T<:AbstractFloat}
    natoms::Int
    n_leaking::Int
    n_outside::Int
    leak_total::T
    leak_worst::T
    worst_atom::Int
    margin_cells::T
    margin_angstrom::T
end

function Base.show(io::IO, f::GridFit{T}) where {T}
    print(io, "GridFit{$T}: ", f.natoms, " atoms, ")
    if f.n_leaking == 0
        print(io, "all interior, margin ", round(f.margin_angstrom; digits = 2), " A")
    else
        print(io, f.n_leaking, " leaking (", f.n_outside, " lost entirely), ",
              "total ", round(f.leak_total; digits = 6), " atoms of weight, ",
              "worst ", round(100 * f.leak_worst; digits = 2), "% at atom ",
              f.worst_atom, ", margin ", round(f.margin_angstrom; digits = 2), " A")
    end
end

"""
    stencil_leak(spec, p) -> Tuple{T,T}

Fraction of one atom's unit weight that trilinear deposition at Cartesian point
`p` would place **outside** `spec`, and the point's distance to the nearest grid
face in cells (negative when outside).

Computed the way the leak actually arises rather than by a bounding-box test:
trilinear weights are separable, so the retained fraction is the product over
the three axes of that axis's retained weight, and the lost fraction is
`1 - product`. Exact, allocation-free, no branches on `natoms`.
"""
@inline function stencil_leak(spec::GridSpec{T}, p::SVector{3,T}) where {T}
    g = (p - spec.origin) * inv(spec.spacing)
    kept = one(T)
    margin = T(Inf)
    @inbounds for ax in 1:3
        n = spec.n[ax]
        ga = g[ax]
        i0 = floor(ga)
        f = ga - i0
        i = unsafe_trunc(Int, i0)

        lo_in = 0 <= i <= n - 1
        hi_in = 0 <= i + 1 <= n - 1
        kept *= (lo_in ? (one(T) - f) : zero(T)) + (hi_in ? f : zero(T))
        margin = min(margin, min(ga, T(n - 1) - ga))
    end
    return (one(T) - kept, margin)
end

"""
    grid_fit(spec, coords) -> GridFit
    grid_fit(spec, at::AtomTable) -> GridFit

Measure what trilinear deposition of `coords` onto `spec` would lose. Allocates
nothing; `O(natoms)`.

It answers a different question from "is the answer right" -- it answers "is the input
representable", which is the question a silent clip makes unanswerable afterwards.
"""
function grid_fit(spec::GridSpec{T}, coords::AbstractVector{SVector{3,T}}) where {T}
    n_leaking = 0
    n_outside = 0
    leak_total = zero(T)
    leak_worst = zero(T)
    worst = 0
    margin = T(Inf)
    @inbounds for a in eachindex(coords)
        leak, m = stencil_leak(spec, coords[a])
        margin = min(margin, m)
        if leak > 0
            n_leaking += 1
            leak_total += leak
            leak >= one(T) && (n_outside += 1)
            if leak > leak_worst
                leak_worst = leak
                worst = a
            end
        end
    end
    isempty(coords) && (margin = zero(T))
    return GridFit{T}(length(coords), n_leaking, n_outside, leak_total,
                      leak_worst, worst, margin, margin * spec.spacing)
end

grid_fit(spec::GridSpec{T}, at::AtomTable{T}) where {T} = grid_fit(spec, at.coords)

"""
    grid_fit_rotated(spec, coords) -> GridFit

Worst case over **every** rotation about the coordinate centroid, in `O(natoms)`
and without enumerating a rotation set.

The bound is the enclosing sphere: a rotation moves each atom onto the sphere of
its own radius about the centroid, so if the sphere of the largest such radius
clears the grid faces by one cell, no rotation can push any stencil out. This is
conservative -- a real rotation set never attains the bound simultaneously on
all six faces -- which is the right direction for a fit check.

The returned `n_leaking`/`leak_total` are therefore *worst-case* quantities: they
report `natoms` and the whole enclosing-sphere overhang rather than a per-atom
tally, because per-atom leakage is not rotation-invariant. Read `margin_cells`,
which is the quantity this method actually computes; it is `>= 1` exactly when
no rotation can lose any mass.
"""
function grid_fit_rotated(spec::GridSpec{T}, coords::AbstractVector{SVector{3,T}}) where {T}
    isempty(coords) && return GridFit{T}(0, 0, 0, zero(T), zero(T), 0, zero(T), zero(T))
    c = sum(coords) / length(coords)
    r2max = zero(T)
    worst = 1
    @inbounds for a in eachindex(coords)
        d = coords[a] - c
        r2 = d[1] * d[1] + d[2] * d[2] + d[3] * d[3]
        if r2 > r2max
            r2max = r2
            worst = a
        end
    end
    rcells = sqrt(r2max) * inv(spec.spacing)
    gc = (c - spec.origin) * inv(spec.spacing)
    margin = T(Inf)
    @inbounds for ax in 1:3
        n = spec.n[ax]
        margin = min(margin, min(gc[ax] - rcells, T(n - 1) - gc[ax] - rcells))
    end

    leaks = margin < one(T)
    return GridFit{T}(length(coords), leaks ? length(coords) : 0,
                      margin < zero(T) ? length(coords) : 0,
                      leaks ? T(length(coords)) : zero(T),
                      leaks ? one(T) : zero(T), leaks ? worst : 0,
                      margin, margin * spec.spacing)
end

grid_fit_rotated(spec::GridSpec{T}, at::AtomTable{T}) where {T} =
    grid_fit_rotated(spec, at.coords)

grid_fit_rotated(spec::GridSpec{T}, s::Structure{T}) where {T} =
    grid_fit_rotated(spec, s.coords)

"""
    mass_tolerance(T, natoms, ncells) -> T

Relative tolerance for a mass-conservation check, **derived rather than tuned**.

Three roundings enter. Per atom the eight trilinear weights sum to
`((1-fx)+fx)((1-fy)+fy)((1-fz)+fz)`, exactly 1 in the reals; in floating point
each `1-f` costs at most half an ulp, the three-fold product two more, so an
atom's deposited weight is within about `4 eps` relatively. Then the two sides
of the comparison are summed independently -- `natoms` terms on one side,
`ncells` on the other -- and Julia's pairwise summation grows error like the
logarithm of the count, not the count.

So the bound is `eps(T) * (4 + log2(natoms) + log2(ncells))`, times a factor of
8 that is a safety margin on the pairwise-summation constant and is *stated*
rather than fitted. At 128^3 in Float64 that is ~5.6e-15; in Float32, ~3.0e-6.

Compare against a defect scaled by `sum(abs, weights)`, never by
`abs(sum(weights))` -- the electrostatic term's weights are partial charges that
sum to nearly zero on a neutral protein, so a relative defect against the signed
sum is unbounded for reasons that have nothing to do with the grid.
"""
mass_tolerance(::Type{T}, natoms::Integer, ncells::Integer) where {T<:AbstractFloat} =
    T(8) * eps(T) * T(4 + log2(max(natoms, 2)) + log2(max(ncells, 2)))

"""
    mass_defect(dst, s::Structure, term) -> NamedTuple

Compare what [`voxelize!`](@ref) deposited into `dst` against what the structure
carries for that term. Returns

    (deposited, expected, scale, absdefect, reldefect, tol, ok)

with one entry per batch slice in `deposited`/`absdefect`/`reldefect`. `scale`
is `sum(abs, termw[:, term])` and `reldefect` is measured against it, for the
cancellation reason in [`mass_tolerance`](@ref). `ok` is the conjunction over
slices of `reldefect <= tol`; it is a *report*, not an assertion, so a caller
that wants to fail on it must say so.

Allocates the small per-slice vectors. Not for the hot path -- this is a
verification tool, and calling it inside the rotation loop would cost a full
pass over `dst` per rotation.
"""
function mass_defect(dst::AbstractArray{T,4}, s::Structure{T}, term::Integer) where {T}
    B = size(dst, 4)
    w = @view s.termw[:, Int(term)]
    expected = sum(w)
    scale = sum(abs, w)
    scale == 0 && (scale = one(T))
    deposited = [sum(@view dst[:, :, :, b]) for b in 1:B]
    absdefect = [abs(d - expected) for d in deposited]
    reldefect = absdefect ./ scale
    tol = mass_tolerance(T, natoms(s), prod(size(dst)[1:3]))
    return (deposited = deposited, expected = expected, scale = scale,
            absdefect = absdefect, reldefect = reldefect, tol = tol,
            ok = all(<=(tol), reldefect))
end
