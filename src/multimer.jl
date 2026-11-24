"""
    CLUSPRO_SYM_TOL

The published slab half-thickness, **2.0 A** (2017 p.9), recorded at its own
value. ClusPro's translational grid is 1.0 A, so for them this is two cells;
on JuPIPER's 1.2 A grid the same number of *cells* would be 2.4 A and the same
number of *angstroms* is what this constant is. We keep the angstrom reading --
it is what the sentence says, and it bounds the assembly's closure error rather
than the sampling -- and expose `tol` so the other reading costs one argument.
"""
const CLUSPRO_SYM_TOL = 2.0

"""
    sym_tol_floor(spacing) -> Float64

Smallest slab half-thickness at which the admitted lattice points still sample
the symmetry plane at the grid's own in-plane resolution: **`spacing / 2`**.

Derived, not chosen. Lattice points have density `1/h^3`, so a slab of full
thickness `2*tol` projects onto the plane with areal density `2*tol/h^3`. The
in-plane lattice resolution is `1/h^2`. Requiring the first to reach the second
gives `tol >= h/2` -- 0.6 A at our 1.2 A spacing.

Below this the slab is not merely tight, it is *undersampled*: the admitted
translations no longer resolve the plane as finely as the grid resolves space,
so the restricted search has a coarser translational resolution than the
unrestricted one and its search floor rises for a reason that has nothing to do
with symmetry. Above it, the tolerance trades scan work against how far from
exact closure an admitted assembly may be.

`bench/multimer.jl` sweeps `tol` against the restricted search floor and reports where
it plateaus.
"""
sym_tol_floor(spacing::Real) = Float64(spacing) / 2


"""
    symmetry_axes(T, naxes; hemisphere = false) -> Vector{SVector{3,T}}

`naxes` unit vectors from a spherical Fibonacci spiral -- uniform in `z`, hence
area-uniform on S^2 (Archimedes), golden-angle in `phi`. The same construction
`rotations_hopf` uses for its base directions, so the two sets are comparable
point for point. Allocates; setup-time.

With `hemisphere = true` the same number of points covers only `z > 0`, i.e.
twice the density on half the sphere. That is exact rather than approximate for
**C2 and only for C2**: `R(n, pi) == R(-n, pi)` as matrices (the quaternion is
`(0, n)`, and negating a quaternion is the same rotation), so antipodal axes are
literally the same rotation and searching both is duplication. For `N > 2`,
`R(n, 2pi/N)` and `R(-n, 2pi/N)` are different rotations -- they generate the
*same* ring with the subunits relabelled, but they score a different pair, so
the reduction is not free and is not taken by default.

Generated in Float64 and narrowed, for the reason in [`rotations_fibonacci`](@ref).
"""
function symmetry_axes(::Type{T}, naxes::Integer; hemisphere::Bool = false) where {T<:AbstractFloat}
    naxes > 0 || throw(ArgumentError("naxes must be positive"))
    out = Vector{SVector{3,T}}(undef, naxes)
    golden = (1 + sqrt(5.0)) / 2
    @inbounds for i in 0:(naxes - 1)

        z = hemisphere ? 1 - (2i + 1) / (2 * naxes) : 1 - (2i + 1) / naxes
        r = sqrt(max(0.0, 1 - z * z))
        phi = rem(2pi * i / golden, 2pi, RoundNearest)
        s, c = sincos(phi)
        out[i + 1] = SVector{3,T}(T(r * c), T(r * s), T(z))
    end
    return out
end

symmetry_axes(naxes::Integer; kw...) = symmetry_axes(Float64, naxes; kw...)

"""
    hopf_axes(rs::RotationSet) -> Vector{SVector{3,T}}

The `ndir` base directions of a Hopf set, read off the set itself as
`R * xhat` for one member of each fibre. Errors on an unstructured set rather
than inventing a grouping, exactly as [`fiber`](@ref) does.

This is what "restrict the Hopf rotation grid to its S^2 base" means literally: the
fibre invariant *is* the base direction, so the axis set and the docking rotation set
are two views of one object. A set built this way inherits the balanced split's property
that the S^2 spacing equals the fibre spacing, so the axis resolution matches the
resolution the full set achieves on SO(3).
"""
function hopf_axes(rs::RotationSet{T}) where {T}
    rs.nspin > 0 || throw(ArgumentError("$(rs.kind) set has no fibre structure; use rotations_hopf"))
    xhat = SVector{3,T}(one(T), zero(T), zero(T))
    return [rs.rots[first(fiber(rs, i))] * xhat for i in 1:rs.ndir]
end


"""
    _half_turn(T, order) -> (sin, cos) of `pi / order`

The quaternion half-angle for a `2*pi/order` turn, written down for the
crystallographic orders instead of computed from a rounded `pi`.

This is not tidiness. `sincos(pi/2)` returns `cos = 6.12e-17`, not zero, and that
residue is enough to break the one exact statement C2 symmetry has: `R(n, pi)` and
`R(-n, pi)` are the same rotation, and with a nonzero `w` the two matrices differ in the
last bit -- so a hemisphere axis set stops being an exact halving and becomes an
approximate one, which is a much weaker thing to claim. With `w == 0` exactly, every
entry of the matrix is a product of two vector components and is therefore invariant
under `n -> -n` bit for bit.
"""
@inline function _half_turn(::Type{T}, order::Integer) where {T<:AbstractFloat}
    order == 2 && return (one(T), zero(T))
    order == 3 && return (T(sqrt(3.0) / 2), T(0.5))
    order == 4 && return (T(sqrt(2.0) / 2), T(sqrt(2.0) / 2))
    order == 6 && return (T(0.5), T(sqrt(3.0) / 2))
    return sincos(T(pi / order))
end

"""
    CyclicSymmetry{T}

A Cn symmetry-restricted search space: one rotation per axis, fixed at
`2*pi/order`, plus the slab half-thickness the perpendicular-plane constraint is
relaxed by.

`rots` is what [`dock_symmetric!`](@ref) and [`pose_transform`](@ref) consume --
it is an ordinary `Vector{SMatrix{3,3,T,9}}`, so every existing driver, floor
and pose routine takes it unchanged. Nothing here is a new rotation *sampler*;
it is a projection of one.

Fields: `order` (N), `axes`, `rots`, `quats`, `tol` (angstrom).
"""
struct CyclicSymmetry{T<:AbstractFloat}
    order::Int
    axes::Vector{SVector{3,T}}
    rots::Vector{SMatrix{3,3,T,9}}
    quats::Vector{SVector{4,T}}
    tol::T
end

Base.length(s::CyclicSymmetry) = length(s.rots)
Base.getindex(s::CyclicSymmetry, i::Integer) = s.rots[i]
Base.eltype(::CyclicSymmetry{T}) where {T} = T

function Base.show(io::IO, s::CyclicSymmetry{T}) where {T}
    print(io, "CyclicSymmetry{", T, "}(C", s.order, ", ", length(s.axes),
          " axes, ", round(Float64(s.tol); digits = 2), " A plane tolerance)")
end

"""
    cyclic_symmetry(order, axes; tol = CLUSPRO_SYM_TOL) -> CyclicSymmetry

Build the operation set from an explicit axis list. Each axis `n` contributes
the single rotation `R(n, 2*pi/order)`; the fibre dimension of a general
rotation set is not sampled at all, because for a Cn assembly the rotation
*angle* is fixed by `N` and only its axis is free.

`order` must be at least 2. `order = 1` is the unconstrained problem wearing a
constraint's name -- `R` would be the identity, every `t` would satisfy
`n . t = 0` vacuously for no `n`, and the caller almost certainly meant
[`dock!`](@ref); it is refused rather than silently degenerating.
"""
function cyclic_symmetry(order::Integer, axes::AbstractVector{SVector{3,T}};
                         tol::Real = CLUSPRO_SYM_TOL) where {T<:AbstractFloat}
    order >= 2 || throw(ArgumentError(
        "cyclic order must be >= 2, got $order -- C1 is unconstrained docking, " *
        "call dock! instead"))
    isempty(axes) && throw(ArgumentError("need at least one symmetry axis"))
    tol > 0 || throw(ArgumentError("plane tolerance must be positive, got $tol"))
    s, c = _half_turn(T, order)
    quats = Vector{SVector{4,T}}(undef, length(axes))
    @inbounds for i in eachindex(axes)
        n = axes[i]
        nn = norm(n)
        nn > 0 || throw(ArgumentError("axis $i is the zero vector"))
        u = n / nn
        quats[i] = SVector{4,T}(c, s * u[1], s * u[2], s * u[3])
    end
    unit = [a / norm(a) for a in axes]
    return CyclicSymmetry{T}(Int(order), unit, map(rotmatrix, quats), quats, T(tol))
end

"""
    cyclic_symmetry(T, order, naxes; tol, hemisphere = (order == 2)) -> CyclicSymmetry

Build from a point count, using [`symmetry_axes`](@ref). The `hemisphere`
default is a theorem rather than a preference: for C2 the antipodal axis gives
the *same matrix*, so half the sphere at twice the density is an exact 2x with
no loss, and it is asserted in `test_multimer.jl` rather than argued here.
"""
function cyclic_symmetry(::Type{T}, order::Integer, naxes::Integer;
                         tol::Real = CLUSPRO_SYM_TOL,
                         hemisphere::Bool = (order == 2)) where {T<:AbstractFloat}
    return cyclic_symmetry(order, symmetry_axes(T, naxes; hemisphere = hemisphere); tol = tol)
end

"""
    cyclic_symmetry(order, rs::RotationSet; tol = CLUSPRO_SYM_TOL) -> CyclicSymmetry

Restrict an existing Hopf rotation set to its S^2 base. The resulting search
visits `rs.ndir` rotations where the full set visits `rs.ndir * rs.nspin`, and
the ratio -- exactly `rs.nspin` -- is the transform-work reduction, because one
FFT schedule is issued per rotation.

Note this route does **not** take the C2 hemisphere halving: a Fibonacci base is
not antipodally paired, so both hemispheres are sampled and a C2 search over
them covers RP^2 at twice the necessary density. That is a resolution gain, not
an error, and it is the price of reusing the set the unconstrained arm runs on
-- which is what makes the A/B a comparison of one set against its own base.
"""
cyclic_symmetry(order::Integer, rs::RotationSet{T}; tol::Real = CLUSPRO_SYM_TOL) where {T} =
    cyclic_symmetry(order, hopf_axes(rs); tol = tol)

"""
    axis_angle(R) -> (axis, angle)

Unit rotation axis and angle in radians (`angle` in `[0, pi]`), via the
quaternion rather than via the antisymmetric part of `R`.

The route matters. The textbook `axis = [R32-R23, R13-R31, R21-R12] / (2 sin t)`
divides by `sin t`, which is exactly what vanishes at the angle a C2 search
cares most about: the A-B operation of a real dimer measures 179.46 degrees on
4XAU, where `sin t = 0.0095` and the axis loses two digits. Shepperd's
quaternion extraction picks the branch with the largest denominator and is
stable everywhere, including at `pi`.

For the identity the axis is undefined; `(0,0,1)` is returned with angle 0 so a
caller comparing angles is not forced to branch. Allocation-free.
"""
function axis_angle(R::SMatrix{3,3,T}) where {T}
    t = R[1,1] + R[2,2] + R[3,3]
    local w::T, x::T, y::T, z::T
    if t > 0
        s = sqrt(t + one(T)) * 2
        w = s / 4; x = (R[3,2] - R[2,3]) / s; y = (R[1,3] - R[3,1]) / s; z = (R[2,1] - R[1,2]) / s
    elseif R[1,1] > R[2,2] && R[1,1] > R[3,3]
        s = sqrt(one(T) + R[1,1] - R[2,2] - R[3,3]) * 2
        w = (R[3,2] - R[2,3]) / s; x = s / 4; y = (R[1,2] + R[2,1]) / s; z = (R[1,3] + R[3,1]) / s
    elseif R[2,2] > R[3,3]
        s = sqrt(one(T) + R[2,2] - R[1,1] - R[3,3]) * 2
        w = (R[1,3] - R[3,1]) / s; x = (R[1,2] + R[2,1]) / s; y = s / 4; z = (R[2,3] + R[3,2]) / s
    else
        s = sqrt(one(T) + R[3,3] - R[1,1] - R[2,2]) * 2
        w = (R[2,1] - R[1,2]) / s; x = (R[1,3] + R[3,1]) / s; y = (R[2,3] + R[3,2]) / s; z = s / 4
    end
    v = SVector{3,T}(x, y, z)
    nv = norm(v)
    if nv <= eps(T)
        return (SVector{3,T}(zero(T), zero(T), one(T)), zero(T))
    end

    ang = 2 * atan(nv, abs(w))
    ax = w < 0 ? -v / nv : v / nv
    return (ax, T(ang))
end


"""
    SymmetryFit{T}

The rigid operation carrying one subunit onto another, read as a symmetry
element:

  * `order` -- `round(2*pi / angle)`, the Cn it is closest to
  * `axis`, `angle` (degrees)
  * `rise`  -- `axis . t`, the **screw** component, in angstrom
  * `defect` -- `|angle - 360/order|` in degrees
  * `rmsd`  -- residual of the superposition, i.e. how alike the subunits are

`rise` is the number that decides whether a Cn *docking* search can reproduce
the assembly at all: a closed ring has `rise == 0` by the derivation at the head
of this file, and a helical arrangement does not, however good the scoring
function is. It is invariant under a change of origin, so it is a property of
the assembly rather than of the frame -- unlike the raw closure residual
`|(sum_k R^k) t|`, which scales with the distance from the origin to the axis
and is therefore worthless as a diagnostic.

!!! warning "For a 2-fold, read `|rise|`: its sign carries no information"
    `R(n, pi) == R(-n, pi)`, so a 2-fold's axis has no direction and the sign of
    `axis . t` is whichever branch the quaternion extraction landed on. At
    exactly `pi` the scalar part is at round-off, so re-fitting the *same*
    assembly from coordinates shifted by a constant can return `-rise` -- the
    magnitude is stable, the sign is not. That is real geometry, not an
    instability to be papered over: the two ends of a 2-fold axis are the same
    axis. For `order >= 3` the angle is below `pi`, the right-hand rule fixes
    the direction, and the sign is meaningful.
"""
struct SymmetryFit{T<:AbstractFloat}
    order::Int
    axis::SVector{3,T}
    angle::T
    rise::T
    defect::T
    rmsd::T
end

function Base.show(io::IO, f::SymmetryFit{T}) where {T}
    print(io, "SymmetryFit{", T, "}: C", f.order, " (",
          round(Float64(f.angle); digits = 3), " deg, defect ",
          round(Float64(f.defect); digits = 3), " deg), rise ",
          round(Float64(f.rise); digits = 4), " A, subunit RMSD ",
          round(Float64(f.rmsd); digits = 3), " A")
end

"""
    symmetry_fit(P, Q) -> SymmetryFit

Superimpose subunit coordinates `P` onto `Q` ([`kabsch`](@ref)) and report the
result as a symmetry element. `P` and `Q` must be the *same atoms* of the two
subunits, in the same order -- for two chains of one deposition that means
matching on residue number first, which is a caller's job because a structure
file offers no guarantee that two chains model the same residues.

Allocates as `kabsch` does; analysis-time. This is the check to run **before**
docking a putative multimer: it costs one superposition and it says whether the
target is a closed ring, and of what order.
"""
function symmetry_fit(P::AbstractVector{SVector{3,T}},
                      Q::AbstractVector{SVector{3,T}}) where {T}
    R, t = kabsch(P, Q)
    ax, ang = axis_angle(R)
    s = zero(T)
    @inbounds for k in eachindex(P)
        d = R * P[k] + t - Q[k]
        s += dot(d, d)
    end
    deg = T(rad2deg(ang))

    ord = ang <= eps(T) ? 0 : max(1, round(Int, 2pi / ang))
    defect = ord == 0 ? deg : abs(deg - T(360) / T(ord))
    return SymmetryFit{T}(ord, ax, deg, dot(ax, t), defect,
                          sqrt(s / T(length(P))))
end


"""
    plane_cells!(out, axis, spec, tol) -> out

Linear cell indices of the translations within `tol` of the plane through the
origin perpendicular to `axis`. Empties and refills `out`; returns it.

The pose convention is [`pose_transform`](@ref)'s: cell offset `m` (minimal image, so
signed) stands for the translation `t = -spacing * m`, hence the test is `spacing * |m.
axis| <= tol` -- the sign of `t` drops out under the absolute value, which is the one
place in this project where it safely can.

**Not sorted, deliberately.** The set is `O(tol/spacing)` slabs of the whole grid --
tens of thousands of cells at production size -- and sorting it is `O(k log k)` against
a scan that is `O(k)`, so the sort would cost more than the work it orders. Determinism
instead comes from the reduction's tie-break: [`dock_symmetric!`](@ref) compares
`(value, index)` lexicographically, which is order-independent and therefore reproduces
the serial `scan_best!` answer whatever order the cells arrive in.

Allocation-free after the first call on a given buffer (it grows and is reused).
"""
function plane_cells!(out::Vector{Int32}, axis::SVector{3,T}, spec::GridSpec{T},
                      tol::Real) where {T}
    empty!(out)
    nx, ny, nz = spec.n
    h = spec.spacing
    n = axis / norm(axis)
    d = T(tol) / h
    lo = (_mi_range(nx), _mi_range(ny), _mi_range(nz))

    k = argmax(SVector{3,T}(abs(n[1]), abs(n[2]), abs(n[3])))
    i, j = k == 1 ? (2, 3) : k == 2 ? (1, 3) : (1, 2)
    nk = n[k]
    dims = (nx, ny, nz)
    strides = (1, nx, nx * ny)
    @inbounds for mj in lo[j][1]:lo[j][2], mi in lo[i][1]:lo[i][2]

        base = n[i] * mi + n[j] * mj
        a = (-d - base) / nk
        b = ( d - base) / nk
        a, b = minmax(a, b)
        m0 = max(lo[k][1], ceil(Int, a))
        m1 = min(lo[k][2], floor(Int, b))
        m0 > m1 && continue
        cij = 1 + strides[i] * mod(mi, dims[i]) + strides[j] * mod(mj, dims[j])
        sk = strides[k]
        for mk in m0:m1
            push!(out, Int32(cij + sk * mod(mk, dims[k])))
        end
    end
    return out
end

"""
    plane_cells(axis, spec, tol) -> Vector{Int32}

Non-mutating [`plane_cells!`](@ref). # convenience: allocates
"""
plane_cells(axis::SVector{3,T}, spec::GridSpec{T}, tol::Real) where {T} =
    plane_cells!(Int32[], axis, spec, tol)


"""
    symmetric_search_floor(sym, spec, y, z; mask = nothing) -> SearchFloor

The best RMSD the **restricted** space contains: the minimum over
(symmetry axes) x (lattice cells admitted by that axis's plane slab) of the
RMSD between `z` and `R*y + t`. No energy is consulted.

This is [`search_floor`](@ref)'s job for the constrained space, and it cannot be
delegated to it: `search_floor` rounds the optimal translation to the lattice
per axis, which is exact when every lattice point is admissible and simply wrong
when they are not -- the rounded cell usually leaves the slab. Here the slab is
enumerated, so the floor describes the space `dock_symmetric!` actually searches
and a docking result read against it is read against its own search space.

`O(naxes * (plane cells + natoms))`. Allocates the cell buffer once.
"""
function symmetric_search_floor(sym::CyclicSymmetry{T}, spec::GridSpec{T},
                                y::AbstractVector{SVector{3,T}},
                                z::AbstractVector{SVector{3,T}};
                                mask::Union{Nothing,AbstractVector{Bool}} = nothing) where {T}
    length(y) == length(z) || throw(DimensionMismatch("y and z differ in length"))
    h = spec.spacing
    nx, ny, nz = spec.n
    ybar = zero(SVector{3,T}); zbar = zero(SVector{3,T}); M = 0
    @inbounds for k in eachindex(y)
        (mask === nothing || mask[k]) || continue
        ybar += y[k]; zbar += z[k]; M += 1
    end
    M == 0 && throw(ArgumentError("mask selects no atoms"))
    ybar /= T(M); zbar /= T(M)

    cells = Int32[]
    best = T(Inf); bestrot = 0; bestcells = (0, 0, 0)
    invM = one(T) / T(M)
    for r in eachindex(sym.rots)
        R = sym.rots[r]
        plane_cells!(cells, sym.axes[r], spec, sym.tol)

        tstar = zbar - R * ybar
        bd = T(Inf); bc = (0, 0, 0)
        @inbounds for lin in cells
            cx, cy, cz = unravel(spec, Int(lin))
            mx = minimal_image(cx, nx); my = minimal_image(cy, ny); mz = minimal_image(cz, nz)
            dx = -h * T(mx) - tstar[1]
            dy = -h * T(my) - tstar[2]
            dz = -h * T(mz) - tstar[3]
            v = dx * dx + dy * dy + dz * dz
            if v < bd
                bd = v; bc = (mx, my, mz)
            end
        end
        t = -h * SVector{3,T}(T(bc[1]), T(bc[2]), T(bc[3]))
        s = zero(T)
        @inbounds for k in eachindex(y)
            (mask === nothing || mask[k]) || continue
            d = R * y[k] + t - z[k]
            s += dot(d, d)
        end
        v = sqrt(s * invM)
        if v < best
            best = v; bestrot = r; bestcells = bc
        end
    end
    ang = T(rad2deg(_rot_angle(sym.rots[bestrot])))
    return SearchFloor{T}(best, bestrot, bestcells, ang)
end


"""
    SymmetryReport

What a symmetric run searched, with the two reductions kept apart because only
one of them removes transform work.

`nrot_full` is the size of the unconstrained rotation set the axes were derived
from (or `naxes` if they were not derived from one), so `rotation_reduction`
is a like-for-like ratio rather than a comparison against a number nobody would
have run.
"""
struct SymmetryReport
    order::Int
    naxes::Int
    nrot_full::Int
    ncells::Int
    plane_total::Int
    tol::Float64
    t_cells::Float64
    t_dock::Float64
end

"""
    rotation_reduction(report) -> Float64

`nrot_full / naxes`: the factor by which the symmetry constraint cut the number
of FFT schedules issued. **This is the whole of the speedup**, because one
schedule is issued per rotation.
"""
rotation_reduction(r::SymmetryReport) = r.naxes == 0 ? 0.0 : r.nrot_full / r.naxes

"""
    translation_reduction(report) -> Float64

`ncells / (mean admitted cells)`: the factor by which the plane constraint cut the
*scan*. It removes **no** transform work -- one transform already produces every
translation -- so this number must never be multiplied by [`rotation_reduction`](@ref)
to form a speedup.
"""
translation_reduction(r::SymmetryReport) =
    r.plane_total == 0 ? 0.0 : r.ncells * r.naxes / r.plane_total

function Base.show(io::IO, r::SymmetryReport)
    print(io, "SymmetryReport(C", r.order, ", ", r.naxes, " axes vs ", r.nrot_full,
          " rotations = ", round(rotation_reduction(r); digits = 1),
          "x transforms; ", round(translation_reduction(r); digits = 1),
          "x fewer translations scanned, worth no transform work)")
end

"""
    dock_symmetric!(poses, lig, sym, rs, pool; nrot_full = 0) -> (poses, report)

Cn-symmetric docking: one rotation per symmetry axis, scored only at the
translations its plane slab admits.

`poses` is one slot per axis. The pose written for axis `i` carries `rot = i`,
which indexes `sym.rots` -- so [`pose_transform`](@ref)`(pose, sym.rots, spec)`
is the placement, exactly as for an unconstrained run.

`nrot_full` is recorded in the report for the reduction ratio and has no effect
on the computation; pass the size of the unconstrained set the axes came from.

The reduction over the admitted cells uses a **lexicographic `(value, index)`**
tie-break, so the answer does not depend on the order [`plane_cells!`](@ref)
emits and is bit-identical to the minimum `scan_best!` would return over the
same cells. That is asserted in `test_multimer.jl` on landscapes built to make
the tie-break decide.
"""
function dock_symmetric!(poses::AbstractVector{Pose{T}}, lig::Structure{T},
                         sym::CyclicSymmetry{T}, rs::ReceptorSpectra{T},
                         pool::AbstractVector{<:Workspace{T}};
                         nrot_full::Integer = 0) where {T}
    naxes = length(sym)
    length(poses) == naxes || throw(DimensionMismatch(
        "one pose slot per symmetry axis ($naxes), got $(length(poses))"))
    isempty(pool) && throw(ArgumentError("pool must not be empty"))
    all(w -> w.batch == 1, pool) ||
        throw(ArgumentError("dock_symmetric! requires batch == 1 workspaces"))

    spec = first(pool).spec
    ncells = prod(spec.n)
    nchunks = length(pool)
    chunks = collect(Iterators.partition(1:naxes, cld(naxes, nchunks)))
    @assert length(chunks) <= nchunks "chunking invariant violated: would share workspaces"

    ptot  = zeros(Int, length(chunks))
    tcell = zeros(Float64, length(chunks))
    tdock = zeros(Float64, length(chunks))

    @sync for (ic, chunk) in enumerate(chunks)
        ws = pool[ic]
        buf = Int32[]
        Threads.@spawn begin
            for i in chunk
                t0 = time_ns()
                cells = plane_cells!(buf, sym.axes[i], spec, sym.tol)
                tcell[ic] += (time_ns() - t0) * 1e-9

                @assert !isempty(cells) "empty plane slab -- tol must be positive"
                ptot[ic] += length(cells)
                t1 = time_ns()
                correlate!(ws, lig, view(sym.rots, i:i), rs)
                sc = ws.score
                best = typemax(T); bi = Int(cells[1])
                @inbounds for c in cells
                    v = sc[c]

                    if v < best || (v == best && Int(c) < bi)
                        best = v; bi = Int(c)
                    end
                end
                tdock[ic] += (time_ns() - t1) * 1e-9
                x, y, z = unravel(spec, bi)
                poses[i] = Pose{T}(best, Int32(i), Int16(x), Int16(y), Int16(z))
            end
        end
    end

    return (poses, SymmetryReport(sym.order, naxes,
                                  nrot_full > 0 ? Int(nrot_full) : naxes,
                                  ncells, sum(ptot), Float64(sym.tol),
                                  sum(tcell), sum(tdock)))
end
