"""
    minimal_image(t, n) -> Int

Signed cell offset equivalent to `t` on a periodic axis of `n` cells, chosen in
`(-n/2, n/2]`. Allocation-free.

A translation index from [`unravel`](@ref) is in `0:n-1`; under the FFT's
periodic boundary `t` and `t - n` name the same placement, and the minimal
image is the one that keeps the ligand next to the receptor rather than a box
away. See the header of this file for why the choice is forced.
"""
@inline function minimal_image(t::Integer, n::Integer)
    tt = mod(Int(t), Int(n))
    return tt > Int(n) >> 1 ? tt - Int(n) : tt
end

"""
    pose_transform(pose, rots, spec) -> (R, t)

The rigid placement a [`Pose`](@ref) stands for: a ligand atom whose **centred**
coordinate is `y` sits at `R * y + t` in the receptor's frame -- the frame the
receptor's own `AtomTable` is in, which is what `spec.origin` was built from.

`rots` may be a `Vector{<:SMatrix{3,3,T}}` or a [`RotationSet`](@ref).
Allocation-free; returns stack values.

The translation is `t = -spacing * minimal_image(cells)`. The minus sign is the
correlation convention (file header); it is not a free choice and it is pinned
by test rather than by argument.
"""
function pose_transform(pose::Pose{T}, rots::AbstractVector{<:SMatrix{3,3,T}},
                        spec::GridSpec{T}) where {T}
    R = rots[Int(pose.rot)]
    nx, ny, nz = spec.n
    t = -spec.spacing * SVector{3,T}(T(minimal_image(pose.tx, nx)),
                                     T(minimal_image(pose.ty, ny)),
                                     T(minimal_image(pose.tz, nz)))
    return (R, t)
end

pose_transform(pose::Pose{T}, rs::RotationSet{T}, spec::GridSpec{T}) where {T} =
    pose_transform(pose, rs.rots, spec)

"""
    pose_transforms(poses, rots, spec) -> (Rs, ts)

[`pose_transform`](@ref) over a vector of poses, in the layout
[`pairwise_rmsd!`](@ref) consumes. Allocates two vectors; analysis-time.
"""
function pose_transforms(poses::AbstractVector{Pose{T}}, rots, spec::GridSpec{T}) where {T}
    Rs = Vector{SMatrix{3,3,T,9}}(undef, length(poses))
    ts = Vector{SVector{3,T}}(undef, length(poses))
    for (i, p) in enumerate(poses)
        Rs[i], ts[i] = pose_transform(p, rots, spec)
    end
    return (Rs, ts)
end

"""
    pose_coords!(out, y, R, t) -> out

Place centred ligand coordinates `y` under the rigid transform `(R, t)`.
Mutates `out`; allocates nothing.
"""
function pose_coords!(out::AbstractVector{SVector{3,T}},
                      y::AbstractVector{SVector{3,T}},
                      R::SMatrix{3,3,T}, t::SVector{3,T}) where {T}
    length(out) == length(y) || throw(DimensionMismatch("out and y differ in length"))
    @inbounds for k in eachindex(y)
        out[k] = R * y[k] + t
    end
    return out
end

"""
    pose_coords(y, R, t) -> Vector{SVector{3,T}}

Non-mutating [`pose_coords!`](@ref). # convenience: allocates
"""
pose_coords(y::AbstractVector{SVector{3,T}}, R::SMatrix{3,3,T}, t::SVector{3,T}) where {T} =
    pose_coords!(similar(y), y, R, t)

"""
    native_ligand_coords(rec, lig) -> Vector{SVector{3,T}}

The ligand's **native** coordinates expressed in the receptor's current frame,
which is the frame [`pose_coords`](@ref) returns poses in. Allocates one vector.

`AtomTable.centroid` is what makes this possible: `center!` records the shift it
removed, so `coords + centroid` reconstructs the deposited frame whatever has
been done since, and subtracting the receptor's centroid moves into whatever
frame the receptor is in *now*. Both tables must have been read from the same
complex; nothing here can check that, and there is no meaningful answer if they
were not.
"""
function native_ligand_coords(rec::AtomTable{T}, lig::AtomTable{T}) where {T}
    shift = lig.centroid - rec.centroid
    return [c + shift for c in lig.coords]
end


"""
    ca_mask(at) -> BitVector

Alpha carbons: atom name `CA` **and** element `C`. Allocates one `BitVector`.

The element test is not redundant. A calcium ion is also named `CA`, and `infer_element`
exists precisely because that collision is unresolvable from the name alone. A Ca(2+)
silently entering a C-alpha RMSD would move the metric by a residue's worth of error
with nothing to show for it.
"""
function ca_mask(at::AtomTable)
    m = falses(length(at))
    @inbounds for i in eachindex(m)
        m[i] = at.atomname[i] == "CA" && at.element[i] == "C"
    end
    return m
end

"""Backbone atom test: `N`, `C`, `O`, or a C-alpha that really is carbon."""
@inline function _is_backbone(at::AtomTable, i::Integer)
    an = at.atomname[i]
    return an == "N" || an == "C" || an == "O" ||
           (an == "CA" && at.element[i] == "C")
end

"""
    residue_ids(at) -> (ids, nres)

Dense `1:nres` residue label per atom, keyed on `(chain, resnum, inscode)`.
Allocates; analysis-time.

Keyed rather than assumed contiguous: insertion codes and multi-chain input make
"atoms of one residue are adjacent" a property of most files rather than of the
format.
"""
function residue_ids(at::AtomTable)
    ids = Vector{Int32}(undef, length(at))
    seen = Dict{Tuple{String,Int32,Char},Int32}()
    nid = Int32(0)
    @inbounds for i in eachindex(ids)
        k = (at.chain[i], at.resnum[i], at.inscode[i])
        id = get(seen, k, Int32(0))
        if id == 0
            nid += Int32(1)
            id = nid
            seen[k] = id
        end
        ids[i] = id
    end
    return (ids, Int(nid))
end

"""
    interface_masks(rec, lig; reccoords, ligcoords, cutoff=10.0, backbone=true)
        -> (recmask, ligmask)

Atoms belonging to *interface residues*: a residue is at the interface if any of
its heavy atoms lies within `cutoff` of any heavy atom of the partner. With
`backbone = true` the returned masks additionally keep only `N, CA, C, O`.

`O(natoms_rec * natoms_lig)` -- about 7e5 distance tests for E2A-HPr, which is
analysis-time and not worth a neighbour list. Allocates the two masks plus the
residue labelling.

!!! note "This convention is adopted, not cited"
    Neither document in the third-party inventory defines an interface RMSD. The
    preprint's own metric, and the one this project reports as its headline, is the
    C-alpha ligand RMSD of PDF p.22; see [`ligand_rmsd`](@ref).
"""
function interface_masks(rec::AtomTable{T}, lig::AtomTable{T};
                         reccoords::AbstractVector{SVector{3,T}} = rec.coords,
                         ligcoords::AbstractVector{SVector{3,T}} = lig.coords,
                         cutoff::Real = 10.0, backbone::Bool = true) where {T}
    length(reccoords) == length(rec) ||
        throw(DimensionMismatch("reccoords must be one per receptor atom"))
    length(ligcoords) == length(lig) ||
        throw(DimensionMismatch("ligcoords must be one per ligand atom"))
    rid, nr = residue_ids(rec)
    lid, nl = residue_ids(lig)
    rhot = falses(nr)
    lhot = falses(nl)
    c2 = T(cutoff)^2
    @inbounds for j in eachindex(ligcoords)
        lj = ligcoords[j]
        for i in eachindex(reccoords)
            d = reccoords[i] - lj
            if dot(d, d) <= c2
                rhot[rid[i]] = true
                lhot[lid[j]] = true
            end
        end
    end
    recmask = falses(length(rec))
    ligmask = falses(length(lig))
    @inbounds for i in eachindex(recmask)
        recmask[i] = rhot[rid[i]] && (!backbone || _is_backbone(rec, i))
    end
    @inbounds for j in eachindex(ligmask)
        ligmask[j] = lhot[lid[j]] && (!backbone || _is_backbone(lig, j))
    end
    return (recmask, ligmask)
end


"""
    ligand_rmsd(a, b; mask=nothing) -> T

RMSD between two placements of the same atoms, **without re-superposition**.
Allocates nothing.

This is the preprint's metric (PDF p.22): the receptor frame is fixed, so the
ligand RMSD is a direct comparison and not a fit. Pass [`ca_mask`](@ref) as
`mask` to get the C-alpha figure the paper's 10 A hit threshold is defined on.

Unlike [`pairwise_rmsd!`](@ref) this evaluates the definition directly, so it
carries no cancellation floor and is the right function for a single pair.
"""
function ligand_rmsd(a::AbstractVector{SVector{3,T}}, b::AbstractVector{SVector{3,T}};
                     mask::Union{Nothing,AbstractVector{Bool}} = nothing) where {T}
    length(a) == length(b) || throw(DimensionMismatch("a and b differ in length"))
    s = zero(T)
    m = 0
    @inbounds for k in eachindex(a)
        (mask === nothing || mask[k]) || continue
        d = a[k] - b[k]
        s += dot(d, d)
        m += 1
    end
    m == 0 && throw(ArgumentError("mask selects no atoms"))
    return sqrt(s / T(m))
end

"""
    kabsch(P, Q) -> (R, t)

Rigid transform minimising `sum_k ||R p_k + t - q_k||^2`, by the SVD of the
cross-covariance with the reflection determinant fixed. Allocates a few 3x3
matrices; analysis-time.

The determinant correction is not optional: without it the SVD will happily
return an improper rotation, which superimposes a structure on its own mirror
image and reports a *smaller* RMSD than any rigid motion can achieve.
"""
function kabsch(P::AbstractVector{SVector{3,T}}, Q::AbstractVector{SVector{3,T}}) where {T}
    n = length(P)
    n == length(Q) || throw(DimensionMismatch("P and Q differ in length"))
    n >= 3 || throw(ArgumentError("need at least 3 points to superimpose, got $n"))
    pbar = sum(P) / T(n)
    qbar = sum(Q) / T(n)
    H = zero(SMatrix{3,3,T,9})
    @inbounds for k in 1:n
        H += (P[k] - pbar) * (Q[k] - qbar)'
    end
    F = LinearAlgebra.svd(Matrix{T}(H))
    d = sign(LinearAlgebra.det(F.V * F.U'))
    D = SMatrix{3,3,T,9}(one(T), 0, 0, 0, one(T), 0, 0, 0, T(d))
    R = SMatrix{3,3,T,9}(F.V) * D * SMatrix{3,3,T,9}(F.U)'
    return (R, qbar - R * pbar)
end

"""
    superposed_rmsd(P, Q) -> T

RMSD after optimal rigid superposition of `P` onto `Q` ([`kabsch`](@ref)).
Allocates as `kabsch` does.
"""
function superposed_rmsd(P::AbstractVector{SVector{3,T}},
                         Q::AbstractVector{SVector{3,T}}) where {T}
    R, t = kabsch(P, Q)
    s = zero(T)
    @inbounds for k in eachindex(P)
        d = R * P[k] + t - Q[k]
        s += dot(d, d)
    end
    return sqrt(s / T(length(P)))
end

"""
    interface_rmsd(rec, lig, native_lig, model_lig; cutoff=10.0, backbone=true) -> T

Interface RMSD: superimpose the interface atoms of the modelled complex on those
of the native complex and take the residual. Interface membership is defined on
the **native** geometry, so it is one fixed atom set across every pose -- which
is also what keeps this comparable pose to pose. Allocates; analysis-time.

The receptor contributes its interface atoms unchanged (it does not move in this
frame) and the ligand contributes the same atoms in the two placements. Read
[`interface_masks`](@ref) on why the convention is `[ASSUMED]` rather than cited.
"""
function interface_rmsd(rec::AtomTable{T}, lig::AtomTable{T},
                        native_lig::AbstractVector{SVector{3,T}},
                        model_lig::AbstractVector{SVector{3,T}};
                        cutoff::Real = 10.0, backbone::Bool = true) where {T}
    recmask, ligmask = interface_masks(rec, lig; ligcoords = native_lig,
                                       cutoff = cutoff, backbone = backbone)
    nat = SVector{3,T}[]
    mod = SVector{3,T}[]
    @inbounds for i in eachindex(recmask)
        if recmask[i]
            push!(nat, rec.coords[i])
            push!(mod, rec.coords[i])
        end
    end
    @inbounds for j in eachindex(ligmask)
        if ligmask[j]
            push!(nat, native_lig[j])
            push!(mod, model_lig[j])
        end
    end
    isempty(nat) && throw(ArgumentError(
        "no interface atoms at cutoff $cutoff A -- the two chains do not touch"))
    return superposed_rmsd(mod, nat)
end

"""
    subset_frame(Rs, ts, y, mask) -> (ts_sub, S_sub, M)

Re-express a set of poses in the frame of an atom **subset**, so that
[`pairwise_rmsd!`](@ref)'s closed form evaluates the RMSD over that subset
rather than over the whole ligand. Allocates one translation vector;
analysis-time.

# Why this is not just "pass a different S"

The identity `RMSD^2 = ||t_i - t_j||^2 + (2/M)(tr S - tr(R_i' R_j S))` drops a
cross term `2 (t_i - t_j)' (R_i - R_j) sum_k y_k`, and that term vanishes only
because the coordinates are centred. Centring is over **all** atoms, so on a
subset `sum_k y_k != 0` and the cross term is back. Handing `pairwise_rmsd!` a
subset `S` while leaving the translations alone therefore returns a number that
is neither the subset RMSD nor the whole-ligand one -- a plausible wrong answer,
silently.

The fix is exact and costs one matrix-vector product per pose: with `c` the
subset centroid, `R y_k + t = R (y_k - c) + (t + R c)`, so the same placement is
described by subset-centred coordinates and the shifted translation `t + R c`.

The precondition the docstring of [`pairwise_rmsd!`](@ref) states -- the subset must be
*fixed across poses* -- is unchanged and is not something this function can check.
"""
function subset_frame(Rs::AbstractVector{<:SMatrix{3,3,T}},
                      ts::AbstractVector{<:SVector{3,T}},
                      y::AbstractVector{SVector{3,T}},
                      mask::AbstractVector{Bool}) where {T}
    length(Rs) == length(ts) || throw(DimensionMismatch("Rs and ts differ in length"))
    length(mask) == length(y) || throw(DimensionMismatch("mask must be one per atom"))
    c = zero(SVector{3,T}); M = 0
    @inbounds for k in eachindex(y)
        mask[k] || continue
        c += y[k]; M += 1
    end
    M == 0 && throw(ArgumentError("mask selects no atoms"))
    c /= T(M)
    S = zero(SMatrix{3,3,T,9})
    @inbounds for k in eachindex(y)
        mask[k] || continue
        d = y[k] - c
        S += d * d'
    end
    ts_sub = [ts[i] + Rs[i] * c for i in eachindex(Rs)]
    return (ts_sub, S, M)
end


"""
    SearchFloor{T}

The best pose the discrete search space contains, and where it is:

  * `rmsd`  -- lowest attainable RMSD over (rotation set) x (integer cell shifts)
  * `rot`   -- rotation index attaining it
  * `cells` -- the cell offsets, already minimal-image signed
  * `angle` -- geodesic angle in degrees between `rots[rot]` and the rotation
               that would be exactly right

Not a result of docking: no energy is consulted anywhere in its construction.
It is the yardstick a docking result has to be read against, because a
bound-bound run cannot return 0 A -- the native orientation is almost surely not
in the rotation set and the native offset is almost surely not on the lattice.
"""
struct SearchFloor{T<:AbstractFloat}
    rmsd::T
    rot::Int
    cells::NTuple{3,Int}
    angle::T
end

function Base.show(io::IO, f::SearchFloor{T}) where {T}
    print(io, "SearchFloor{", T, "}: ", round(f.rmsd; digits = 3),
          " A at rotation ", f.rot, " (", round(f.angle; digits = 2),
          " deg off), cells ", f.cells)
end

"""
    search_floor(rots, spec, y, z; mask=nothing) -> SearchFloor

Minimum RMSD between `z` (the native ligand, in the receptor frame) and any
`R*y + t` with `R` in `rots` and `t` on the translation lattice. Allocates
nothing beyond the result; `O(nrot * natoms)`.

This is exact, not a bound on a bound. For a fixed `R` the objective is
`M*|t - t*|^2 + const` with `t* = zbar - R*ybar`, so rounding `t*` to the lattice
per axis is the true minimiser -- there is no search over translations to get
wrong. The rotation set is then enumerated.

**Why it is worth the arithmetic.** Reporting "the native pose came back at
1.8 A" invites the reading that 1.8 A is error. Most of it is resolution: the
rotation set has a covering radius of several degrees and the lattice is 1.2 A,
and this function says how much of the residual those two alone account for.
A docked RMSD *below* this number is not a good result, it is a bug -- and that
is a check no other test in the suite can make.
"""
function search_floor(rots::AbstractVector{<:SMatrix{3,3,T}}, spec::GridSpec{T},
                      y::AbstractVector{SVector{3,T}},
                      z::AbstractVector{SVector{3,T}};
                      mask::Union{Nothing,AbstractVector{Bool}} = nothing) where {T}
    length(y) == length(z) || throw(DimensionMismatch("y and z differ in length"))
    h = spec.spacing

    ybar = zero(SVector{3,T}); zbar = zero(SVector{3,T}); M = 0
    @inbounds for k in eachindex(y)
        (mask === nothing || mask[k]) || continue
        ybar += y[k]; zbar += z[k]; M += 1
    end
    M == 0 && throw(ArgumentError("mask selects no atoms"))
    ybar /= T(M); zbar /= T(M)

    best = T(Inf); bestrot = 0; bestcells = (0, 0, 0)
    invM = one(T) / T(M)
    @inbounds for r in eachindex(rots)
        R = rots[r]
        tstar = zbar - R * ybar

        cx = round(Int, -tstar[1] / h)
        cy = round(Int, -tstar[2] / h)
        cz = round(Int, -tstar[3] / h)
        t = -h * SVector{3,T}(T(cx), T(cy), T(cz))
        s = zero(T)
        for k in eachindex(y)
            (mask === nothing || mask[k]) || continue
            d = R * y[k] + t - z[k]
            s += dot(d, d)
        end
        v = sqrt(s * invM)
        if v < best
            best = v; bestrot = r; bestcells = (cx, cy, cz)
        end
    end

    ang = T(rad2deg(_rot_angle(rots[bestrot])))
    return SearchFloor{T}(best, bestrot, bestcells, ang)
end

search_floor(rs::RotationSet{T}, spec::GridSpec{T}, y, z; kwargs...) where {T} =
    search_floor(rs.rots, spec, y, z; kwargs...)

"""Geodesic angle of a rotation matrix, in radians. Numerically clamped."""
@inline function _rot_angle(R::SMatrix{3,3,T}) where {T}
    c = (tr(R) - one(T)) / T(2)
    return acos(clamp(c, -one(T), one(T)))
end

"""
    hit_rank(rmsds, threshold) -> Int

Index of the first entry of `rmsds` at or below `threshold`, or `0` if there is
none. `rmsds` is expected in **energy order**, so the return value is the rank
of the best-scoring near-native pose. Allocates nothing.
"""
function hit_rank(rmsds::AbstractVector{<:Real}, threshold::Real)
    @inbounds for i in eachindex(rmsds)
        rmsds[i] <= threshold && return i
    end
    return 0
end
