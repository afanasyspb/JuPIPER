"""
    FibreBound{T}

Setup and scratch for the per-fibre lower bound. Built once per docking job;
`envelopes!` then reuses it for every fibre, allocating nothing.

The receptor side is `2 * nterms` spectra -- the sign-split halves of the same
weighted grids the energy uses. They are produced by [`spectra_from_grids`](@ref),
the *same* function that builds the energy's spectra, so the conjugation, the
`1/N` and the term order cannot drift between the bound and the thing it bounds.
By linearity `pos.data[p] + neg.data[p] == rs.data[p]` exactly, which
`test_fibre.jl` asserts.
"""


struct FibreBound{T<:AbstractFloat,WS}
    spec::GridSpec{T}
    nterms::Int

    rs::ReceptorSpectra{T,Array{Complex{T},3}}
    envelopes::Vector{Array{T,3}}
    stamp::Array{Int32,3}
    tmp::Array{T,4}
    ws::WS
end

"""
How [`envelopes!`](@ref) brackets the ligand grid over a group of rotations.

The two modes exist so that a *loose bound* and a *loose approximation* can be
told apart. If the cheap mode fails to prune, the exact mode says whether that
is a property of the bound or only of the shortcut used to compute it -- which
is the difference between a result and a bug.
"""
abstract type EnvelopeMode end

"""
Each atom contributes its **full** weight to every cell its stencil reaches at
any rotation in the group. Uses `max_psi sum_a (...) <= sum_a max_psi (...)`
plus `trilinear weight <= 1`. One pass of `natoms * |group| * 8` stencil
touches, no grid-sized reduction.

Cheap and **loose in a way that does not vanish at `|group| = 1`**: replacing a
trilinear fraction by 1 inflates the deposited mass by up to 8x for a single
rotation, before any angular smearing.
"""
struct SweptSet <: EnvelopeMode end

"""
`Lmax[j] = max over the group of L_psi[j]`, and likewise the minimum -- the
tightest envelope that exists, computed by voxelising every member and reducing
elementwise. Exact at `|group| = 1`, where it reduces to the ligand grid itself
and the bound becomes the energy.

Costs `nterms * |group|` voxelisations plus the same number of full-grid
reductions, which is more memory traffic than the group's own transforms. **A
diagnostic, not a production path**: its purpose is to measure how much of the
[`SweptSet`](@ref) bound's slack is intrinsic to bracketing and how much is the
shortcut.
"""
struct PerCellExact <: EnvelopeMode end

nterms(fb::FibreBound) = fb.nterms

function Base.show(io::IO, fb::FibreBound{T}) where {T}
    print(io, "FibreBound{", T, "}(", fb.nterms, " terms, grid ", fb.spec.n,
          ", ", 2 * fb.nterms + 1, " transforms per fibre)")
end

"""
    fibre_bound(backend, grids, spec, weights; flags) -> FibreBound

Build the bound from the **same real receptor grids** the energy is built from.
`weights` is [`receptor_weights`](@ref)'s vector; it is applied here before the
sign split, because a negative weight exchanges the two halves and splitting
first would silently pair each half with the wrong envelope.

Allocates `2*nterms` spectra plus `nterms` real envelopes plus a workspace;
setup-time, once per job.
"""
function fibre_bound(backend::Backend, grids::AbstractVector{<:AbstractArray{T,3}},
                     spec::GridSpec{T}, weights::AbstractVector{T};
                     flags = FFTW.MEASURE) where {T}
    n = length(grids)
    length(weights) == n || throw(DimensionMismatch("one weight per grid required"))
    nx, ny, nz = spec.n

    halves = Vector{Array{T,3}}(undef, 2n)
    for p in 1:n
        g = Array{T,3}(undef, nx, ny, nz)
        @. g = max(weights[p] * grids[p], zero(T))
        halves[p] = g
        h = Array{T,3}(undef, nx, ny, nz)
        @. h = min(weights[p] * grids[p], zero(T))
        halves[n+p] = h
    end
    rs = spectra_from_grids(backend, halves, spec, ones(T, 2n); flags = flags)
    envelopes = [zeros(T, nx, ny, nz) for _ in 1:(2n)]
    ws = workspace(backend, spec, 2n; batch = 1, flags = flags)
    return FibreBound{T,typeof(ws)}(spec, n, rs, envelopes,
                                    zeros(Int32, nx, ny, nz),
                                    Array{T,4}(undef, nx, ny, nz, 1), ws)
end

"""
    envelopes!(fb, lig, rots) -> fb

Fill the bracketing envelopes for the group of rotations `rots`. Mutates `fb`;
allocates nothing after construction.

`rots` is any collection of rotation matrices -- **soundness does not depend on them
forming a Hopf fibre.** The bound holds for an arbitrary group of rotations; what the
fibre structure buys is that its members are similar enough for the bracket to be tight,
which is a question about usefulness and is settled by measurement, not by the
inequality.

Boundary handling matches [`voxelize!`](@ref) by construction: both drop the
part of a stencil that falls outside the grid, so a clipped atom contributes to
neither the energy nor the envelope at the cells it cannot reach.
"""
function envelopes!(fb::FibreBound{T}, lig::Structure{T},
                    rots::AbstractVector, ::PerCellExact) where {T}
    n = fb.nterms
    size(lig.termw, 2) == n ||
        throw(DimensionMismatch("ligand has $(size(lig.termw,2)) terms, bound has $n"))
    isempty(rots) && throw(ArgumentError("an empty group has no envelope"))
    v = reshape(fb.tmp, fb.spec.n...)
    for p in 1:n
        for r in eachindex(rots)
            voxelize!(fb.tmp, lig, view(rots, r:r), p, fb.spec)
            if r == 1
                copyto!(fb.envelopes[p], v)
                copyto!(fb.envelopes[n+p], v)
            else
                @. fb.envelopes[p] = min(fb.envelopes[p], v)
                @. fb.envelopes[n+p] = max(fb.envelopes[n+p], v)
            end
        end
    end
    return fb
end

function envelopes!(fb::FibreBound{T}, lig::Structure{T},
                    rots::AbstractVector, ::SweptSet = SweptSet()) where {T}
    n = fb.nterms
    size(lig.termw, 2) == n ||
        throw(DimensionMismatch("ligand has $(size(lig.termw,2)) terms, bound has $n"))
    nx, ny, nz = fb.spec.n
    for e in fb.envelopes
        fill!(e, zero(T))
    end

    fill!(fb.stamp, Int32(0))

    origin = fb.spec.origin
    inv_h = inv(fb.spec.spacing)
    coords = lig.coords
    W = lig.termw
    @inbounds for a in eachindex(coords)
        y = coords[a]
        ai = Int32(a)
        for R in rots
            g = (R * y - origin) * inv_h
            i0 = unsafe_trunc(Int, floor(g[1]))
            j0 = unsafe_trunc(Int, floor(g[2]))
            k0 = unsafe_trunc(Int, floor(g[3]))
            for dk in 0:1, dj in 0:1, di in 0:1
                i = i0 + di + 1; j = j0 + dj + 1; k = k0 + dk + 1
                (1 <= i <= nx && 1 <= j <= ny && 1 <= k <= nz) || continue
                fb.stamp[i, j, k] == ai && continue
                fb.stamp[i, j, k] = ai
                for p in 1:n
                    w = W[a, p]
                    if w > 0
                        fb.envelopes[n+p][i, j, k] += w
                    elseif w < 0
                        fb.envelopes[p][i, j, k] += w
                    end
                end
            end
        end
    end
    return fb
end

"""
    bound_landscape!(fb) -> Array{T,3}

Evaluate `B(a)` for every translation, into `fb.ws.score`. Mutates `fb.ws`;
allocates nothing. `2*nterms + 1` transforms.

Mirrors [`correlate!`](@ref) exactly -- same accumulate-then-invert-once
schedule, same `_store!`/`_accumulate!` kernels -- with voxelisation replaced by
a copy of the precomputed envelopes. Sharing those kernels is deliberate: the
bound and the energy must agree about what a correlation *is*, or the soundness
test would be checking two different conventions against each other.
"""
function bound_landscape!(fb::FibreBound{T}) where {T}
    ws = fb.ws
    ngrid = length(ws.accum)
    nx, ny, nz = fb.spec.n
    dst = reshape(ws.lig_real, nx, ny, nz)
    @inbounds for p in 1:(2*fb.nterms)
        copyto!(dst, fb.envelopes[p])
        mul!(ws.lig_spec, ws.fwd, ws.lig_real)
        if p == 1
            _store!(ws.accum, fb.rs.data[p], ws.lig_spec, ngrid)
        else
            _accumulate!(ws.accum, fb.rs.data[p], ws.lig_spec, ngrid)
        end
    end
    mul!(ws.score, ws.bwd, ws.accum)
    return reshape(ws.score, nx, ny, nz)
end

"""
    fibre_bound_min(fb, lig, rots) -> T

Lowest energy any member of `rots` can attain at any translation. Allocates
nothing after setup.

This is the quantity a branch-and-bound driver compares against its incumbent:
if it exceeds the K-th best energy seen so far, no member of the group can enter
the top K, and all of them may be skipped.
"""
function fibre_bound_min(fb::FibreBound{T}, lig::Structure{T},
                         rots::AbstractVector,
                         mode::EnvelopeMode = SweptSet()) where {T}
    envelopes!(fb, lig, rots, mode)
    return minimum(bound_landscape!(fb))
end

"""
    fibre_core!(core, tmp, lig, rots, spec) -> core

Mutates `core` and `tmp`; allocates nothing.

That is exact for a hard occupancy indicator, which is what the reference uses and what
this project deliberately does not. Against a trilinear grid the inclusion no longer
bounds the value, and the per-cell minimum is the statement that survives. It is also
strictly tighter than the indicator version.

Cost is `nspin` voxelisations plus `nspin` elementwise minima **of one term**,
which is affordable; the same construction applied to all `nterms` would cost
more memory traffic than the fibre's transforms, which is why
[`envelopes!`](@ref) uses the swept-set relaxation instead.
"""
function fibre_core!(core::AbstractArray{T,3}, tmp::AbstractArray{T,4},
                     lig::Structure{T}, rots::AbstractVector,
                     spec::GridSpec{T}) where {T}
    size(core) == spec.n || throw(DimensionMismatch("core must be $(spec.n)"))
    size(tmp) == (spec.n..., 1) || throw(DimensionMismatch("tmp must be $((spec.n..., 1))"))
    isempty(rots) && throw(ArgumentError("a fibre with no rotations has no core"))
    v = reshape(tmp, spec.n...)
    first = true
    for r in eachindex(rots)
        voxelize!(tmp, lig, view(rots, r:r), 1, spec)
        if first
            copyto!(core, v)
            first = false
        else
            @. core = min(core, v)
        end
    end
    return core
end


"""
    PruneReport

What a pruned run skipped, and what the skipping cost. Every field is a count
or a duration; none of them is a verdict.

`nrot_pruned / nrot` is the pruning *rate*, and it is only half the story --
`t_bound` is paid on every group including the ones that are not pruned, so a
run can prune substantially and still be slower. `speedup_vs` reports that
honestly rather than leaving it to be inferred.
"""
struct PruneReport
    ngroups::Int
    npruned::Int
    nrot::Int
    nrot_pruned::Int
    t_bound::Float64
    t_dock::Float64
    incumbent::Float64
end

prune_rate(r::PruneReport) = r.nrot == 0 ? 0.0 : r.nrot_pruned / r.nrot

function Base.show(io::IO, r::PruneReport)
    print(io, "PruneReport(", r.npruned, "/", r.ngroups, " groups pruned, ",
          r.nrot_pruned, "/", r.nrot, " rotations = ",
          round(100 * prune_rate(r); digits = 2), "%, bound ",
          round(r.t_bound; digits = 2), " s, dock ",
          round(r.t_dock; digits = 2), " s)")
end


mutable struct TopK{T}
    vals::Vector{T}
    n::Int
end
TopK{T}(K::Integer) where {T} = TopK{T}(Vector{T}(undef, K), 0)

"""Current pruning threshold: the K-th best energy, or `Inf` until K are seen."""
@inline threshold(t::TopK{T}) where {T} =
    t.n < length(t.vals) ? T(Inf) : @inbounds t.vals[t.n]

function push_topk!(t::TopK{T}, v::T) where {T}
    K = length(t.vals)
    @inbounds begin
        if t.n < K
            i = t.n
            while i >= 1 && t.vals[i] > v
                t.vals[i+1] = t.vals[i]; i -= 1
            end
            t.vals[i+1] = v
            t.n += 1
        elseif v < t.vals[K]
            i = K - 1
            while i >= 1 && t.vals[i] > v
                t.vals[i+1] = t.vals[i]; i -= 1
            end
            t.vals[i+1] = v
        end
    end
    return t
end

"""
    dock_pruned!(poses, lig, rots, rs, pool, fb; K, group, mode) -> PruneReport

Branch-and-bound docking: evaluate the lower bound once per group of rotations
and skip the group entirely when the bound proves no member can enter the top
`K`. Mutates `poses` and `fb`.

**Pruned slots are filled with `typemax(T)` energy**, not with a real score.
That is the whole contract: a pruned rotation has not been evaluated, and
writing a plausible number there would be a lie the caller could not detect.
The sentinel sorts last, so the top-`K` prefix of a pruned run is comparable to
an exhaustive one -- which is exactly the obligation `test_fibre.jl` asserts,
bit for bit.

`rots` must be a [`RotationSet`](@ref) with fibre structure: groups are formed inside a
fibre and never across two, because a group spanning two base directions has an envelope
wide enough to be useless (measured). `group` defaults to the whole fibre.

Every workspace in `pool` must have `batch == 1`. Batching interacts with the pruning
granularity -- a batch straddling a pruned and an unpruned group would have to be split
anyway.

!!! warning "This prunes nothing on the current scoring function"
    Measured on E2A-HPr: **0 groups pruned at any `K`**, because the bound's slack
    exceeds the entire energy spread by 54x even with the tightest possible envelope.
    The driver is correct and the bound is sound; the bound is not *tight*. This is
    kept, tested and documented rather than deleted because the correctness obligation
    is the expensive part and it is now discharged: a future bound drops in behind the
    same interface.
"""
function dock_pruned!(poses::AbstractVector{Pose{T}}, lig::Structure{T},
                      rots::RotationSet{T}, rs::ReceptorSpectra{T},
                      pool::AbstractVector{<:Workspace{T}}, fb::FibreBound{T};
                      K::Integer = 1000, group::Integer = rots.nspin,
                      mode::EnvelopeMode = SweptSet()) where {T}
    nrot = length(rots)
    length(poses) == nrot || throw(DimensionMismatch("one pose slot per rotation"))
    rots.nspin > 0 || throw(ArgumentError(
        "$(rots.kind) has no fibre structure; use rotations_hopf"))
    all(w -> w.batch == 1, pool) ||
        throw(ArgumentError("dock_pruned! requires batch == 1 workspaces"))
    1 <= group <= rots.nspin ||
        throw(ArgumentError("group must be in 1:$(rots.nspin), got $group"))
    K >= 1 || throw(ArgumentError("K must be positive"))

    top = TopK{T}(min(Int(K), nrot))
    ngroups = npruned = nrot_pruned = 0
    t_bound = t_dock = 0.0
    buf = Vector{Pose{T}}(undef, group)

    @inbounds for i in 1:rots.ndir
        f = fiber(rots, i)
        for lo in first(f):group:last(f)
            hi = min(lo + group - 1, last(f))
            rng = lo:hi
            ngroups += 1
            t_bound += @elapsed b = fibre_bound_min(fb, lig, view(rots.rots, rng), mode)
            if b > threshold(top)

                npruned += 1
                nrot_pruned += length(rng)
                for r in rng
                    poses[r] = Pose{T}(typemax(T), Int32(r), Int16(0), Int16(0), Int16(0))
                end
                continue
            end
            gp = view(buf, 1:length(rng))
            t_dock += @elapsed dock!(gp, lig, view(rots.rots, rng), rs, pool)
            for (j, r) in enumerate(rng)
                p = gp[j]
                poses[r] = Pose{T}(p.energy, Int32(r), p.tx, p.ty, p.tz)
                push_topk!(top, p.energy)
            end
        end
    end
    return PruneReport(ngroups, npruned, nrot, nrot_pruned, t_bound, t_dock,
                       Float64(threshold(top)))
end
