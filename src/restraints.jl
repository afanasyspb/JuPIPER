"""
    Restraint{T}

One pairwise distance restraint: satisfied at a translation when some atom of
`rec` is between `dmin` and `dmax` of some atom of `lig`.

`rec` indexes the **receptor** atom table and `lig` the **ligand** `Structure`'s
centred coordinates -- two different frames, which is the trap here and is why
they are separate fields rather than one list of pairs.
"""
struct Restraint{T<:AbstractFloat}
    rec::Vector{Int32}
    lig::Vector{Int32}
    dmin::T
    dmax::T
end

"""
    restraint(rec, lig, dmin, dmax; T = Float64) -> Restraint{T}

Checked constructor. An empty atom set is refused rather than silently
satisfied-nowhere: it is what a selection that matched no residue produces, and
it would otherwise make the whole set infeasible at every rotation and be read
as "the restraints are too tight".
"""
function restraint(rec, lig, dmin::Real, dmax::Real; T::Type = Float64)
    isempty(rec) && throw(ArgumentError("restraint has no receptor atoms -- a " *
                                        "selection that matched nothing makes every " *
                                        "translation infeasible"))
    isempty(lig) && throw(ArgumentError("restraint has no ligand atoms -- see above"))
    dmin >= 0 || throw(ArgumentError("dmin must be non-negative, got $dmin"))
    dmax > dmin || throw(ArgumentError("dmax ($dmax) must exceed dmin ($dmin)"))
    return Restraint{T}(Int32.(collect(rec)), Int32.(collect(lig)), T(dmin), T(dmax))
end

"""
    RestraintGroup{T}

Restraints satisfied together at a translation when **more than `cutoff`** of
them are individually satisfied there.

`cutoff = 0` -- the default -- is the union: any one member suffices. `cutoff =
length(members) - 1` is the intersection. The `>` rather than `>=` is the
paper's own wording, "translations that appear more often than the cutoff"
(2017 p.8), kept literally so the mapping to the published algorithm needs no
argument.
"""
struct RestraintGroup{T<:AbstractFloat}
    members::Vector{Restraint{T}}
    cutoff::Int
end

function restraint_group(members::AbstractVector{Restraint{T}}; cutoff::Integer = 0) where {T}
    isempty(members) && throw(ArgumentError("a restraint group needs at least one member"))
    0 <= cutoff < length(members) ||
        throw(ArgumentError("cutoff must be in 0:$(length(members)-1) -- a cutoff " *
                            "of $(length(members)) can never be exceeded, so the " *
                            "group would be unsatisfiable everywhere"))
    return RestraintGroup{T}(collect(members), Int(cutoff))
end

"""
    RestraintSet{T}

Groups merged the same way: feasible where **more than `cutoff`** groups are.
"""
struct RestraintSet{T<:AbstractFloat}
    groups::Vector{RestraintGroup{T}}
    cutoff::Int
end

function restraint_set(groups::AbstractVector{RestraintGroup{T}};
                       cutoff::Integer = 0) where {T}
    isempty(groups) && throw(ArgumentError("a restraint set needs at least one group"))
    0 <= cutoff < length(groups) ||
        throw(ArgumentError("cutoff must be in 0:$(length(groups)-1)"))
    return RestraintSet{T}(collect(groups), Int(cutoff))
end

"""
    restraint_set(restraints; cutoff = 0) -> RestraintSet

The flat case: every restraint its own group, so `cutoff` counts *restraints*.
This is what an NOE list is -- 20 independent distance observations, of which
one may want at least `k` satisfied.
"""
restraint_set(rs::AbstractVector{Restraint{T}}; cutoff::Integer = 0) where {T} =
    restraint_set([restraint_group([r]) for r in rs]; cutoff = cutoff)

nrestraints(s::RestraintSet) = sum(length(g.members) for g in s.groups)

Base.length(s::RestraintSet) = length(s.groups)

function Base.show(io::IO, s::RestraintSet{T}) where {T}
    print(io, "RestraintSet{", T, "}(", length(s.groups), " groups, ",
          nrestraints(s), " restraints, cutoff ", s.cutoff, ")")
end


"""
    RestraintWorkspace{T}

Scratch for [`feasible_translations!`](@ref). One per concurrent task, for the
same reason a [`Workspace`](@ref) is: it is mutable state indexed by cell, and
two tasks sharing one would interleave their stamps silently.

The counters are dense (`ncells`) but are **cleared through the touched list**,
not by `fill!`. That is the difference between `O(touched)` and `O(ncells)` per
rotation, and at 70,000 rotations on a 340,000-cell grid it is the difference
between microseconds and a run that costs more than the docking it filters.
"""
mutable struct RestraintWorkspace{T<:AbstractFloat}
    stamp::Vector{Int32}
    gcount::Vector{Int16}
    scount::Vector{Int16}
    gtouched::Vector{Int32}
    stouched::Vector{Int32}
    feasible::Vector{Int32}
    token::Int32
end

"""
    restraint_workspace(T, spec) -> RestraintWorkspace{T}

Allocate scratch for one task.

**8 bytes per cell** -- one `Int32` stamp and two `Int16` counters -- so
**16.0 MiB at 128^3**, which is *twice* the Float32 `score` it filters and equal
to the Float64 one. Multiply by the pool size: one workspace per task is not
optional (they are mutable per-cell state, and two tasks sharing one would
interleave their stamps silently), so an 18-thread host pays ~288 MiB at that
grid. That is the price of clearing through the touched list instead of
`fill!`ing, and it is the right trade at 70,000 rotations -- but it is a real
cost and is stated rather than buried.
"""
function restraint_workspace(::Type{T}, spec::GridSpec) where {T<:AbstractFloat}
    nc = prod(spec.n)
    return RestraintWorkspace{T}(zeros(Int32, nc), zeros(Int16, nc), zeros(Int16, nc),
                                 Int32[], Int32[], Int32[], Int32(0))
end


@inline _mi_range(n::Integer) = ((n >> 1) - n + 1, n >> 1)

"""
    feasible_translations!(ws, set, rec, lig, R, spec) -> Vector{Int32}

Linear cell indices of the translations satisfying `set` for rotation `R`,
sorted ascending. Mutates `ws` and returns a view of its buffer -- valid until
the next call on the same workspace.

`rec` are receptor coordinates in the grid's frame; `lig` are the **centred**
ligand coordinates a [`Structure`](@ref) holds.

# The convention, which is the whole correctness question

[`pose_transform`](@ref) places ligand atom `y` at `R*y + t` with
`t = -spacing * minimal_image(cell)`. So with `v = x_i - R*y_j` the restraint
reads `dmin <= ||v + h*m|| <= dmax`, where `m` is the signed cell offset. The
feasible `m` therefore form a spherical **shell centred at `-v/h`**, and the
sign there is not a free choice: getting it backwards mirrors the feasible set
through the receptor centre, which leaves every count and every timing
unchanged and silently restrains the wrong half of the search. It is pinned by
a test that places a single atom pair at a known separation, not by this
comment.
"""
function feasible_translations!(ws::RestraintWorkspace{T}, set::RestraintSet{T},
                                rec::AbstractVector{SVector{3,T}},
                                lig::AbstractVector{SVector{3,T}},
                                R::SMatrix{3,3,T}, spec::GridSpec{T}) where {T}
    empty!(ws.feasible); empty!(ws.stouched)
    nx, ny, nz = spec.n
    h = spec.spacing
    for g in set.groups
        empty!(ws.gtouched)
        for r in g.members
            ws.token += Int32(1)
            tok = ws.token
            for i in r.rec, j in r.lig
                _stamp_shell!(ws, rec[i] - R * lig[j], r.dmin, r.dmax, h,
                              nx, ny, nz, tok)
            end
        end
        @inbounds for c in ws.gtouched
            if ws.gcount[c] > g.cutoff
                ws.scount[c] == 0 && push!(ws.stouched, c)
                ws.scount[c] += Int16(1)
            end
            ws.gcount[c] = Int16(0)
        end
    end
    @inbounds for c in ws.stouched
        ws.scount[c] > set.cutoff && push!(ws.feasible, c)
        ws.scount[c] = Int16(0)
    end
    sort!(ws.feasible)
    return ws.feasible
end


@inline function _stamp_shell!(ws::RestraintWorkspace{T}, v::SVector{3,T},
                               dmin::T, dmax::T, h::T,
                               nx::Int, ny::Int, nz::Int, tok::Int32) where {T}
    inv_h = inv(h)
    d2lo = dmin * dmin
    d2hi = dmax * dmax
    c = -v * inv_h
    rc = dmax * inv_h
    lo1, hi1 = _mi_range(nx); lo2, hi2 = _mi_range(ny); lo3, hi3 = _mi_range(nz)
    a1 = max(lo1, ceil(Int, c[1] - rc)); b1 = min(hi1, floor(Int, c[1] + rc))
    a2 = max(lo2, ceil(Int, c[2] - rc)); b2 = min(hi2, floor(Int, c[2] + rc))
    a3 = max(lo3, ceil(Int, c[3] - rc)); b3 = min(hi3, floor(Int, c[3] + rc))
    @inbounds for m3 in a3:b3
        w3 = v[3] + h * m3
        s3 = w3 * w3
        s3 > d2hi && continue
        k3 = mod(m3, nz)
        for m2 in a2:b2
            w2 = v[2] + h * m2
            s23 = s3 + w2 * w2
            s23 > d2hi && continue
            k2 = mod(m2, ny)
            base = 1 + nx * (k2 + ny * k3)
            for m1 in a1:b1
                w1 = v[1] + h * m1
                s = s23 + w1 * w1
                (s > d2hi || s < d2lo) && continue
                cell = base + mod(m1, nx)
                if ws.stamp[cell] != tok
                    ws.stamp[cell] = tok
                    ws.gcount[cell] == 0 && push!(ws.gtouched, Int32(cell))
                    ws.gcount[cell] += Int16(1)
                end
            end
        end
    end
    return nothing
end


"""
    RestraintReport

What a restrained run actually did, so a cost claim can be measured rather than
inherited from the reference (see the header of this file).

`nskipped` is the only quantity that removes transform work: those rotations
had no feasible translation and their FFTs were never issued.
"""
struct RestraintReport
    nrot::Int
    nskipped::Int
    ncells::Int
    feasible_total::Int
    t_feasible::Float64
    t_dock::Float64
end

"""
    skip_rate(report) -> Float64

Fraction of rotations whose feasible set was empty -- the fraction of the
transform work the restraints removed.
"""
skip_rate(r::RestraintReport) = r.nrot == 0 ? 0.0 : r.nskipped / r.nrot

"""
    feasible_fraction(report) -> Float64

Mean fraction of the translation grid that survived, over the rotations that
were not skipped. This is the *scan* saving and it removes no transform work.
"""
function feasible_fraction(r::RestraintReport)
    live = r.nrot - r.nskipped
    return live == 0 ? 0.0 : r.feasible_total / (live * r.ncells)
end

function Base.show(io::IO, r::RestraintReport)
    print(io, "RestraintReport(", r.nrot, " rotations, ", r.nskipped,
          " skipped = ", round(100 * skip_rate(r); digits = 2), "%, ",
          round(100 * feasible_fraction(r); digits = 3), "% of cells feasible)")
end

"""
    dock_restrained!(poses, lig, rots, rs, pool, set, rec; kwargs...) -> (poses, report)

Docking under a [`RestraintSet`](@ref): each rotation's feasible translations
are computed first, and only those are scored.

**Rotations with an empty feasible set are not evaluated, and their pose slot
gets `typemax(T)` energy** -- the same sentinel and the same contract as
[`dock_pruned!`](@ref). A restrained run genuinely has no admissible pose there,
and writing a plausible number would be a lie the caller could not detect.

`rec` are the receptor coordinates in the grid frame -- the same frame
`gridspec_for` was built from.

See [`RestraintReport`](@ref) for what is measured and the file header for what
must not be claimed from it.
"""
function dock_restrained!(poses::AbstractVector{Pose{T}}, lig::Structure{T},
                          rots::AbstractVector, rs::ReceptorSpectra{T},
                          pool::AbstractVector{<:Workspace{T}},
                          set::RestraintSet{T},
                          rec::AbstractVector{SVector{3,T}}) where {T}
    nrot = length(rots)
    length(poses) == nrot || throw(DimensionMismatch("one pose slot per rotation"))
    all(w -> w.batch == 1, pool) ||
        throw(ArgumentError("dock_restrained! requires batch == 1 workspaces"))
    isempty(pool) && throw(ArgumentError("pool must not be empty"))

    spec = first(pool).spec
    ncells = prod(spec.n)
    nchunks = length(pool)
    chunks = collect(Iterators.partition(1:nrot, cld(nrot, nchunks)))
    @assert length(chunks) <= nchunks "chunking invariant violated"

    nskip = zeros(Int, length(chunks))
    ftot  = zeros(Int, length(chunks))
    tfeas = zeros(Float64, length(chunks))
    tdock = zeros(Float64, length(chunks))

    @sync for (ic, chunk) in enumerate(chunks)
        ws = pool[ic]
        rw = restraint_workspace(T, spec)
        Threads.@spawn begin
            for i in chunk
                R = rots[i]
                t0 = time_ns()
                cells = feasible_translations!(rw, set, rec, lig.coords, R, spec)
                tfeas[ic] += (time_ns() - t0) * 1e-9
                if isempty(cells)
                    nskip[ic] += 1
                    poses[i] = Pose{T}(typemax(T), Int32(i), Int16(0), Int16(0), Int16(0))
                    continue
                end
                ftot[ic] += length(cells)
                t1 = time_ns()
                correlate!(ws, lig, view(rots, i:i), rs)
                best = typemax(T); bi = Int(cells[1])
                sc = ws.score
                @inbounds for c in cells
                    v = sc[c]
                    if v < best
                        best = v; bi = Int(c)
                    end
                end
                tdock[ic] += (time_ns() - t1) * 1e-9
                x, y, z = unravel(spec, bi)
                poses[i] = Pose{T}(best, Int32(i), Int16(x), Int16(y), Int16(z))
            end
        end
    end

    return (poses, RestraintReport(nrot, sum(nskip), ncells, sum(ftot),
                                   sum(tfeas), sum(tdock)))
end
