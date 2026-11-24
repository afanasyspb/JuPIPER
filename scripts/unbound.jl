"""
    ca_by_residue(at, chain) -> Dict{Int,Tuple{SVector{3,T},String}}

C-alpha coordinate and residue name, keyed by residue number, for one chain. `ca_mask`'s
element test is applied here too: a calcium ion is also named `CA`, and one entering a
superposition would drag it by a residue's worth of error with nothing to show for it.
"""
function ca_by_residue(at::AtomTable{T}, chain::AbstractString) where {T}
    d = Dict{Int,Tuple{SVector{3,T},String}}()
    for i in eachindex(at.coords)
        at.chain[i] == chain || continue
        (at.atomname[i] == "CA" && at.element[i] == "C") || continue
        d[Int(at.resnum[i])] = (at.coords[i], at.resname[i])
    end
    return d
end

"""
    numbering_offset(a, b) -> (delta, nvotes)

The residue-number shift taking `a`'s numbering to `b`'s, recovered as the mode
of `rb - ra` over pairs whose residue names agree. `O(|a||b|)`, ~46k for a Fab
chain, and setup-time.

Recovered rather than assumed: see the header. The vote count is returned so a
caller can tell "one clear offset" from "two sequences that share a few residue
names by chance".
"""
function numbering_offset(a::Dict, b::Dict)
    votes = Dict{Int,Int}()
    for (ra, (_, na)) in a, (rb, (_, nb)) in b
        na == nb || continue
        d = rb - ra
        votes[d] = get(votes, d, 0) + 1
    end
    isempty(votes) && error("no residue-name agreement at any offset")
    delta = 0; best = -1
    for (d, n) in votes
        n > best && (best = n; delta = d)
    end
    return (delta, best)
end

"""
    superpose_chains(src, dst, chainmap; minfrac = 0.8, verbose = true) -> (R, t, n, rmsd)

Rigid transform carrying `src`'s frame onto `dst`'s, from the C-alphas of the
chain pairs in `chainmap` (`src_chain => dst_chain`).

Throws rather than returning a poor fit: a chain pair that matches fewer than
`minfrac` of the shorter chain is a wrong correspondence, not a hard case, and
`kabsch` would happily return a transform for three atoms.
"""
function superpose_chains(src::AtomTable{T}, dst::AtomTable{T},
                          chainmap::AbstractVector{<:Pair};
                          minfrac::Real = 0.8, verbose::Bool = true) where {T}
    P = SVector{3,T}[]; Q = SVector{3,T}[]
    for (cs, cd) in chainmap
        A = ca_by_residue(src, String(cs)); B = ca_by_residue(dst, String(cd))
        (isempty(A) || isempty(B)) &&
            error("chain pair $cs => $cd has no C-alphas on one side")
        delta, _ = numbering_offset(A, B)
        n = 0
        for (r, (p, nm)) in A
            haskey(B, r + delta) || continue
            q, nm2 = B[r + delta]
            nm == nm2 || error("residue $cs$r is $nm but $cd$(r + delta) is $nm2 " *
                               "at the recovered offset $delta -- the chain " *
                               "correspondence is wrong")
            push!(P, p); push!(Q, q); n += 1
        end
        frac = n / min(length(A), length(B))
        frac >= minfrac || error("chain pair $cs => $cd matched only $n residues " *
                                 "($(round(100frac; digits = 1))% of the shorter " *
                                 "chain) at offset $delta; refusing to superimpose")
        verbose && @printf("    %-4s -> %-4s  offset %+5d  matched %4d of %4d/%4d\n",
                           String(cs), String(cd), delta, n, length(A), length(B))
    end
    R, t = kabsch(P, Q)
    s = zero(T)
    for k in eachindex(P)
        d = R * P[k] + t - Q[k]; s += dot(d, d)
    end
    return (R, t, length(P), sqrt(s / length(P)))
end

"""
    subtable(at, keep) -> AtomTable

A new table holding the atoms flagged in `keep`, with `centred = false` and a
zero centroid: the coordinates have been moved by an arbitrary rigid transform,
so the recorded centroid of the parent no longer describes them and carrying it
forward would make `native_ligand_coords`-style arithmetic silently wrong.
"""
function subtable(at::AtomTable{T}, keep::AbstractVector{Bool};
                  coords::Union{Nothing,Vector{SVector{3,T}}} = nothing) where {T}
    idx = findall(keep)
    c = coords === nothing ? at.coords[idx] : coords
    return AtomTable{T}(c, at.element[idx], at.atomname[idx], at.resname[idx],
                        at.resnum[idx], at.inscode[idx], at.chain[idx],
                        at.occupancy[idx], at.tempfactor[idx],
                        BitVector(at.is_nterm[idx]), BitVector(at.is_cterm[idx]),
                        zero(SVector{3,T}), false, at.report)
end

"""
    unbound_native(urec, ulig, nat, recmap, ligmap; verbose = true) -> (native, diag)

The native placement of the **unbound ligand's own atoms**, in the frame `urec`
is currently in -- i.e. the frame [`pose_coords`](@ref) returns poses in.

`recmap` and `ligmap` are `unbound_chain => native_chain` correspondences into
the native complex table `nat`. `diag.lig_unbound_rmsd` is the conformational
floor described in the header; nothing computed from `native` can beat it.
"""
function unbound_native(urec::AtomTable{T}, ulig::AtomTable{T}, nat::AtomTable{T},
                        recmap::AbstractVector{<:Pair},
                        ligmap::AbstractVector{<:Pair};
                        verbose::Bool = true) where {T}
    verbose && println("  receptor superposition (native -> unbound-receptor frame):")

    Rr, tr, nr, rms_r = superpose_chains(nat, urec, [Pair(b, a) for (a, b) in recmap];
                                         verbose = verbose)
    want = Set(String(last(p)) for p in ligmap)
    keep = [nat.chain[i] in want for i in eachindex(nat.coords)]
    count(keep) > 0 || error("native table has no atoms in chains $(collect(want))")
    moved = [Rr * nat.coords[i] + tr for i in eachindex(nat.coords) if keep[i]]
    natlig = subtable(nat, keep; coords = moved)
    verbose && println("  ligand superposition (unbound ligand -> native ligand placement):")
    Rl, tl, nl, rms_l = superpose_chains(ulig, natlig, ligmap; verbose = verbose)
    native = [Rl * c + tl for c in ulig.coords]
    return (native, (rec_matched = nr, rec_rmsd = rms_r,
                     lig_matched = nl, lig_unbound_rmsd = rms_l))
end
