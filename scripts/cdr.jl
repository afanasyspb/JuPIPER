const _THREE2ONE = Dict(
    "ALA" => 'A', "ARG" => 'R', "ASN" => 'N', "ASP" => 'D', "CYS" => 'C',
    "GLN" => 'Q', "GLU" => 'E', "GLY" => 'G', "HIS" => 'H', "ILE" => 'I',
    "LEU" => 'L', "LYS" => 'K', "MET" => 'M', "PHE" => 'F', "PRO" => 'P',
    "SER" => 'S', "THR" => 'T', "TRP" => 'W', "TYR" => 'Y', "VAL" => 'V',
    "MSE" => 'M', "SEC" => 'U', "PYL" => 'O')

"""
    chain_residues(at, chain) -> (ids, seq)

Residues of `chain` in file order: `ids` is a vector of `(resnum, inscode)` and
`seq` the one-letter sequence, `'X'` for anything unrecognised. Allocates.

Order is the order the atoms appear in the table, which for a PDB file is
sequence order. It is deliberately not `sort`ed by residue number: a chain that
is not monotonic in `resnum` is exactly the case where the anchor rules must be
applied to the *sequence*, and sorting would hide it.
"""
function chain_residues(at, chain::AbstractString)
    ids = Tuple{Int,Char}[]
    seq = Char[]
    seen = Set{Tuple{Int,Char}}()
    for i in eachindex(at.coords)
        at.chain[i] == chain || continue
        key = (Int(at.resnum[i]), at.inscode[i])
        key in seen && continue
        push!(seen, key)
        push!(ids, key)
        push!(seq, get(_THREE2ONE, at.resname[i], 'X'))
    end
    isempty(ids) && throw(ArgumentError("chain '$chain' has no residues in this table"))
    return (ids, String(seq))
end

_at(s, i) = (1 <= i <= length(s)) ? s[i] : '\0'

function _require(cond, what, got)
    cond || error("CDR anchor check failed: $what (found '$got'). This chain " *
                  "cannot be annotated by the published sequence rules; see the " *
                  "header of scripts/cdr.jl. Refusing to guess.")
    return nothing
end

"""
    light_cdrs(seq) -> (l1, l2, l3)

Index ranges (into `seq`) of CDR-L1/L2/L3 under the Kabat u AbM union. Throws
if any published anchor is absent.
"""
function light_cdrs(seq::AbstractString)
    n = length(seq)

    cys = findall(==('C'), collect(seq[1:min(n, 30)]))
    cand = filter(c -> 18 <= c <= 27, cys)
    isempty(cand) && error("no Cys in positions 18-27 of the light chain, so " *
                           "CDR-L1's 'residue before is always a Cys' anchor " *
                           "cannot be placed")
    c1 = last(cand)
    l1_start = c1 + 1
    trp = findfirst(==('W'), collect(seq[(l1_start + 9):min(n, l1_start + 17)]))
    trp === nothing && error("no Trp 10-18 residues after the light-chain Cys, " *
                             "so CDR-L1's 'residue after is always a Trp' anchor " *
                             "cannot be placed")
    l1_end = l1_start + 9 + trp - 2
    _require(10 <= l1_end - l1_start + 1 <= 17, "CDR-L1 length in 10-17",
             l1_end - l1_start + 1)
    _require(_at(seq, l1_end + 1) == 'W', "residue after CDR-L1 is Trp",
             _at(seq, l1_end + 1))

    l2_start = l1_end + 16
    l2_end   = l2_start + 6
    _require(l2_end <= n, "CDR-L2 fits in the chain", "$l2_end > $n")

    pre2 = string(_at(seq, l2_start - 2), _at(seq, l2_start - 1))
    pre2 in ("IY", "VY", "IK", "IF") ||
        @warn "residues before CDR-L2 are '$pre2', not one of IY/VY/IK/IF; the " *
              "16-residue offset from CDR-L1 may have landed wrong"

    l3_start = l2_end + 33
    _require(l3_start <= n, "CDR-L3 start fits in the chain", "$l3_start > $n")
    _require(_at(seq, l3_start - 1) == 'C', "residue before CDR-L3 is Cys",
             _at(seq, l3_start - 1))
    l3_end = nothing
    for e in (l3_start + 6):min(n - 3, l3_start + 10)
        if _at(seq, e + 1) == 'F' && _at(seq, e + 2) == 'G' && _at(seq, e + 4) == 'G'
            l3_end = e; break
        end
    end
    l3_end === nothing && error("no Phe-Gly-XXX-Gly 7-11 residues after the " *
                                "light-chain CDR-L3 start, so its end anchor " *
                                "cannot be placed")
    return (l1_start:l1_end, l2_start:l2_end, l3_start:l3_end)
end

"""
    heavy_cdrs(seq) -> (h1, h2, h3)

Index ranges of CDR-H1/H2/H3 under the Kabat u AbM union: H1 takes the
Chothia/AbM start (4 after the Cys) and the Kabat end (before the Trp), H2 the
Kabat end. Throws if any published anchor is absent.
"""
function heavy_cdrs(seq::AbstractString)
    n = length(seq)

    cys = findall(==('C'), collect(seq[1:min(n, 30)]))
    cand = filter(c -> 16 <= c <= 26, cys)
    isempty(cand) && error("no Cys in positions 16-26 of the heavy chain, so " *
                           "CDR-H1's 'always 4 after a Cys' anchor cannot be placed")
    h1_start = last(cand) + 4
    trp = findfirst(==('W'), collect(seq[(h1_start + 9):min(n, h1_start + 12)]))
    trp === nothing && error("no Trp 10-13 residues after the heavy-chain CDR-H1 " *
                             "start, so its end anchor cannot be placed")
    h1_end = h1_start + 9 + trp - 2
    _require(10 <= h1_end - h1_start + 1 <= 12, "CDR-H1 length in 10-12 (AbM)",
             h1_end - h1_start + 1)
    _require(_at(seq, h1_end + 1) == 'W', "residue after CDR-H1 is Trp",
             _at(seq, h1_end + 1))

    h2_start = h1_end + 15
    _require(h2_start <= n, "CDR-H2 start fits in the chain", "$h2_start > $n")
    pre5 = String([_at(seq, h2_start - k) for k in 5:-1:1])
    pre5 == "LEWIG" ||
        @warn "residues before CDR-H2 are '$pre5', not the typical Leu-Glu-Trp-" *
              "Ile-Gly; the 15-residue offset from CDR-H1 may have landed wrong"
    h2_end = nothing
    for e in (h2_start + 15):min(n - 3, h2_start + 18)
        if _at(seq, e + 1) in ('K', 'R') &&
           _at(seq, e + 2) in ('L', 'I', 'V', 'F', 'T', 'A') &&
           _at(seq, e + 3) in ('T', 'S', 'I', 'A')
            h2_end = e; break
        end
    end
    h2_end === nothing && error("no Lys/Arg-[LIVFTA]-[TSIA] 16-19 residues after " *
                                "the heavy-chain CDR-H2 start, so its Kabat end " *
                                "anchor cannot be placed")

    wg = nothing
    for e in (h2_end + 10):(n - 3)
        if seq[e] == 'W' && _at(seq, e + 1) == 'G' && _at(seq, e + 3) == 'G'
            wg = e; break
        end
    end
    wg === nothing && error("no Trp-Gly-XXX-Gly after CDR-H2, so CDR-H3's end " *
                            "anchor cannot be placed")
    h3_end = wg - 1
    cbefore = findlast(==('C'), collect(seq[1:h3_end]))
    cbefore === nothing && error("no Cys before CDR-H3")
    h3_start = cbefore + 3
    _require(h3_start <= h3_end, "CDR-H3 is non-empty", "$h3_start > $h3_end")
    _require(_at(seq, h3_start - 3) == 'C', "residues before CDR-H3 are Cys-XXX-XXX",
             String([_at(seq, h3_start - k) for k in 3:-1:1]))
    return (h1_start:h1_end, h2_start:h2_end, h3_start:h3_end)
end

"""
    fab_cdr_residues(at; heavy, light) -> (cdr, loops)

`cdr` is a `Set` of `(chain, resnum)` pairs covering the union of the Kabat,
AbM and Chothia CDR definitions; `loops` is a NamedTuple of the six loops, each
a vector of `(resnum, inscode, aa)`, for reporting.

Throws if the chains carry insertion codes inside a CDR: `residue_mask`
selects on `(chain, resnum)` alone, so `100` and `100A` would be indis-
tinguishable and the mask would silently take both.
"""
function fab_cdr_residues(at; heavy::AbstractString, light::AbstractString)
    hid, hseq = chain_residues(at, heavy)
    lid, lseq = chain_residues(at, light)
    h1, h2, h3 = heavy_cdrs(hseq)
    l1, l2, l3 = light_cdrs(lseq)
    cdr = Set{Tuple{String,Int}}()
    loops = Pair{Symbol,Vector{Tuple{Int,Char,Char}}}[]
    for (nm, ch, ids, seq, rng) in ((:H1, heavy, hid, hseq, h1), (:H2, heavy, hid, hseq, h2),
                                    (:H3, heavy, hid, hseq, h3), (:L1, light, lid, lseq, l1),
                                    (:L2, light, lid, lseq, l2), (:L3, light, lid, lseq, l3))
        v = Tuple{Int,Char,Char}[]
        for i in rng
            rn, ic = ids[i]
            ic == ' ' || error("residue $ch$rn$ic inside CDR-$nm carries an " *
                               "insertion code; residue_mask selects on " *
                               "(chain, resnum) alone and could not distinguish it")
            push!(cdr, (ch, rn))
            push!(v, (rn, ic, seq[i]))
        end
        push!(loops, nm => v)
    end
    return (cdr, (; loops...))
end

"""
    noncdr_selection(at, cdr) -> BitVector

Per-atom flag for "this atom is NOT in a CDR", i.e. the set the 2017 protocol
puts repulsion on. Atoms of any chain not represented in `cdr` count as
non-CDR, which is correct for an Fab read as receptor: every constant-domain
residue is masked, and that is most of the molecule.
"""
function noncdr_selection(at, cdr::Set{Tuple{String,Int}})
    sel = falses(length(at.coords))
    for i in eachindex(at.coords)
        sel[i] = !((at.chain[i], Int(at.resnum[i])) in cdr)
    end
    return sel
end
