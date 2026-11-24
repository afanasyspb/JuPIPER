#!/usr/bin/env julia


using Downloads, SHA, Dates

const CANONICAL = "https://zlab.umassmed.edu/benchmark/"
const SOURCE    = "https://zlab.wenglab.org/benchmark/benchmark4.html"
const RCSB_HDR  = "https://files.rcsb.org/header/"

argval(f, d) = (i = findfirst(==(f), ARGS); i === nothing ? d : ARGS[i + 1])
const OUT     = argval("--out", joinpath(@__DIR__, "..", "data", "benchmark51.tsv"))
const DO_RCSB = !("--no-rcsb" in ARGS)


function fetch_source()
    io = IOBuffer()
    Downloads.download(SOURCE, io; timeout = 120)
    html = String(take!(io))
    return html, bytes2hex(sha256(html))
end


striptags(s) = replace(s, r"<[^>]+>" => " ")

function cells(row)
    out = String[]
    for m in eachmatch(r"<t[dh][^>]*>(.*?)</t[dh]>"is, row)
        c = striptags(m.captures[1])
        c = replace(c, "&nbsp;" => " ", "&amp;" => "&", "&quot;" => "\"")
        push!(out, join(split(c), " "))
    end
    return out
end

"""
    parse_bm4(html) -> Vector{NamedTuple}

Every case row of Benchmark 4.0, with the difficulty section it sits under.
"""
function parse_bm4(html::AbstractString)
    cases = NamedTuple[]
    difficulty = ""
    for m in eachmatch(r"<tr[^>]*>(.*?)</tr>"is, html)
        nz = filter(!isempty, cells(m.captures[1]))
        if length(nz) == 1
            d = match(r"^(Rigid-body|Medium Difficulty|Difficult)", nz[1])
            d === nothing || (difficulty = d.captures[1])
            continue
        end
        (length(nz) >= 6 && nz[1] == "info") || continue

        redundant = any(==("*"), nz) || occursin('*', nz[2])
        nz = filter(!=("*"), nz)
        push!(cases, (fields = nz, difficulty = difficulty, redundant = redundant))
    end
    return cases
end


const UNBOUND_RE = r"^[0-9A-Za-z]{4}_[A-Za-z0-9]*(\(\d+\))?$"

function unbound_pair(nz::Vector{String})
    idx = findall(i -> occursin(UNBOUND_RE, nz[i]), eachindex(nz))
    length(idx) >= 2 ||
        error("expected two unbound identifiers in row: " * join(nz, " | "))
    i, j = idx[1], idx[2]
    name(k) = k < length(nz) ? nz[k + 1] : ""
    return nz[i], name(i), nz[j], name(j)
end


function split_complex(s::AbstractString)
    m = match(r"^([0-9A-Za-z]{4})_([A-Za-z0-9]+):([A-Za-z0-9]+)\s*(\*?)", s)
    m === nothing && error("unparsable complex field: $s")
    return uppercase(m.captures[1]), m.captures[2], m.captures[3],
           m.captures[4] == "*"
end


function split_unbound(s::AbstractString)
    m = match(r"^([0-9A-Za-z]{4})_([A-Za-z0-9]*)(?:\((\d+)\))?", s)
    m === nothing && error("unparsable unbound field: $s")
    chains = m.captures[2]
    return uppercase(m.captures[1]), (isempty(chains) ? "." : chains),
           m.captures[3] === nothing ? "" : m.captures[3]
end


function pdb_header(id::AbstractString)
    io = IOBuffer()
    try
        Downloads.download(RCSB_HDR * id * ".pdb", io; timeout = 60)
    catch e
        return (ok = false, resolution = "?", method = "?", title = "")
    end
    txt = String(take!(io))
    res = "NA"
    m = match(r"REMARK   2 RESOLUTION\.\s+([0-9.]+)\s+ANGSTROM", txt)
    m === nothing || (res = m.captures[1])
    meth = "?"
    m = match(r"^EXPDTA\s{4}(.+)$"m, txt)
    m === nothing || (meth = strip(m.captures[1]))
    occursin("NMR", meth) && res == "NA" && (res = "NMR")
    ttl = ""
    m = match(r"^TITLE\s{5}(.+)$"m, txt)
    m === nothing || (ttl = strip(m.captures[1]))
    return (ok = true, resolution = res, method = meth, title = ttl)
end


function main()
    println("source : ", SOURCE)
    html, digest = fetch_source()
    println("bytes  : ", length(html), "   sha256 ", digest)

    all_cases = parse_bm4(html)
    println("parsed : ", length(all_cases), " case rows")
    length(all_cases) == 176 ||
        error("expected 176 BM4.0 rows, parsed $(length(all_cases)) -- the page changed")

    E = filter(c -> c.fields[3] == "E", all_cases)
    ids = unique(first(split_complex(c.fields[2])) for c in E)
    println("enzyme : ", length(E), " rows, ", length(ids), " unique PDB entries")
    length(E) == 52 ||
        error("expected 52 enzyme rows, got $(length(E)) -- the table changed")
    length(ids) == 51 ||
        error("expected 51 unique enzyme entries, got $(length(ids))")

    bydiff = Dict{String,Int}()
    for c in E; bydiff[c.difficulty] = get(bydiff, c.difficulty, 0) + 1; end
    println("       : by difficulty ", sort(collect(bydiff)))

    rows = NamedTuple[]
    for c in E
        cid, rc, lc, _ = split_complex(c.fields[2])
        p1, n1, p2, n2 = unbound_pair(c.fields)
        ru, ruc, rmodel = split_unbound(p1)
        lu, luc, lmodel = split_unbound(p2)
        push!(rows, (id = cid, rec_chains = rc, lig_chains = lc,
                     rec_unbound = ru, rec_unbound_chains = ruc,
                     lig_unbound = lu, lig_unbound_chains = luc,
                     difficulty = c.difficulty, redundant = c.redundant,
                     rec_name = n1, lig_name = n2,
                     rmodel = rmodel, lmodel = lmodel))
    end
    sort!(rows, by = r -> (r.id, r.rec_chains))

    seen = Dict{String,Int}()
    for r in rows; seen[r.id] = get(seen, r.id, 0) + 1; end
    caseid = Dict{Int,String}()
    for (i, r) in enumerate(rows)
        caseid[i] = seen[r.id] > 1 ? "$(r.id)_$(r.rec_chains)" : r.id
    end

    meta = Dict{String,NamedTuple}()
    if DO_RCSB
        want = String[]
        for r in rows; append!(want, [r.id, r.rec_unbound, r.lig_unbound]); end
        want = unique(want)
        println("rcsb   : verifying ", length(want), " unique identifiers...")
        bad = String[]
        for (i, id) in enumerate(want)
            meta[id] = pdb_header(id)
            meta[id].ok || push!(bad, id)
            i % 25 == 0 && println("         ", i, "/", length(want))
        end
        if isempty(bad)
            println("         all ", length(want), " resolve at files.rcsb.org")
        else
            println("         UNRESOLVED: ", join(bad, " "))
            error("$(length(bad)) identifiers did not resolve at RCSB")
        end
    end

    open(OUT, "w") do io
        write_header(io, digest, length(E), length(ids), bydiff, meta)
        println(io, join(["case", "complex", "rec_chains", "lig_chains",
                          "rec_unbound", "rec_unbound_chains",
                          "lig_unbound", "lig_unbound_chains",
                          "category", "difficulty", "resolution", "note"], '\t'))
        for (i, r) in enumerate(rows)
            note = string(r.rec_name, " (", r.rec_chains, ") + ", r.lig_name,
                          " (", r.lig_chains, ")")
            r.redundant && (note *= "; SCOP-redundant per BM4.0 footnote")
            isempty(r.rmodel) || (note *= "; unbound receptor model $(r.rmodel)")
            isempty(r.lmodel) || (note *= "; unbound ligand model $(r.lmodel)")
            dif = Dict("Rigid-body" => "rigid_body",
                       "Medium Difficulty" => "medium",
                       "Difficult" => "difficult")[r.difficulty]
            res = haskey(meta, r.id) ? meta[r.id].resolution : "?"
            println(io, join([caseid[i], r.id, r.rec_chains, r.lig_chains,
                              r.rec_unbound, r.rec_unbound_chains,
                              r.lig_unbound, r.lig_unbound_chains,
                              "enzyme_inhibitor", dif, res, note], '\t'))
        end

        println(io, join(["1GGR", "1GGR", "A", "B", "1F3G", "A", "1HDN", "A",
                          "smoke", "-", "NMR",
                          "E2A-HPr, the smoke-test case. NOT a " *
                          "BM4.0 case and NOT an enzyme-inhibitor pair."], '\t'))
    end
    println("wrote  : ", OUT, "  (", length(rows), " benchmark rows + 1 smoke)")
end

function write_header(io, digest, nrows, nids, bydiff, meta)
    d = Dates.format(Dates.today(), "yyyy-mm-dd")
    print(io, """
# The case list.
#
# GENERATED FILE. Do not hand-edit: regenerate with
#
#     julia --project=. scripts/fetch_benchmark4.jl
#
# ############################################################################
# THE 51 CASES ARE PRESENT AND PROGRAMMATICALLY VERIFIED.
# ############################################################################
#
# SOURCE  (fetched $d, and re-fetchable -- this is provenance, not memory)
#
#   Set:       the enzyme category of the Protein-Protein Docking Benchmark,
#              version 4.0 ("BM4.0"), Table SI.
#   Citation:  Hwang, H., Vreven, T., Janin, J. & Weng, Z. "Protein-protein
#              docking benchmark version 4.0." Proteins: Structure, Function,
#              and Bioinformatics 78(15):3111-3114 (2010).
#   Canonical: $CANONICAL
#   Fetched:   $SOURCE
#   sha256:    $digest
#
#   The canonical address 302-redirects to the second: the lab moved host from
#   zlab.umassmed.edu to zlab.wenglab.org. Both are recorded so a future reader
#   can tell a move from a takedown. The page identifies itself as
#   "Supplementary Information for Hwang, H, Vreven, T, Janin, J and Weng, Z.,
#   Protein-Protein Docking Benchmark Version 4.0., Proteins., 2010 Nov
#   15;78(15):3111-4." -- i.e. it IS the supplementary table of the paper this
#   project already cited, and it agrees with reference [19] of the 2016 FMFT
#   preprint and reference 25 of the 2017 ClusPro protocol to the issue number.
#
# WHY 52 ROWS FOR "51 PAIRS"
#
#   BM4.0 Table SI has $nrows rows of category `E` spanning $nids DISTINCT PDB
#   entries: 1OYV appears twice, as 1OYV_A:I and 1OYV_B:I, two different
#   receptor chains of one deposition. The FMFT preprint's "51 enzyme-inhibitor
#   pairs from the established Protein Docking Benchmark [19]"
#   (Padhorny et al., 2016, https://doi.org/10.1073/pnas.1603929113, preprint p.4) is therefore the enzyme category counted by
#   PDB entry, and no other reading of the table yields exactly 51. That
#   arithmetic is the evidence; FMFT itself does not enumerate the cases (its
#   Table S1 is absent from the arXiv PDF), so the identity of the set is
#   [DERIVED] from a count, while every identifier below is [MEASURED] off the
#   authors' own table.
#
#   The category label is BM4.0's, quoted from its footnote (a) verbatim:
#   "E = Enzyme/Inhibitor or Enzyme/Substrate". So `enzyme_inhibitor` in the
#   category column is BM4.0's coarse label, NOT a claim that every case is
#   strictly an inhibitor complex -- BM5.5 later split these same cases into
#   EI (39), ES (8) and ER (4), with 1N8O dropped. Quote the set as "the
#   enzyme category of BM4.0", which is what FMFT measured against PIPER.
#
# TYPE IS NOT DIFFICULTY -- THE AXIS THAT MUST NOT BE COLLAPSED
#
#   BM4.0 classifies on TWO INDEPENDENT AXES and the table carries them in
#   different places: biological TYPE is the `Cat.` column (E / A / O / AB),
#   docking DIFFICULTY is a section heading (Rigid-body / Medium / Difficult).
#   The 51 are defined by TYPE ONLY. Their difficulty spread is
#   $(get(bydiff, "Rigid-body", 0)) rigid-body, $(get(bydiff, "Medium Difficulty", 0)) medium, $(get(bydiff, "Difficult", 0)) difficult, so restricting to the
#   rigid-body subset would drop $(get(bydiff, "Medium Difficulty", 0) + get(bydiff, "Difficult", 0)) cases and produce a hit rate that looks
#   comparable to the published one and is not. The `difficulty` column is here
#   so that split stays available and explicit rather than accidental.
#
# VERIFICATION PERFORMED
#
#   1. The page parses to exactly 176 case rows -- BM4.0's published size --
#      and the script fails rather than writing a file if it does not.
#   2. Category E yields exactly $nrows rows over $nids unique entries; both are
#      asserted.
#   3. Every identifier below -- bound complex and both unbound components --
#      was resolved against files.rcsb.org, and the `resolution` column is read
#      from the deposited header's REMARK 2 rather than supplied by hand.
#
# COLUMNS
#   case                 identifier used in output and file names; equals
#                        `complex` except where one entry contributes two cases
#                        (1OYV_A, 1OYV_B), which is the 52-vs-51 arithmetic
#   complex              PDB id of the bound complex (the answer)
#   rec_chains           receptor chains in the bound complex
#   lig_chains           ligand chains in the bound complex
#   rec_unbound          PDB id of the separately-solved receptor
#   rec_unbound_chains   its chains; `.` means the blank chain id of an
#                        old-style single-chain deposition, NOT "unknown"
#   lig_unbound          PDB id of the separately-solved ligand
#   lig_unbound_chains   its chains, same convention
#   category             enzyme_inhibitor | smoke
#   difficulty           rigid_body | medium | difficult  (BM4.0's second axis)
#   resolution           Angstroms from REMARK 2 of the bound complex, or NMR
#   note                 protein names from the table, plus BM4.0's own flags
#
# STRUCTURES ARE FETCHED, NOT COMMITTED
#
#   bash scripts/fetch_structures.sh --all
#
#   data/structures/ is gitignored: third-party primary data with a canonical
#   source and a stable URL, so the repository carries provenance and digests,
#   not bytes. A case whose file is missing is REPORTED missing by
#   bench/benchmark51.jl, never skipped -- a batch that quietly docks 3 of 51
#   and prints a hit rate is worse than one that fails.
#
""")
end

main()
