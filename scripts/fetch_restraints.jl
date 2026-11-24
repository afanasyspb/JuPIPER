#!/usr/bin/env julia


using Downloads, SHA, Dates

const AIR_URL  = "https://cluspro.org/examples/ja026939x_s2.txt"
const JSON_URL = "https://cluspro.org/examples/ja026939x_s2.json"

argval(f, d) = (i = findfirst(==(f), ARGS); i === nothing ? d : ARGS[i + 1])
const OUT = argval("--out",
                   joinpath(@__DIR__, "..", "data", "restraints", "1f3g_1poh_20noe.json"))


function fetch_text(url)
    io = IOBuffer()
    Downloads.download(url, io)
    b = take!(io)

    return (bytes2hex(sha256(b)), String(b))
end


struct Group
    anchor_seg::String
    anchor_res::Int
    partner_seg::String
    partner_res::Vector{Int}
    d::Float64
    dminus::Float64
    dplus::Float64
end

function parse_air(txt)
    groups = Group[]
    for m in eachmatch(r"assign\s*\(\s*resid\s+(\d+)\s+and\s+segid\s+(\w+)\s*\)\s*\((.*?)\)\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)"s, txt)
        anchor_res = parse(Int, m.captures[1])
        anchor_seg = m.captures[2]
        body       = m.captures[3]
        partners   = [(parse(Int, p.captures[1]), p.captures[2])
                      for p in eachmatch(r"resid\s+(\d+)\s+and\s+segid\s+(\w+)", body)]
        isempty(partners) && error("AIR statement for $anchor_seg$anchor_res has no partners")
        segs = unique(last.(partners))
        length(segs) == 1 ||
            error("AIR statement for $anchor_seg$anchor_res spans segids $segs -- " *
                  "this parser assumes one partner chain per restraint")
        push!(groups, Group(anchor_seg, anchor_res, segs[1], sort(first.(partners)),
                            parse(Float64, m.captures[4]), parse(Float64, m.captures[5]),
                            parse(Float64, m.captures[6])))
    end
    return groups
end


function parse_cluspro_json(txt)

    reqs   = [parse(Int, m.captures[1]) for m in eachmatch(r"\"required\"\s*:\s*(\d+)", txt)]
    chunks = split(txt, r"\"required\"\s*:\s*\d+")

    length(chunks) == length(reqs) + 1 ||
        error("JSON group/`required` structure not as expected")
    groups = Vector{NamedTuple}(undef, length(reqs))
    for i in eachindex(reqs)
        c = chunks[i]
        recs = [parse(Int, m.captures[1]) for m in eachmatch(r"\"rec_resid\"\s*:\s*\"(\d+)\"", c)]
        ligs = [parse(Int, m.captures[1]) for m in eachmatch(r"\"lig_resid\"\s*:\s*\"(\d+)\"", c)]
        dmax = unique([parse(Float64, m.captures[1]) for m in eachmatch(r"\"dmax\"\s*:\s*([\d.]+)", c)])
        dmin = unique([parse(Float64, m.captures[1]) for m in eachmatch(r"\"dmin\"\s*:\s*([\d.]+)", c)])
        typs = unique([m.captures[1] for m in eachmatch(r"\"type\"\s*:\s*\"(\w+)\"", c)])
        rchs = unique([m.captures[1] for m in eachmatch(r"\"rec_chain\"\s*:\s*\"(\w+)\"", c)])
        lchs = unique([m.captures[1] for m in eachmatch(r"\"lig_chain\"\s*:\s*\"(\w+)\"", c)])
        groups[i] = (rec = recs, lig = ligs, dmax = dmax, dmin = dmin,
                     type = typs, rec_chain = rchs, lig_chain = lchs, required = reqs[i])
    end
    return groups
end


hr(t) = (println(); println(t); println(repeat("-", length(t))))

hr("E2A/HPr restraint set")
println("AIR   ", AIR_URL)
println("JSON  ", JSON_URL)

air_sha,  air_txt  = fetch_text(AIR_URL)
json_sha, json_txt = fetch_text(JSON_URL)
println("\nAIR   ", length(air_txt),  " bytes  sha256 ", air_sha)
println("JSON  ", length(json_txt), " bytes  sha256 ", json_sha)

air = parse_air(air_txt)
jsn = parse_cluspro_json(json_txt)


length(air) == 20 || error("expected 20 AIR statements, got $(length(air)) -- " *
                           "the published file has changed; do not write a set " *
                           "whose count no longer matches the protocol's own " *
                           "'20 intermolecular distance restraints'")
length(jsn) == 20 || error("expected 20 JSON groups, got $(length(jsn))")


nA = count(g -> g.anchor_seg == "A", air)
nB = count(g -> g.anchor_seg == "B", air)
(nA, nB) == (11, 9) ||
    error("expected 11 E2A-anchored and 9 HPr-anchored AIRs, got ($nA, $nB)")


dset = unique([(g.d, g.dminus, g.dplus) for g in air])
length(dset) == 1 || error("AIR file mixes distance conventions: $dset")
air_lo, air_hi = dset[1][1] - dset[1][2], dset[1][1] + dset[1][3]

jdmax = unique(reduce(vcat, [g.dmax for g in jsn]))
jdmin = unique(reduce(vcat, [g.dmin for g in jsn]))
jtype = unique(reduce(vcat, [g.type for g in jsn]))
length(jdmax) == 1 && length(jdmin) == 1 ||
    error("ClusPro JSON mixes distance bounds: dmin $jdmin dmax $jdmax")
jtype == ["residue"] || error("ClusPro JSON restraint type is $jtype, not 'residue' -- " *
                              "the atom-level form has different semantics")


for (k, g) in enumerate(air)
    j = jsn[k]
    if g.anchor_seg == "A"
        want_rec = [g.anchor_res]; want_lig = g.partner_res
    else
        want_rec = g.partner_res;  want_lig = [g.anchor_res]
    end
    sort(unique(j.rec)) == sort(unique(want_rec)) ||
        error("group $k: JSON receptor residues $(sort(unique(j.rec))) != AIR $(sort(unique(want_rec)))")
    sort(unique(j.lig)) == sort(unique(want_lig)) ||
        error("group $k: JSON ligand residues $(sort(unique(j.lig))) != AIR $(sort(unique(want_lig)))")
    j.required == 1 ||
        error("group $k has required=$(j.required); this transcription assumes the " *
              "AIR semantics (any one member suffices)")
end


e2a = sort(unique(vcat([g.anchor_seg == "A" ? [g.anchor_res] : g.partner_res for g in air]...)))
hpr = sort(unique(vcat([g.anchor_seg == "B" ? [g.anchor_res] : g.partner_res for g in air]...)))


outgroups = map(air) do g
    if g.anchor_seg == "A"
        (rec = [g.anchor_res], lig = g.partner_res)
    else
        (rec = g.partner_res, lig = [g.anchor_res])
    end
end
npair = sum(length(g.rec) * length(g.lig) for g in outgroups)

println("\ngroups                 ", length(air), "  (", nA, " anchored on E2A, ", nB, " on HPr)")
println("residue pairs          ", npair)
println("E2A residues involved  ", length(e2a), "  ", e2a)
println("HPr residues involved  ", length(hpr), "  ", hpr)
println("AIR distance range     [", air_lo, ", ", air_hi, "] A  (HADDOCK effective distance)")
println("ClusPro residue bound  [", jdmin[1], ", ", jdmax[1], "] A  (min heavy-atom distance) <- adopted")
println("per-group cutoff       required = 1  ->  RestraintGroup cutoff 0")
println("set-level cutoff       UNPUBLISHED -- written as null")


jstr(s) = string('"', replace(String(s), '\\' => "\\\\", '"' => "\\\""), '"')
jarr(v) = string('[', join(v, ", "), ']')

mkpath(dirname(OUT))
open(OUT, "w") do io
    println(io, "{")
    println(io, "  \"_provenance\": {")
    println(io, "    \"what\": \"The 20 ambiguous interaction restraints (AIRs) that the 2017 ",
                "ClusPro protocol uses for its own worked E2A-HPr example. NOT the 20 NOE ",
                "distances themselves: the NOE measurements are Wang et al., EMBO J ",
                "19:5635-5649 (2000), which this repository does not hold, and the AIRs ",
                "derived from them are Dominguez, Boelens & Bonvin, JACS 125:1731-1737 (2003) ",
                "[ja026939x] supplementary file S2.\",")
    println(io, "    \"cited_by\": \"Kozakov, Hall, Xia, Porter, Padhorny, Yueh, Beglov & Vajda, ",
                "'The ClusPro web server for protein-protein docking', Nature Protocols ",
                "12(2):255-278 (2017), published p.273 = PMC author manuscript p.21, ",
                "'Accounting for pairwise distance restraint'.\",")
    println(io, "    \"source_air\": ", jstr(AIR_URL), ",")
    println(io, "    \"source_air_sha256\": ", jstr(air_sha), ",")
    println(io, "    \"source_json\": ", jstr(JSON_URL), ",")
    println(io, "    \"source_json_sha256\": ", jstr(json_sha), ",")
    println(io, "    \"retrieved\": ", jstr(string(today())), ",")
    println(io, "    \"regenerated_by\": \"scripts/fetch_restraints.jl\",")
    println(io, "    \"checked\": \"Both files parsed independently and required to agree: ",
                "20 groups; 11 anchored on E2A and 9 on HPr; every receptor and ligand ",
                "residue list identical group for group; one distance convention across the ",
                "set; every ClusPro group carrying required=1; ClusPro restraint type ",
                "'residue'. The script refuses to write on any mismatch.\",")
    println(io, "    \"case\": {")
    println(io, "      \"receptor\": {\"pdb\": \"1F3G\", \"chain\": \"A\", \"molecule\": ",
                "\"glucose-specific phosphocarrier protein IIAglc (E2A), unbound X-ray\"},")
    println(io, "      \"ligand\": {\"pdb\": \"1POH\", \"chain\": \"A\", \"molecule\": ",
                "\"histidine-containing phosphocarrier protein HPr, unbound X-ray\"},")
    println(io, "      \"native\": {\"pdb\": \"1GGR\", \"rec_chain\": \"A\", \"lig_chain\": \"B\"}")
    println(io, "    },")
    println(io, "    \"distance_convention\": {")
    println(io, "      \"adopted\": {\"dmin\": ", jdmin[1], ", \"dmax\": ", jdmax[1],
                ", \"meaning\": \"minimum heavy-atom separation between the two residues\", ",
                "\"source\": \"ClusPro JSON, type 'residue'\"},")
    println(io, "      \"air_original\": {\"dmin\": ", air_lo, ", \"dmax\": ", air_hi,
                ", \"meaning\": \"HADDOCK effective distance, <d^-6>^(-1/6) over all atom pairs\", ",
                "\"source\": \"CNS assign statement '", dset[1][1], " ", dset[1][2], " ", dset[1][3], "'\"},")
    println(io, "      \"note\": \"The two are different quantities, not two roundings of one. ",
                "JuPIPER's Restraint is a minimum-separation test, so the ClusPro reading is ",
                "the one it can express; the AIR numbers are recorded so the substitution is ",
                "visible.\"")
    println(io, "    },")
    println(io, "    \"group_cutoff\": {\"value\": 0, \"meaning\": \"RestraintGroup cutoff, ",
                "i.e. MORE THAN 0 members satisfied = at least one. This is the AIR semantics ",
                "and matches every published group's required=1.\"},")
    println(io, "    \"set_cutoff\": {\"value\": null, \"status\": \"[UNPUBLISHED]",
                "\"meaning\": \"How many of the 20 groups a translation must satisfy. The ",
                "protocol (p.8) says 'the required number of restraint groups', and the ",
                "published JSON carries no set-level field. Swept by ",
                "scripts/validate_constraints.jl rather than guessed.\"},")
    println(io, "    \"status\": \"[MEASURED] as a transcription of two fetched, mutually ",
                "cross-checked files. The distance reading is [ASSUMED] (see ",
                "distance_convention) and the set-level cutoff is [UNPUBLISHED].\"")
    println(io, "  },")
    println(io, "  \"receptor_residues\": ", jarr(e2a), ",")
    println(io, "  \"ligand_residues\": ", jarr(hpr), ",")
    println(io, "  \"groups\": [")
    for (k, g) in enumerate(outgroups)
        print(io, "    {\"rec_residues\": ", jarr(g.rec),
              ", \"lig_residues\": ", jarr(g.lig),
              ", \"dmin\": ", jdmin[1], ", \"dmax\": ", jdmax[1], ", \"cutoff\": 0}")
        println(io, k == length(outgroups) ? "" : ",")
    end
    println(io, "  ]")
    println(io, "}")
end

println("\nwrote ", OUT)
println("sha256 ", bytes2hex(sha256(read(OUT))))
