"""
`bench/benchmark51.jl` docks the set and `scripts/survey_caseset.jl` characterises it
without docking. So they live here once, in the pattern `bench/gpuarm.jl` already
establishes.
"""
module CaseList

using SHA: sha256

export Case, read_cases, caselist_complete, chain_ids, scrub


const _ROOT = normpath(joinpath(@__DIR__, ".."))

"""
    scrub(s) -> String

Strip this checkout's absolute path out of a string, leaving a repo-relative one,
and normalise separators.

Not cosmetic.

It rewrites *this* checkout's root only. A path deliberately pointed outside the
repository -- `--structures` at a scratch directory, say -- is left alone, since
there is no repo-relative form of it to produce; such a run is a rehearsal, and
its output is not what gets committed.
"""
function scrub(s::AbstractString)
    t = replace(String(s), '\\' => '/')
    r = replace(normpath(_ROOT), '\\' => '/')
    endswith(r, '/') || (r *= "/")
    return replace(t, r => "")
end

"""Default location of the generated case list."""
default_casefile() = joinpath(_ROOT, "data", "benchmark51.tsv")

"""Default location of the fetched structures (deliberately not committed)."""
default_structdir() = joinpath(_ROOT, "data", "structures")


struct Case
    id::String
    complex::String
    rec_chains::String
    lig_chains::String
    rec_unbound::String
    rec_unbound_chains::String
    lig_unbound::String
    lig_unbound_chains::String
    category::String
    difficulty::String
    resolution::String
    note::String
end

const NCOLS = fieldcount(Case)

"""
    read_cases(path) -> Vector{Case}

Parse the provenance-headed case list. Comments and the column-name row are
skipped; `-` means absent and `.` means the blank chain identifier of an
old-style single-chain deposition, which is a real chain and not a gap.
"""
function read_cases(path::AbstractString)
    out = Case[]
    lineno = 0
    open(path, "r") do io
        for line in eachline(io)
            lineno += 1
            s = strip(line)
            (isempty(s) || startswith(s, "#")) && continue
            f = split(s, '\t')
            length(f) == NCOLS ||
                error("$path:$lineno: expected $NCOLS tab-separated fields, " *
                      "got $(length(f)). The list is generated -- regenerate " *
                      "it with scripts/fetch_benchmark4.jl rather than editing.")
            strip(f[1]) == "case" && continue
            push!(out, Case(String.(strip.(f))...))
        end
    end
    return out
end

"""
    caselist_complete(cases) -> Bool

Check the 52 bound enzyme/inhibitor cases from BM4.0 Table SI, including both
receptor chains of 1OYV. The reference digest is over the sorted, tab-separated
(case ID, complex, receptor chains, ligand chains) rows joined by LF, without a
final newline. Sorting permits reordered input and preserves duplicate rows.
Other categories, including the 1GGR control, are outside this check.
"""
function caselist_complete(cases)
    keys = sort!([join((c.id, c.complex, c.rec_chains, c.lig_chains), '\t')
                  for c in cases if c.category == "enzyme_inhibitor"])
    return bytes2hex(sha256(join(keys, '\n'))) ==
           "d82ac902b770f63b30be1ee6e4ef4c87975ad6a727ec615b02f554ff3589a6b8"
end

"""
    chain_ids(spec) -> Vector{String}

Split a case-list chain specification into the individual ids `read_atoms` wants.

The column is BM4.0's own notation, in which `1HIA_AB:I` means receptor chains A
**and** B -- so `"AB"` is a *set of two* one-character PDB chain ids, not one id
named `"AB"`. `read_atoms` cannot make that assumption on our behalf and should
not: it takes a string as a **single** id because an mmCIF chain id is
legitimately multi-character, and collapsing the two conventions inside the
reader would break the very format the ingestion seam exists to accept. So the
split belongs here, at the layer that knows the notation.

Getting this wrong is silent in the worst way. `chains = "AB"` matches no chain, the
table comes back **empty** rather than short, and every downstream step is still
perfectly well-defined on zero atoms -- so the failure surfaces as a grid-fit or typing
error deep inside a run, attributed to the wrong subsystem. Six of the 52 cases are
multi-chain on one side or the other, which is 11.5% of the campaign.

`.` is the blank chain id of an old-style single-chain deposition -- the case
list's own convention, and a real chain rather than a missing value.
"""
function chain_ids(spec::AbstractString)
    s = strip(spec)
    (isempty(s) || s == "-") &&
        error("empty chain specification: the case list writes `-` for absent " *
              "and `.` for the blank chain id, and neither can be docked")
    return [c == '.' ? " " : string(c) for c in s]
end

end
