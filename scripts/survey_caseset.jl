#!/usr/bin/env julia
"""
    julia --project=test scripts/survey_caseset.jl \\
          --json results/caseset-benchmark51.json

# Why this is a script and not a paragraph

This measures it.

It is also the cheapest possible answer to "what should we rent?", and it costs
nothing to obtain: the whole thing runs on a laptop in under a minute, where the
alternative is discovering the sizing on a metered instance.

# What is measured, and what is only derived

**Measured** here, in the sense that it is computed from real coordinates by the
same code the docking path uses: atom counts, the `gridspec_for` grid, the cell
count, and the byte model of `src/vram.jl` evaluated on those grids. Nothing is
recalled and nothing is typed in by hand.

Both directions of error are known and neither is measured -- linear-in-cells
**over**-estimates, because an FFT is `N log N` rather than `N`; excluding per-case
setup (ingestion, ACP typing, three receptor fields, `FFTW.MEASURE` planning and
1000-pose clustering, once per case) **under**-estimates. It is a planning input. It is
not a benchmark result and must never be quoted as one.

# Environment

Needs `BioStructures` -- the PDB reader is a package extension -- so `--project=test` or
`--project=gpu`, never `--project=bench`. Structures are fetched, not committed:

    bash scripts/fetch_structures.sh --all

Options:
    --cases FILE       case list          (data/benchmark51.tsv)
    --structures DIR   fetched PDBs       (data/structures)
    --json FILE        write the document
"""

using JuPIPER, Printf
const JP = JuPIPER

include(joinpath(@__DIR__, "..", "bench", "caselist.jl"))
using .CaseList

const HAVE_BIO = Base.identify_package("BioStructures") !== nothing
HAVE_BIO && @eval using BioStructures

function argval(flag::AbstractString, default)
    i = findfirst(==(flag), ARGS)
    i === nothing && return default
    i == length(ARGS) && error("$flag needs a value")
    return ARGS[i+1]
end

const CASEFILE  = argval("--cases", CaseList.default_casefile())
const STRUCTDIR = argval("--structures", CaseList.default_structdir())
const JSONOUT   = argval("--json", nothing)


const REF_CELLS = 66 * 66 * 78
const REF_NROT  = 69_960
const REF_F32_S = 20.722
const REF_F64_S = 60.504


_j(::Nothing) = "null"
_j(x::Bool) = x ? "true" : "false"
_j(x::Integer) = string(x)
_j(x::AbstractFloat) = isfinite(x) ? string(x) : "null"
_j(x::AbstractString) = '"' * replace(String(x), '\\' => "\\\\", '"' => "\\\"",
                                      '\n' => "\\n", '\r' => "\\r", '\t' => "\\t") * '"'
_j(x::Symbol) = _j(String(x))
_j(x::Union{AbstractVector,Tuple}) = "[" * join((_j(v) for v in x), ",") * "]"
_j(x::NamedTuple) = "{" * join((_j(String(k)) * ":" * _j(v) for (k, v) in pairs(x)), ",") * "}"
_j(x) = _j(string(x))

median_of(v) = (s = sort(collect(float.(v))); n = length(s);
                isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2)

"""
    survey_case(case, dir, P) -> NamedTuple

Ingest one case's bound components and report its grid. No docking, no timing.
"""
function survey_case(case::Case, dir::AbstractString, P::Integer)
    path = normpath(joinpath(dir, case.complex * ".pdb"))
    isfile(path) || error("structure not found: $(scrub(path))  --  " *
                          "fetch it: bash scripts/fetch_structures.sh --all")
    rec = read_atoms(path; T = Float64, chains = chain_ids(case.rec_chains))
    lig = read_atoms(path; T = Float64, chains = chain_ids(case.lig_chains))
    length(rec) > 0 ||
        error("receptor chain(s) $(case.rec_chains) yielded no atoms; " *
              "$(case.complex) has [$(join(rec.report.chains_present, ","))]")
    length(lig) > 0 ||
        error("ligand chain(s) $(case.lig_chains) yielded no atoms; " *
              "$(case.complex) has [$(join(lig.report.chains_present, ","))]")

    spec = gridspec_for(rec, lig; spacing = JP.PIPER_SPACING, pad = 2.0)
    n = Tuple(Int.(spec.n))
    cells = prod(n)

    dev = NamedTuple()
    for (tag, T) in (("f32", Float32), ("f64", Float64))
        sp = GridSpec{T}(n, T(JP.PIPER_SPACING), zero(JP.SVector{3,T}))
        m = JP.memory_model(sp, 2 + P; natoms = length(lig), nrot = 70_000, T = T)
        for b in (1, 2, 4)
            dev = merge(dev, (Symbol("dev_$(tag)_b$(b)_bytes") =>
                              JP.job_bytes(m, b; fft_factor = JP.FFT_FACTOR_MEASURED),))
        end
    end

    return merge((case = case.id, complex = case.complex,
                  category = case.category, difficulty = case.difficulty,
                  rec_chains = case.rec_chains, lig_chains = case.lig_chains,
                  rec_atoms = length(rec), lig_atoms = length(lig),
                  grid_n = collect(n), cells = cells,
                  cells_vs_1ggr = cells / REF_CELLS,
                  max_dim = maximum(n),
                  est_dock_f32_s = REF_F32_S * cells / REF_CELLS,
                  est_dock_f64_s = REF_F64_S * cells / REF_CELLS), dev)
end

function main()
    if !HAVE_BIO
        println("SKIP -- BioStructures is not in the active environment.")
        println("       Run under --project=test or --project=gpu.")
        return 0
    end

    cases = read_cases(CASEFILE)
    complete = caselist_complete(cases)
    P = dars_rank(load_dars())

    rows, failures = Any[], Any[]
    for c in cases
        try
            push!(rows, survey_case(c, STRUCTDIR, P))
        catch e
            msg = scrub(first(split(sprint(showerror, e), '\n')))
            push!(failures, (case = c.id, complex = c.complex, error = msg))
            @printf("FAIL %-8s %s\n", c.id, msg)
        end
    end
    isempty(rows) && (println("nothing surveyed"); return 1)

    enz = [r for r in rows if r.category == "enzyme_inhibitor"]
    sort!(enz; by = r -> r.cells)
    cells = [r.cells for r in enz]

    println("\nCase set -- grid and memory survey (no docking, no timing)")
    println(repeat("-", 74))
    @printf("%-8s %6s %6s %-15s %11s %8s\n", "case", "rec", "lig", "grid", "cells", "vs 1GGR")
    for r in enz
        @printf("%-8s %6d %6d %-15s %11d %7.2fx\n", r.case, r.rec_atoms, r.lig_atoms,
                join(r.grid_n, "x"), r.cells, r.cells_vs_1ggr)
    end
    println(repeat("-", 74))
    @printf("enzyme cases    %d  (%d distinct entries)\n", length(enz),
            length(unique(r.complex for r in enz)))
    @printf("cells           min %d / median %.0f / max %d\n",
            minimum(cells), median_of(cells), maximum(cells))
    @printf("vs 1GGR         median %.2fx / max %.2fx\n",
            median_of(cells) / REF_CELLS, maximum(cells) / REF_CELLS)
    @printf("largest single grid dimension  %d\n", maximum(r.max_dim for r in enz))

    worst_f32 = maximum(r.dev_f32_b4_bytes for r in enz)
    worst_f64 = maximum(r.dev_f64_b4_bytes for r in enz)
    @printf("\nworst-case device memory, batch 4:  %.3f GiB (f32)  %.3f GiB (f64)\n",
            worst_f32 / 2^30, worst_f64 / 2^30)
    println("  -- so VRAM is not the binding constraint on this set at any of the")
    println("     batch sizes the measured optimum brackets. L2 selects the card.")

    t32 = sum(r.est_dock_f32_s for r in enz)
    t64 = sum(r.est_dock_f64_s for r in enz)
    @printf("\ncampaign dock! time [ESTIMATE], 70k rotations on an RTX 4090:\n")
    @printf("  Float32 %6.1f s = %.2f h      Float64 %6.1f s = %.2f h\n",
            t32, t32 / 3600, t64, t64 / 3600)
    println("  Scaled linearly in cells from a reference run of 1GGR on that card. An FFT is")
    println("  N log N, so this over-estimates; it excludes per-case setup, so it")
    println("  under-estimates. NOT a benchmark result.")

    println(complete ?
            "\nVERDICT: full BM4.0 enzyme set (51 entries / 52 cases)." :
            "\nVERDICT: SUBSET only -- not the full enzyme set.")
    isempty(failures) || @printf("\n%d case(s) FAILED to survey -- see the JSON\n",
                                 length(failures))

    if JSONOUT !== nothing
        doc = (kind = "caseset-survey",
               note = "Grid and memory characterisation of the case set. " *
                      "No docking and no timing, so this document is " *
                      "machine-independent and carries no noise floor; the " *
                      "wall-clock fields are [ESTIMATE] scaled from a reference run of 1GGR.",
               caselist = scrub(CASEFILE),
               caselist_complete = complete,
               date = Base.Libc.strftime("%Y-%m-%d", time()),
               julia = string(VERSION),
               spacing = JP.PIPER_SPACING, pad = 2.0,
               P = P, nterms = 2 + P,
               fft_factor = JP.FFT_FACTOR_MEASURED,
               reference = (case = "1GGR", cells = REF_CELLS,
                            nrot = REF_NROT, f32_s = REF_F32_S, f64_s = REF_F64_S,
                            device = "NVIDIA GeForce RTX 4090"),
               n_enzyme_cases = length(enz),
               n_distinct_entries = length(unique(r.complex for r in enz)),
               cells_min = minimum(cells), cells_median = median_of(cells),
               cells_max = maximum(cells),
               max_grid_dim = maximum(r.max_dim for r in enz),
               worst_device_bytes_f32_b4 = worst_f32,
               worst_device_bytes_f64_b4 = worst_f64,
               est_campaign_f32_s = t32, est_campaign_f64_s = t64,
               est_status = "ESTIMATE -- linear in cells, excludes per-case setup",
               failures = failures,
               rows = enz)
        open(JSONOUT, "w") do io
            write(io, _j(doc))
        end
        println("\nwrote ", scrub(JSONOUT))
    end
    return isempty(failures) ? 0 : 1
end

exit(main())
