#!/usr/bin/env julia
"""
    julia --project=gpu -t auto bench/benchmark51.jl --T float32 --json out.json
    julia --project=test -t auto bench/benchmark51.jl --limit 1 --nrot 2000

# What this is

`scripts/validate_bound.jl` docks **one** complex and is the authority on the
physics; `bench/realdock.jl` docks one complex on **two arms** and is the
authority on the device. Neither produces a *distribution*, and a distribution
is what turns "the pipeline reassembles 1GGR" into "the pipeline docks
proteins". This harness is the third thing: the same pipeline over a case list,
with per-case isolation, and an output shaped so the reference's own published
per-case results are the comparator.

The metric is the one both PIPER and FMFT report, so it is comparable without our
inventing one: **hits among the top 1000 poses by energy**, a hit being a pose within
`--hit` angstrom C-alpha ligand RMSD of the native.

# The case list, and the verdict printed beside every number

`data/benchmark51.tsv` is **generated**, not hand-written: `scripts/fetch_benchmark4.jl`
downloads Table SI of the Protein-Protein Docking Benchmark 4.0 (Hwang, Vreven, Janin &
Weng, Proteins 78(15):3111-3114, 2010; https://zlab.umassmed.edu/benchmark/) and writes
the identifiers from it.

Two properties of the set are easy to get wrong and are carried in the file
explicitly. It is the enzyme *type* category -- BM4.0's own footnote reads
"E = Enzyme/Inhibitor or Enzyme/Substrate" -- and **not** the rigid-body
*difficulty* class, which is an orthogonal axis; the 52 rows are 41 rigid-body,
6 medium and 5 difficult, so intersecting the axes would silently drop 11 cases.
And the 52 rows span **51 distinct PDB entries**, 1OYV contributing two through
two receptor chains, which is the arithmetic that identifies the set as the "51
enzyme-inhibitor pairs" the FMFT preprint measured PIPER against.

The JSON reports the completeness of the selected enzyme/inhibitor cohort,
whether the cohort has every input and at least one completed search
(`is_benchmark`), and whether every requested row succeeded (`run_complete`).
Recovery and timing summaries are computed separately for each category.

# Error isolation

One bad structure must not cost a batch. Each case runs inside its own `try`,
and a failure is recorded as a row with `status` and the exception text rather
than propagating: the run continues, the JSON is still written, and the exit
code is nonzero only if *every* case failed. Missing input files are reported
as `status = "missing"` rather than skipped, because a batch that quietly docks
3 of 51 and prints a hit rate is worse than one that stops.

Every row carries the *same columns* whatever its status -- see `ROW_TEMPLATE`
-- so a consumer filters on `status` rather than on key presence, and an error
row still names the case, its category and its difficulty.

# Environment

It skips with an explanation rather than failing. Under `--project=gpu` it reaches the
device purely by array type, exactly as the other tiers do.

Structures are fetched, not committed:

    bash scripts/fetch_structures.sh

Options:
    --cases FILE     case list           (data/benchmark51.tsv)
    --limit N        first N cases only  (0 = all)
    --only ID[,ID]   run just these case ids
    --nrot N         rotations           (70000)
    --profile KEY    ClusPro mode        (balanced)  -- see the [ASSUMED] note below
    --w1 X           override the shape repulsion weight; default 7.0, see below
    --charges zero|formal    partial-charge model    (default: zero)
    --gb capped|truncated    electrostatic cutoff    (capped)
    --T float32|float64                    (float64 -- approx. 3x the device
                     time of float32, measured; the run says so if you leave it)
    --hit A          hit threshold, C-alpha ligand RMSD   (10.0, PDF p.22)
    --topk N         hit-counting pool                    (1000, 2017 p.5)
    --cluster-radius A                                    (10.0)
    --batch N        rotations per correlate! call        (1)
    --gpu            require a device; fail rather than silently running on host
    --json FILE      write the batch document
"""

using JuPIPER, FFTW, StaticArrays, LinearAlgebra, Printf
const JP = JuPIPER

include(joinpath(@__DIR__, "gpuarm.jl"))
using .GPUArm


include(joinpath(@__DIR__, "caselist.jl"))
using .CaseList


const HAVE_BIO = Base.identify_package("BioStructures") !== nothing
HAVE_BIO && @eval using BioStructures


function argval(flag::AbstractString, default)
    i = findfirst(==(flag), ARGS)
    i === nothing && return default
    i == length(ARGS) && error("$flag needs a value")
    return ARGS[i+1]
end
argflag(flag::AbstractString) = findfirst(==(flag), ARGS) !== nothing
argint(flag, d) = (v = argval(flag, nothing); v === nothing ? d : parse(Int, v))
argnum(flag, d) = (v = argval(flag, nothing); v === nothing ? d : parse(Float64, v))

const CASEFILE = argval("--cases", joinpath(@__DIR__, "..", "data", "benchmark51.tsv"))
const STRUCTDIR = argval("--structures", joinpath(@__DIR__, "..", "data", "structures"))
const LIMIT    = argint("--limit", 0)
const ONLY     = (v = argval("--only", nothing);
                  v === nothing ? String[] : String.(split(v, ',')))
const NROT     = argint("--nrot", 70_000)
const PROFILE  = Symbol(argval("--profile", "balanced"))
const HIT      = argnum("--hit", 10.0)
const TOPK     = argint("--topk", 1000)
const CLUSTRAD = argnum("--cluster-radius", 10.0)
const MAXCLUST = argint("--max-clusters", 30)
const BATCH    = argint("--batch", 1)
const REQGPU   = argflag("--gpu")
const CHARGES  = argval("--charges", "zero")
const GBMODE   = argval("--gb", "capped")
const ELT      = lowercase(argval("--T", "float64"))


const T_EXPLICIT = argflag("--T")
const JSONOUT  = argval("--json", nothing)
const TAG      = argval("--tag", "")

const T = ELT == "float32" ? Float32 :
          ELT == "float64" ? Float64 :
          error("--T must be float32 or float64, got $ELT")


const W1 = argnum("--w1", 7.0)

charge_model(s) = s == "zero"   ? ZeroCharges()   :
                  s == "formal" ? FormalCharges() :
                  error("--charges must be zero or formal, got $s")
gb_model(s)     = s == "capped"    ? GBCapped()    :
                  s == "truncated" ? GBTruncated() :
                  error("--gb must be capped or truncated, got $s")

hr(t) = (println(); println(t); println(repeat("-", length(t))))


_j(::Nothing) = "null"
_j(::Missing) = "null"
_j(x::Bool) = x ? "true" : "false"
_j(x::Integer) = string(x)
_j(x::AbstractFloat) = isfinite(x) ? string(x) : "null"
_j(x::AbstractString) = '"' * replace(String(x), '\\' => "\\\\", '"' => "\\\"",
                                      '\n' => "\\n", '\r' => "\\r", '\t' => "\\t") * '"'
_j(x::Symbol) = _j(String(x))
_j(x::Union{AbstractVector,Tuple}) = "[" * join((_j(v) for v in x), ",") * "]"
_j(x::NamedTuple) = "{" * join((_j(String(k)) * ":" * _j(v) for (k, v) in pairs(x)), ",") * "}"
_j(x::AbstractDict) = "{" * join((_j(string(k)) * ":" * _j(v) for (k, v) in x), ",") * "}"


_j(x) = _j(string(x))


"""
    dock_case(case, ctx) -> NamedTuple

Ingest, parametrise, grid, dock, cluster and score one complex. Returns a row.

Throws on anything that goes wrong; the caller is what isolates a failure. That
split is deliberate: a `try` around the *whole* case keeps the failure text and
the case identity together, while a `try` inside each step would produce a row
that is half-populated and looks like a result.
"""
function dock_case(case::Case, ctx)
    path = normpath(joinpath(STRUCTDIR, case.complex * ".pdb"))
    isfile(path) || error("structure not found: $(scrub(path))  --  " *
                          "fetch it first: bash scripts/fetch_structures.sh $(case.complex)")

    rec = read_atoms(path; T = T, chains = chain_ids(case.rec_chains))
    lig = read_atoms(path; T = T, chains = chain_ids(case.lig_chains))

    length(rec) > 0 ||
        error("receptor chain(s) $(case.rec_chains) yielded no atoms; " *
              "$(case.complex) has [$(join(rec.report.chains_present, ","))]")
    length(lig) > 0 ||
        error("ligand chain(s) $(case.lig_chains) yielded no atoms; " *
              "$(case.complex) has [$(join(lig.report.chains_present, ","))]")
    swapped = length(lig) > length(rec)

    par_rec = parametrize(ctx.tbl, rec; charges = ctx.charges)
    par_lig = parametrize(ctx.tbl, lig; charges = ctx.charges)
    isempty(par_rec.untyped) || error("$(length(par_rec.untyped)) receptor atom(s) " *
                                      "could not be ACP-typed")
    isempty(par_lig.untyped) || error("$(length(par_lig.untyped)) ligand atom(s) " *
                                      "could not be ACP-typed")

    spec = gridspec_for(rec, lig; spacing = JP.PIPER_SPACING, pad = 2.0)
    fit_rec = grid_fit(spec, rec.coords)

    fit_rec.n_leaking == 0 ||
        error("receptor mass leaks off the grid: $(fit_rec.n_leaking) atom(s), " *
              "$(round(100 * fit_rec.leak_total / length(rec); digits = 2))% of total")

    native = native_ligand_coords(rec, lig)
    camask = ca_mask(lig)
    count(camask) >= 3 || error("only $(count(camask)) C-alpha atoms in the ligand; " *
                                "the reported RMSD would not be the paper's metric")

    bk = ctx.backend
    rs = piper_receptor_spectra(bk, rec, par_rec, ctx.pot, spec;
                                P = ctx.P, w = ctx.w, truncation = ctx.gb,
                                flags = FFTW.MEASURE)
    ligS = JP.adapt_backend(bk, piper_structure(lig, par_lig, ctx.pot; P = ctx.P))
    nchunks = max(1, Threads.nthreads())
    pool = [workspace(bk, spec, 2 + ctx.P; batch = BATCH, flags = FFTW.MEASURE)
            for _ in 1:nchunks]

    rots = ctx.rots
    nrot = BATCH * fld(length(rots.rots), BATCH)
    rots_dev = JP.adapt_rots(bk, nrot == length(rots.rots) ? rots.rots : rots.rots[1:nrot])

    floor_ca = search_floor(rots, spec, lig.coords, native; mask = camask)

    poses = Vector{JP.Pose{T}}(undef, nrot)
    t_dock = @elapsed begin
        JP.dock!(poses, ligS, rots_dev, rs, pool)
        gpu_sync()
    end

    order = sortperm(poses; by = p -> p.energy)
    rmsds = Vector{T}(undef, nrot)
    placed = Vector{SVector{3,T}}(undef, length(lig))
    for (rank, idx) in enumerate(order)
        R, t = pose_transform(poses[idx], rots.rots, spec)
        pose_coords!(placed, lig.coords, R, t)
        rmsds[rank] = ligand_rmsd(placed, native; mask = camask)
    end
    rank_hit = hit_rank(rmsds, HIT)
    best_rmsd, best_at = findmin(rmsds)
    ktop = min(TOPK, nrot)
    nhits = count(<=(HIT), @view rmsds[1:ktop])

    top_idx = order[1]
    R1, t1 = pose_transform(poses[top_idx], rots.rots, spec)
    pose_coords!(placed, lig.coords, R1, t1)
    top_lrmsd = ligand_rmsd(placed, native; mask = camask)
    top_irmsd = interface_rmsd(rec, lig, native, placed; cutoff = 10.0)

    nclu = min(TOPK, nrot)
    Rs, ts = pose_transforms(view(poses, order[1:nclu]), rots.rots, spec)
    S = JP.second_moment(lig.coords)
    D = Matrix{T}(undef, nclu, nclu)
    pairwise_rmsd!(D, Rs, ts, S, length(lig))
    clusters = greedy_cluster(D, T(CLUSTRAD); maxclusters = MAXCLUST)
    bypop = sortperm(clusters; by = length, rev = true)
    native_cluster = 0
    for (ci, c) in enumerate(bypop)

        if rmsds[clusters[c][1]] <= HIT
            native_cluster = ci
            break
        end
    end

    return normalise((
            status = "ok", case = case.id, complex = case.complex,
            category = case.category, difficulty = case.difficulty,
            resolution = case.resolution,
            unbound = false, swapped_roles = swapped,
            rec_chains = case.rec_chains, lig_chains = case.lig_chains,
            rec_atoms = length(rec), lig_atoms = length(lig),
            lig_ca = count(camask),
            rec_qtotal = par_rec.net_charge, lig_qtotal = par_lig.net_charge,
            grid_n = collect(spec.n), ncells = prod(spec.n), spacing = spec.spacing,
            rec_margin_a = fit_rec.margin_angstrom,
            nrot = nrot, search_floor_ca = floor_ca.rmsd,
            search_floor_angle_deg = floor_ca.angle,
            native_rank = rank_hit,
            hits_in_topk = nhits, topk = ktop,
            hit_rate = nhits / ktop,
            best_rmsd_ca = best_rmsd, best_rmsd_rank = best_at,
            rank1_energy = poses[top_idx].energy,
            rank1_lrmsd_ca = top_lrmsd, rank1_irmsd = top_irmsd,
            ncluster = length(clusters),
            native_cluster_rank = native_cluster,
            cluster_sizes = [length(clusters[c]) for c in bypop],
            t_dock_s = t_dock, ms_per_rot = 1000 * t_dock / nrot,
            error = ""))
end

"""
    ROW_TEMPLATE

Every column a row can carry, in `dock_case`'s order, with the measured ones
null.

This exists so a **failure** row has the same keys as a success row. That
promise was previously made in this file's docstrings and not kept: a failed
case carried 5 columns of 37, so a reader indexing `t_dock_s` hit a missing key
on exactly the rows a long batch is most likely to contain -- and on a metered
run those are the rows that cannot be recovered by re-running. Uniformity is
what lets a consumer filter on `status` instead of on key presence.
"""
const ROW_TEMPLATE = (
    status = "", case = "", complex = "", category = "", difficulty = "",
    resolution = "", unbound = nothing, swapped_roles = nothing,
    rec_chains = "", lig_chains = "",
    rec_atoms = nothing, lig_atoms = nothing, lig_ca = nothing,
    rec_qtotal = nothing, lig_qtotal = nothing,
    grid_n = nothing, ncells = nothing, spacing = nothing,
    rec_margin_a = nothing,
    nrot = nothing, search_floor_ca = nothing, search_floor_angle_deg = nothing,
    native_rank = nothing,
    hits_in_topk = nothing, topk = nothing, hit_rate = nothing,
    best_rmsd_ca = nothing, best_rmsd_rank = nothing,
    rank1_energy = nothing, rank1_lrmsd_ca = nothing, rank1_irmsd = nothing,
    ncluster = nothing, native_cluster_rank = nothing, cluster_sizes = nothing,
    t_dock_s = nothing, ms_per_rot = nothing, error = "")

const _SCHEMA_WARNED = Ref(false)

"""
    normalise(row) -> NamedTuple

Force a row onto [`ROW_TEMPLATE`](@ref)'s column set and order.

Normalising rather than asserting is deliberate. Schema drift is a programming
error, but throwing on it here would abort a batch that may already hold fifty
completed cases, which is the one outcome this harness is built to prevent. So a
drifting row is *carried* and the drift is reported once -- loudly, because a
column that exists on success rows and not on failure rows is the defect this
template was added to remove.
"""
function normalise(row::NamedTuple)
    out = merge(ROW_TEMPLATE, row)
    if length(out) != length(ROW_TEMPLATE) && !_SCHEMA_WARNED[]
        _SCHEMA_WARNED[] = true
        @warn "row carries columns absent from ROW_TEMPLATE, so failure rows " *
              "will lack them -- add them to the template" *
              " extra = $(collect(setdiff(keys(out), keys(ROW_TEMPLATE))))"
    end
    return out
end

"""A failure row: the same columns as a success, measured ones null, so a reader
need not special-case it. Everything filled in here comes from the case list
rather than from the run, which is why a failure still says *which* case, in
which category and at what difficulty."""
failed_row(case::Case, status::AbstractString, msg::AbstractString) =
    normalise((status = status, case = case.id, complex = case.complex,
               category = case.category, difficulty = case.difficulty,
               resolution = case.resolution, rec_chains = case.rec_chains,
               lig_chains = case.lig_chains, error = msg))


"""Report recovery and timings using completed cases of a single category."""
function category_summary(rows)
    ok = filter(r -> r.status == "ok", rows)
    n_ok = length(ok)
    rank1 = count(r -> r.native_rank == 1, ok)
    near = count(r -> r.native_rank > 0, ok)
    cluster1 = count(r -> r.native_cluster_rank == 1, ok)
    @printf("%s: %d of %d completed\n", first(rows).category, n_ok, length(rows))
    if !isempty(ok)
        @printf("native rank 1   %d of %d\n", rank1, n_ok)
        @printf("near-native anywhere  %d of %d\n", near, n_ok)
        @printf("near-native cluster rank 1  %d of %d\n", cluster1, n_ok)
        hits = [r.hits_in_topk for r in ok]
        @printf("hits in top %d   median %.1f, range %d-%d\n",
                maximum(r.topk for r in ok), median_of(hits),
                minimum(hits), maximum(hits))
        @printf("wall            %.1f s total, %.2f ms/rotation median\n",
                sum(r.t_dock_s for r in ok), median_of([r.ms_per_rot for r in ok]))
    end
    return (; ncases=length(rows), n_ok, n_missing=count(r -> r.status == "missing", rows),
            n_error=count(r -> r.status == "error", rows), rank1, near, cluster1)
end

function main()
    GPUArm.banner()
    if !HAVE_BIO
        println("\nSKIP -- BioStructures is not in the active environment " *
                "($(GPUArm.active_env_name)).")
        println("       This tier reads PDB files, and the reader is a package " *
                "extension.")
        println("       Run under --project=gpu or --project=test, never " *
                "--project=bench.")
        return 0
    end
    REQGPU && require_gpu!("bench/benchmark51.jl")

    cases = read_cases(CASEFILE)
    if !isempty(ONLY)
        cases = filter(c -> c.id in ONLY, cases)
        isempty(cases) && error("--only matched no case in $CASEFILE")
    end
    LIMIT > 0 && (cases = cases[1:min(LIMIT, end)])
    complete = caselist_complete(cases)

    profile = cluspro_profile(PROFILE)

    wprof = assumed_piper_weights(ClusProProfile{T}(profile))
    w = PiperWeights{T}(T(W1), wprof.w2, wprof.w3)

    pot64 = load_dars()
    P = dars_rank(pot64)
    ctx = (tbl = load_acp_types(), pot = Potential{T}(pot64), P = P, w = w,
           charges = charge_model(CHARGES), gb = gb_model(GBMODE),

           backend = gpu_available() ? JP.CUDABackend() : JP.CPUBackend(),
           rots = rotations_hopf_balanced(T, NROT))

    FFTW.set_num_threads(1)
    BLAS.set_num_threads(1)

    hr("batch harness")
    @printf("cases           %d%s\n", length(cases),
            complete ? "" : "   <-- SHORT LIST, see below")
    @printf("case list       %s\n", scrub(CASEFILE))
    @printf("backend         %s\n", ctx.backend isa JP.CPUBackend ? "CPU" : "GPU")
    @printf("threads         %d\n", Threads.nthreads())
    @printf("element type    %s%s\n", string(T),
            (!T_EXPLICIT && !(ctx.backend isa JP.CPUBackend)) ?
                "   <-- DEFAULT, not chosen: see the cost note below" : "")
    @printf("rotations       %d (%s)\n", length(ctx.rots.rots), ctx.rots.kind)
    @printf("P               %d  (%d terms, %d transforms per rotation)\n",
            P, 2 + P, 3 + P)
    @printf("profile         :%s  (%s)\n", PROFILE, profile.label)
    @printf("weights run     w1 = %.4g  w2 = %.4g  w3 = %.4g\n", w.w1, w.w2, w.w3)
    @printf("  profile said  w1 = %.4g  <-- OVERRIDDEN\n", wprof.w1)
    @printf("charges         %s%s\n", CHARGES,
            CHARGES == "zero" ? "   <-- w2 is inert: the electrostatic field is zero" : "")
    @printf("GB truncation   %s\n", GBMODE)
    @printf("hit threshold   %.1f A C-alpha ligand RMSD, top %d (PDF p.22; 2017 p.5)\n",
            HIT, TOPK)

    if !T_EXPLICIT && !(ctx.backend isa JP.CPUBackend)
        println()
        println("!! --T was not given, so this campaign runs in Float64. On a GPU that")
        println("!! is the expensive choice: consumer cards are much slower in double")
        println("!! precision, and Float32 is the intended precision for a docking")
        println("!! campaign. Pass --T float32 unless the precision is the point.")
    end

    if !complete
        println()
        println("The selection differs from the 52 enzyme/inhibitor cases in BM4.0 Table SI.")
    end

    hr("Cases")
    rows = Any[]
    for (i, case) in enumerate(cases)
        @printf("[%2d/%2d] %-8s ", i, length(cases), case.id)
        flush(stdout)
        row = try
            r = dock_case(case, ctx)
            @printf("ok    rank %-6s hits %3d/%-4d  cluster %-3s  floor %.2f A  %.1f s\n",
                    r.native_rank == 0 ? "none" : string(r.native_rank),
                    r.hits_in_topk, r.topk,
                    r.native_cluster_rank == 0 ? "none" : string(r.native_cluster_rank),
                    r.search_floor_ca, r.t_dock_s)
            r
        catch e
            msg = scrub(first(split(sprint(showerror, e), '\n')))
            status = occursin("structure not found", msg) ? "missing" : "error"
            println(status == "missing" ? "MISSING  (fetch it)" : "ERROR  " * msg)
            failed_row(case, status, msg)
        end
        push!(rows, row)
    end

    ok = [r for r in rows if r.status == "ok"]
    hr("Summary")
    @printf("ran             %d of %d  (%d error, %d missing)\n",
            length(ok), length(rows),
            count(r -> r.status == "error", rows),
            count(r -> r.status == "missing", rows))
    summaries = Dict(category => category_summary(filter(r -> r.category == category, rows))
                     for category in unique(r.category for r in rows))

    nmiss = count(r -> r.status == "missing", rows)
    nerr  = count(r -> r.status == "error", rows)
    ran_all = !isempty(rows) && nmiss == 0 && nerr == 0
    enzyme = get(summaries, "enzyme_inhibitor", nothing)
    is_benchmark = complete && enzyme.n_missing == 0 && enzyme.n_ok > 0
    note = if !complete
        "The selected rows do not cover the 52 canonical BM4.0 enzyme/inhibitor cases."
    else
        "BM4.0 enzyme/inhibitor: $(enzyme.n_ok)/$(enzyme.ncases) completed, " *
        "$(enzyme.n_missing) missing, $(enzyme.n_error) errors. See individual row statuses."
    end
    println("\nVERDICT: ", note)

    if JSONOUT !== nothing
        begin
            doc = (kind = "benchmark-batch",

                   is_benchmark = is_benchmark,
                   caselist_complete = complete,
                   run_complete = ran_all,
                   n_ok = length(ok), n_missing = nmiss, n_error = nerr,
                   caselist = scrub(CASEFILE),
                   ncases = length(cases),
                   by_category = summaries,
                   note = note,
                   date = Base.Libc.strftime("%Y-%m-%d", time()),
                   tag = TAG,
                   machine = Sys.CPU_NAME,
                   julia = string(VERSION),
                   threads = Threads.nthreads(),
                   arm = ctx.backend isa JP.CPUBackend ? "cpu" : "gpu",
                   gpu = provenance(),
                   eltype = string(T),
                   P = P, nterms = 2 + P, batch = BATCH,
                   nrot = length(ctx.rots.rots), rot_kind = string(ctx.rots.kind),
                   profile = String(PROFILE),
                   profile_label = profile.label,
                   profile_weights = (w_rep = profile.w_rep, w_attr = profile.w_attr,
                                      w_elec = profile.w_elec, w_dars = profile.w_dars),
                   profile_status = "ASSUMED -- 2017 four-term coefficients, not " *
                                    "the 2006 three-term form this implements " *
                                    "(a different term structure). w1 overridden.",
                   w1 = w.w1, w2 = w.w2, w3 = w.w3, w1_from_profile = wprof.w1,
                   charges = CHARGES, gb_truncation = GBMODE,
                   hit_threshold = HIT, topk = TOPK,
                   cluster_radius = CLUSTRAD, max_clusters = MAXCLUST,
                   rows = rows)
            open(JSONOUT, "w") do io
                write(io, _j(doc))
            end
            println("\nwrote ", JSONOUT)
        end
    end

    return isempty(ok) && !isempty(rows) ? 1 : 0
end

"""Median without pulling in Statistics: `--project=test` would have to re-resolve."""
function median_of(v)
    isempty(v) && return NaN
    s = sort(collect(float.(v)))
    n = length(s)
    return isodd(n) ? s[(n+1)÷2] : (s[n÷2] + s[n÷2+1]) / 2
end

exit(main())
