#!/usr/bin/env julia
"""
Every other tier in this repository measures a *synthetic* case -- `make_case` draws
random coordinates and random term weights, which is right for a throughput roofline and
says nothing about whether the device reproduces a docking *result*. This one runs
E2A-HPr (PDB 1GGR) end to end: real ingestion, real ACP typing, the three receptor
fields, the `P+1` correlation, the translation scan, and the pose geometry -- on both
arms, over the same rotation set, in the same session.

# What it is for, and what it is not for

It answers three questions no synthetic tier can.

  1. Every GPU number this project holds is for `correlate!` alone. A tier that calls
     `dock!` is what keeps that from silently regressing.
  2. **Does the device reach the same pose?** [`pose_parity`](@ref) compares the two
     arms rotation by rotation -- energies to a floating-point tolerance, and the
     selected cell exactly.

It is **not** a validation run. `scripts/validate_bound.jl` is, it is the
authority on the physics, and it is where the committed w1 scan lives. This tier
reproduces its ranking as a *check on the device*, at whatever `--nrot` it was
given, and a rank computed over 2,000 rotations is not comparable to one computed
over 69,960.

# Environment

Needs `BioStructures` (the PDB reader is a package extension), so it runs under
`--project=gpu` or `--project=test`, **not** under `--project=bench`, which is
deliberately thin. It skips with an explanation rather than failing, because `bench/`
staying free of both CUDA and BioStructures is what keeps the committed baseline
manifest valid.

The structure files are fetched, not committed (`data/structures/` is
gitignored):

    bash scripts/fetch_structures.sh 1GGR

Usage:
    julia --project=gpu -t 1 bench/realdock.jl [--json out.json] [--T float32]
                                               [--nrot 69960] [--cpu-nrot 2000]
                                               [--w1 7.0] [--batch 1]
"""

using JuPIPER, FFTW, StaticArrays, LinearAlgebra, Printf
const JP = JuPIPER

include(joinpath(@__DIR__, "gpuarm.jl"))
using .GPUArm


const HAVE_BIO  = Base.identify_package("BioStructures") !== nothing
const HAVE_JSON = Base.identify_package("JSON3") !== nothing
HAVE_BIO  && @eval using BioStructures
HAVE_JSON && @eval using JSON3


function argval(flag::AbstractString, default)
    i = findfirst(==(flag), ARGS)
    i === nothing && return default
    i == length(ARGS) && error("$flag needs a value")
    return ARGS[i+1]
end
argint(flag, d) = (v = argval(flag, nothing); v === nothing ? d : parse(Int, v))
argnum(flag, d) = (v = argval(flag, nothing); v === nothing ? d : parse(Float64, v))

const CASE     = argval("--case", "1GGR")
const PDBFILE  = argval("--complex",
                        joinpath(@__DIR__, "..", "data", "structures", CASE * ".pdb"))
const RECCHAIN = argval("--rec-chain", "A")
const LIGCHAIN = argval("--lig-chain", "B")
const NROT     = argint("--nrot", 70_000)
const CPUNROT  = argint("--cpu-nrot", 2_000)
const BATCH    = argint("--batch", 1)


const W1       = argnum("--w1", 7.0)
const HIT      = argnum("--hit", 10.0)
const ELT      = lowercase(argval("--T", "float64"))
const JSONOUT  = argval("--json", nothing)

const T = ELT == "float32" ? Float32 :
          ELT == "float64" ? Float64 :
          error("--T must be float32 or float64, got $ELT")

hr(t) = (println(); println(t); println(repeat("-", length(t))))

function timeit(f!, warm::Int = 2, reps::Int = 10)
    for _ in 1:warm; f!(); end
    best = Inf
    for _ in 1:reps
        best = min(best, @elapsed f!())
    end
    best
end


"""
    build_case() -> NamedTuple

Ingest, type, parametrise and grid the complex. Host-side throughout: the
receptor fields are built into host `Array{T,3}` and only
[`spectra_from_grids`](@ref) moves them, so this is shared by both arms verbatim
and neither can see different physics from the other.
"""
function build_case()
    rec = read_atoms(PDBFILE; T = T, chains = RECCHAIN)
    lig = read_atoms(PDBFILE; T = T, chains = LIGCHAIN)
    tbl = load_acp_types()

    pot64 = load_dars()
    P     = dars_rank(pot64)
    pot   = Potential{T}(pot64)
    w   = PiperWeights{T}(T(W1), T(PIPER_WEIGHTS_ENZYME_INHIBITOR.w2),
                          T(PIPER_WEIGHTS_ENZYME_INHIBITOR.w3))
    par_rec = parametrize(tbl, rec)
    par_lig = parametrize(tbl, lig)
    spec = gridspec_for(rec, lig; spacing = JP.PIPER_SPACING, pad = 2.0)
    rots = rotations_hopf_balanced(T, NROT)
    native = native_ligand_coords(rec, lig)
    camask = ca_mask(lig)
    return (; rec, lig, pot, P, w, par_rec, par_lig, spec, rots, native, camask)
end

"""Receptor spectra, ligand structure and a one-entry workspace pool on `bk`."""
function arm_setup(bk, c; flags = FFTW.MEASURE)
    rs   = piper_receptor_spectra(bk, c.rec, c.par_rec, c.pot, c.spec;
                                  P = c.P, w = c.w, flags = flags)
    ligS = piper_structure(c.lig, c.par_lig, c.pot; P = c.P)
    ws   = workspace(bk, c.spec, 2 + c.P; batch = BATCH, flags = flags)
    return (rs = rs, lig = JP.adapt_backend(bk, ligS), pool = [ws],
            rots = JP.adapt_rots(bk, c.rots.rots))
end


"""
    run_arm(name, bk, c, nrot; timer) -> NamedTuple

One arm: `nrot` rotations through `dock!`, plus an isolated timing of
`correlate!` and `scan_best!` at this case's real grid size.

`nrot` is rounded down to a multiple of `BATCH` because `dock!` refuses a ragged
batch rather than silently dropping rotations, and it is right to.
"""
function run_arm(name::AbstractString, bk, c, nrot::Integer; timer)
    st = arm_setup(bk, c)
    n = BATCH * fld(min(nrot, length(c.rots.rots)), BATCH)
    poses = Vector{JP.Pose{T}}(undef, n)

    rots_run = n == length(st.rots) ? st.rots : st.rots[1:n]
    warm_rots = st.rots[1:BATCH]

    warm = Vector{JP.Pose{T}}(undef, BATCH)
    JP.dock!(warm, st.lig, warm_rots, st.rs, st.pool)
    gpu_sync()
    t_dock = @elapsed begin
        JP.dock!(poses, st.lig, rots_run, st.rs, st.pool)
        gpu_sync()
    end

    ws = first(st.pool)
    nz = c.spec.n[3]
    vals = similar(ws.score, T, nz, BATCH)
    idxs = similar(ws.score, Int32, nz, BATCH)
    rb = @view st.rots[1:BATCH]
    t_corr = timer(() -> JP.correlate!(ws, st.lig, rb, st.rs))
    t_scan = timer(() -> JP.scan_best!(vals, idxs, ws.score))

    return (; arm = name, T = string(T), case = CASE, n = collect(c.spec.n),
            ncells = prod(c.spec.n), P = c.P, nterms = 2 + c.P, batch = BATCH,
            w1 = W1, natoms_rec = length(c.rec), natoms_lig = length(c.lig),
            nrot = n, t_dock, ms_per_rot = 1000 * t_dock / n,
            t_correlate = t_corr, t_scan,
            scan_frac_of_correlate = t_scan / t_corr,
            extrap_70k_s = 70_000 * t_dock / n,
            poses)
end

"""
    pose_parity(a, b) -> NamedTuple

Rotation-by-rotation agreement between two arms over their common prefix.

`cell_agree` is meaningful here in a way it is not in `bench/pipeline.jl`: that
fixture's minimum is a plateau at numerical zero, so which cell wins is decided by
round-off. A real receptor field has an isolated minimum, so a cell disagreement is a
signal rather than a tie-break.

`rel_energy` is scaled by the landscape's own energy range over the compared
poses, not by an absolute tolerance, so it means the same thing at any `w1`.
"""
function pose_parity(a, b)
    n = min(length(a.poses), length(b.poses))
    n == 0 && return nothing
    ea = [Float64(a.poses[i].energy) for i in 1:n]
    eb = [Float64(b.poses[i].energy) for i in 1:n]
    lo, hi = extrema(ea)
    rng = hi - lo
    dmax = maximum(abs, ea .- eb)
    agree = count(i -> a.poses[i].tx == b.poses[i].tx &&
                       a.poses[i].ty == b.poses[i].ty &&
                       a.poses[i].tz == b.poses[i].tz, 1:n)

    ba = argmin(ea); bb = argmin(eb)
    return (compared = n, abs_max = dmax, energy_range = rng,
            rel_energy = rng > 0 ? dmax / rng : dmax,
            cell_agree = agree / n,
            best_rot_agree = ba == bb,
            best_energy_gap = abs(ea[ba] - ea[bb]) / max(rng, eps()))
end


"""
    score_poses(c, poses) -> NamedTuple

Rank the poses by energy and report the native pose's rank against the search
floor.

**Read `best_rmsd` against `search_floor`, never against 0 A**: the native orientation
is almost surely not a member of a finite rotation set, so a bound-bound run has a
nonzero floor by construction. The floor is computed without consulting any energy, so
it cannot be flattered by the result.
"""
function score_poses(c, poses)
    nrot = length(poses)
    order = sortperm(poses; by = p -> p.energy)
    rmsds = Vector{T}(undef, nrot)
    placed = Vector{SVector{3,T}}(undef, length(c.lig))
    for (rank, idx) in enumerate(order)
        R, t = pose_transform(poses[idx], c.rots.rots, c.spec)
        pose_coords!(placed, c.lig.coords, R, t)
        rmsds[rank] = ligand_rmsd(placed, c.native; mask = c.camask)
    end
    floor_ca = search_floor(view(c.rots.rots, 1:nrot), c.spec, c.lig.coords, c.native;
                            mask = c.camask)
    rank_hit = findfirst(<=(T(HIT)), rmsds)
    best_rmsd, best_at = findmin(rmsds)
    return (; nrot, search_floor = floor_ca.rmsd,
            native_rank = rank_hit === nothing ? 0 : rank_hit,
            best_rmsd, best_rank = best_at,
            rank1_rmsd = rmsds[1], rank1_energy = poses[order[1]].energy,
            hits_top100 = count(<=(T(HIT)), @view rmsds[1:min(100, nrot)]))
end


function report(r)
    @printf("\n[%s] %s  %s  %s  P=%d (%d terms)  batch=%d\n", uppercase(r.arm), r.case,
            r.T, string(Tuple(r.n)), r.P, r.nterms, r.batch)
    @printf("  rotations docked      %9d\n", r.nrot)
    @printf("  wall clock            %9.3f s   (%.4f ms/rotation)\n",
            r.t_dock, r.ms_per_rot)
    @printf("  correlate!            %9.4f ms/rotation\n", 1000 * r.t_correlate)
    @printf("  scan_best!            %9.4f ms/rotation   (%.1f%% of correlate!)\n",
            1000 * r.t_scan, 100 * r.scan_frac_of_correlate)
    @printf("  extrapolated to 70,000 rotations: %8.1f s  (%.2f min)\n",
            r.extrap_70k_s, r.extrap_70k_s / 60)
end

function report_score(s, label)
    @printf("\n  %s -- %d rotations\n", label, s.nrot)
    @printf("    search floor        %8.3f A   <- the best this discrete space contains\n",
            s.search_floor)
    if s.native_rank == 0
        @printf("    native pose         NOT within %.1f A among %d poses\n", HIT, s.nrot)
    else
        @printf("    native pose         rank %d of %d\n", s.native_rank, s.nrot)
    end
    @printf("    best RMSD anywhere  %8.3f A at rank %d\n", s.best_rmsd, s.best_rank)
    @printf("    rank 1              E = %.4f, L-RMSD(CA) = %.3f A\n",
            s.rank1_energy, s.rank1_rmsd)
end

function skip(reason)
    println("\nSKIPPED -- ", reason)
    println("This script needs BioStructures (the PDB reader is a package extension)")
    println("and the fetched structure files. Run it under")
    println("`--project=gpu` or `--project=test`, never `--project=bench`.")
    return nothing
end

function main()
    println("JuPIPER real-protein dock: ", CASE, " chains ", RECCHAIN, "/", LIGCHAIN)
    println("$(Sys.CPU_NAME) | $(Sys.CPU_THREADS) logical cores | Julia $(VERSION) | threads=$(Threads.nthreads)")
    FFTW.set_num_threads(1)
    BLAS.set_num_threads(1)
    require_gpu!("bench/realdock.jl")
    GPUArm.banner()

    HAVE_BIO || return skip("BioStructures is not in the active environment ($(GPUArm.active_env_name))")
    isfile(PDBFILE) || return skip("structure not found: $(basename(PDBFILE)) " *
                                   "-- run `bash scripts/fetch_structures.sh $CASE`")

    hr("Case")
    c = build_case()
    @printf("receptor        %s  %d heavy atoms\n", RECCHAIN, length(c.rec))
    @printf("ligand          %s  %d heavy atoms\n", LIGCHAIN, length(c.lig))
    @printf("grid            %s at %.2f A  (%d cells)\n", string(c.spec.n),
            c.spec.spacing, prod(c.spec.n))
    @printf("scoring         P = %d, %d terms, %d transforms/rotation, w1 = %.2f\n",
            c.P, 2 + c.P, 3 + c.P, W1)
    @printf("rotations       %d available (%s)\n", length(c.rots.rots), c.rots.kind)

    rows = Any[]
    hr("CPU arm")
    cpu = run_arm("cpu", JP.CPUBackend(), c, CPUNROT; timer = timeit)
    report(cpu); push!(rows, Base.structdiff(cpu, (poses = nothing,)))

    gpu = nothing
    parity = nothing
    if gpu_available()
        hr("GPU arm -- same process, same case, same session")
        gpu = run_arm("gpu", JP.CUDABackend(), c, NROT; timer = timeit_gpu)
        report(gpu); push!(rows, Base.structdiff(gpu, (poses = nothing,)))

        hr("Pose parity, CPU vs GPU")
        parity = pose_parity(cpu, gpu)
        if parity !== nothing
            @printf("  compared            %d rotations (the CPU arm's prefix)\n",
                    parity.compared)
            @printf("  max |dE|            %.3e  (%.3e of the energy range)\n",
                    parity.abs_max, parity.rel_energy)
            @printf("  same cell chosen    %.2f%% of rotations\n", 100 * parity.cell_agree)
            @printf("  same best rotation  %s\n", parity.best_rot_agree ? "yes" : "NO")
        end

        @printf("\n  per-rotation speedup: %.1fx on the whole dock, %.1fx on correlate!, %.1fx on scan_best!\n",
                cpu.ms_per_rot / gpu.ms_per_rot,
                cpu.t_correlate / gpu.t_correlate,
                cpu.t_scan / gpu.t_scan)
    else
        println("\nGPU arm skipped -- no CUDA in the active environment.")
    end

    hr("Did it dock?")
    full = gpu === nothing ? cpu : gpu
    s = score_poses(c, full.poses)
    report_score(s, uppercase(full.arm) * " arm")

    if JSONOUT !== nothing && !HAVE_JSON
        @warn "--json was given but JSON3 is not in the active environment; no file written" env=GPUArm.active_env_name()
    elseif JSONOUT !== nothing
        open(JSONOUT, "w") do io
            JSON3.write(io, (machine = Sys.CPU_NAME, julia = string(VERSION),
                             threads = Threads.nthreads(), tier = "realdock",
                             case = CASE, w1 = W1, hit = HIT,
                             gpu = provenance(), rows = rows,
                             parity = parity, score = s, score_arm = full.arm))
        end
        println("\nwrote ", JSONOUT)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    main()
end
