#!/usr/bin/env julia


using JuPIPER, BioStructures, StaticArrays, FFTW, LinearAlgebra, Printf, Dates

const JP = JuPIPER
include(joinpath(@__DIR__, "cdr.jl"))
include(joinpath(@__DIR__, "unbound.jl"))
include(joinpath(@__DIR__, "restraintfile.jl"))


function argval(flag::AbstractString, default)
    i = findfirst(==(flag), ARGS)
    return i === nothing || i == length(ARGS) ? default : ARGS[i + 1]
end
argint(f, d)  = (v = argval(f, nothing); v === nothing ? d : parse(Int, v))
argnum(f, d)  = (v = argval(f, nothing); v === nothing ? d : parse(Float64, v))
arglist(f, d) = (v = argval(f, nothing); v === nothing ? d : parse.(Float64, split(v, ',')))
argints(f, d) = (v = argval(f, nothing); v === nothing ? d : parse.(Int, split(v, ',')))

const T = Float64
const DATA = joinpath(@__DIR__, "..", "data")
const STRUCT = joinpath(DATA, "structures")

const ARM        = argval("--arm", "both")
const NROT_MASK  = argint("--nrot-mask", 10_000)
const NROT_REST  = argint("--nrot-restraint", 70_000)
const NROT_BOTH  = argint("--nrot", 0)
const W1         = argnum("--w1", 7.0)
const SPACING    = argnum("--spacing", JP.PIPER_SPACING)
const PAD        = argnum("--pad", 2.0)
const HIT        = argnum("--hit", 10.0)
const NCLUSTER   = argint("--ncluster", 1000)
const CLUSTRAD   = argnum("--cluster-radius", 10.0)
const IFACE_CUT  = argnum("--interface-cutoff", 10.0)
const STRENGTHS  = arglist("--strengths", [1.0, 2.0, 4.0, 8.0])
const SETCUTS    = argints("--set-cutoffs", [0, 4, 9, 14, 19])
const RESTFILE   = argval("--restraints",
                          joinpath(DATA, "restraints", "1f3g_1poh_20noe.json"))
const JSONOUT    = argval("--json", nothing)
const TAG        = argval("--tag", "")
const MACHINE    = argval("--machine", "unknown")

hr(t) = (println(); println(t); println(repeat("-", length(t))))
sub(t) = (println(); println("  ", t); println("  ", repeat("~", length(t))))


struct SubRun
    label::String
    nrot::Int
    t_dock::Float64
    e_rank1::T
    rank_hit::Int
    best_rmsd::T
    best_rank::Int
    irmsd_rank1::T
    nclusters::Int
    near_rank::Int
    near_irmsd::T
    near_pop::Int
    near_lrmsd::T
    top_pop::Int
    top_irmsd::T
    hit_cluster_rank::Int
    skip_rate::Float64
    feasible_frac::Float64
    t_feasible::Float64
    extra::Vector{Pair{String,Any}}
end

"""
    analyse(label, poses, rots, spec, rec, lig, native, camask; ...) -> SubRun

Everything downstream of the landscape: rank the poses, cluster the top
`NCLUSTER`, and find the cluster whose centre is closest to native. That last
is the quantity the reference reports for both of its worked examples ("this
model is the center of the sixth-largest cluster"), so it is the one the A/B
turns on.
"""
function analyse(label, poses, rots, spec, rec, lig, native, camask;
                 skip_rate = 0.0, feasible_frac = 0.0, t_feasible = 0.0,
                 t_dock = 0.0, extra = Pair{String,Any}[])
    nrot = length(poses)
    live = [p for p in poses if isfinite(p.energy) && p.energy != typemax(T)]
    isempty(live) && error("$label: every rotation was skipped -- no pose to score")
    order = sortperm(poses; by = p -> p.energy)

    nlive = count(p -> p.energy != typemax(T), poses)
    order = order[1:nlive]
    placed = Vector{SVector{3,T}}(undef, length(lig))
    rmsds = Vector{T}(undef, nlive)
    for (rank, idx) in enumerate(order)
        R, t = pose_transform(poses[idx], rots.rots, spec)
        pose_coords!(placed, lig.coords, R, t)
        rmsds[rank] = ligand_rmsd(placed, native; mask = camask)
    end
    irms_at(rank) = begin
        R, t = pose_transform(poses[order[rank]], rots.rots, spec)
        pose_coords!(placed, lig.coords, R, t)
        interface_rmsd(rec, lig, native, placed; cutoff = IFACE_CUT)
    end
    rank_hit = hit_rank(rmsds, HIT)
    best_rmsd, best_at = findmin(rmsds)

    nclu = min(NCLUSTER, nlive)
    cl_idx = order[1:nclu]
    Rs, ts = pose_transforms(view(poses, cl_idx), rots.rots, spec)
    S = JP.second_moment(lig.coords)
    D = Matrix{T}(undef, nclu, nclu)
    pairwise_rmsd!(D, Rs, ts, S, length(lig))
    clusters = greedy_cluster(D, CLUSTRAD; maxclusters = 30)
    bypop = sortperm(clusters; by = length, rev = true)

    near_rank = 0; near_irmsd = T(Inf); near_pop = 0; near_lrmsd = T(Inf)
    hit_cluster = 0
    for (ci, c) in enumerate(bypop)
        centre = clusters[c][1]
        ir = irms_at(centre)
        if ir < near_irmsd
            near_irmsd = ir; near_rank = ci
            near_pop = length(clusters[c]); near_lrmsd = rmsds[centre]
        end
        hit_cluster == 0 && rmsds[centre] <= HIT && (hit_cluster = ci)
    end
    top_pop = length(clusters[bypop[1]])
    top_irmsd = irms_at(clusters[bypop[1]][1])

    return SubRun(label, nrot, t_dock, poses[order[1]].energy, rank_hit,
                  best_rmsd, best_at, irms_at(1), length(clusters),
                  near_rank, near_irmsd, near_pop, near_lrmsd,
                  top_pop, top_irmsd, hit_cluster,
                  skip_rate, feasible_frac, t_feasible, extra)
end

function show_subrun(r::SubRun)
    @printf("  %-26s  rank-1 E %10.3f   near-native cluster: rank %2d of %2d, pop %4d, I-RMSD %6.3f A\n",
            r.label, r.e_rank1, r.near_rank, r.nclusters, r.near_pop, r.near_irmsd)
    @printf("  %-26s  top cluster pop %4d I-RMSD %6.3f A   best L-RMSD %6.3f A at rank %d   [%.1f s]\n",
            "", r.top_pop, r.top_irmsd, r.best_rmsd, r.best_rank, r.t_dock)
    r.skip_rate > 0 && @printf("  %-26s  rotations skipped %.2f%%, feasible cells %.4f%%, feasibility %.1f s\n",
                               "", 100r.skip_rate, 100r.feasible_frac, r.t_feasible)
end


need(p) = isfile(p) ? p : error("""
    structure not found: $p
    fetch it first:  bash scripts/fetch_structures.sh "1F3G 1POH 1GGR 1FGN 1TFH 1AHW" """)

function setup_case(; recfile, recchains, ligfile, ligchains, natfile,
                      recmap, ligmap, nrot)
    rec = read_atoms(need(recfile); T = T, chains = recchains)
    lig = read_atoms(need(ligfile); T = T, chains = ligchains)
    nat = read_atoms(need(natfile); T = T)
    println("  receptor  ", rec.report)
    println("  ligand    ", lig.report)
    println("  native    ", nat.report)
    native, diag = unbound_native(rec, lig, nat, recmap, ligmap)
    camask = ca_mask(lig)
    @printf("  receptor superposition residual %.3f A over %d C-alpha\n",
            diag.rec_rmsd, diag.rec_matched)
    @printf("  UNBOUND LIGAND vs NATIVE        %.3f A over %d C-alpha  <- the conformational floor under every RMSD below\n",
            diag.lig_unbound_rmsd, diag.lig_matched)

    tbl = load_acp_types(); pot = load_dars(); P = dars_rank(pot)
    w = PiperWeights{T}(T(W1), T(PIPER_WEIGHTS_ENZYME_INHIBITOR.w2),
                        T(PIPER_WEIGHTS_ENZYME_INHIBITOR.w3))
    par_rec = parametrize(tbl, rec)
    par_lig = parametrize(tbl, lig)
    spec = gridspec_for(rec, lig; spacing = SPACING, pad = PAD)
    fit = grid_fit(spec, rec.coords)
    @printf("  grid %s = %d cells, spacing %.2f A; receptor fit %s\n",
            string(spec.n), prod(spec.n), spec.spacing,
            fit.n_leaking == 0 ? "clean" : "LEAKING $(fit.n_leaking) atoms")
    FFTW.set_num_threads(1); BLAS.set_num_threads(1)
    rots = rotations_hopf_balanced(T, nrot)
    ligS = piper_structure(lig, par_lig, pot; P = P)
    pool = [workspace(JP.CPUBackend(), spec, 2 + P; batch = 1, flags = FFTW.MEASURE)
            for _ in 1:Threads.nthreads()]
    t_floor = @elapsed fl = search_floor(rots, spec, lig.coords, native; mask = camask)
    @printf("  rotations %d (%s); search floor %.3f A C-alpha [%.1f s]\n",
            length(rots), rots.kind, fl.rmsd, t_floor)
    return (; rec, lig, nat, native, camask, diag, par_rec, par_lig, pot, P, w,
              spec, rots, ligS, pool, floor = fl)
end

function build_spectra(c; mask = nothing)
    return piper_receptor_spectra(JP.CPUBackend(), c.rec, c.par_rec, c.pot, c.spec;
                                  P = c.P, w = c.w, flags = FFTW.MEASURE, mask = mask)
end

function dock_plain(c, rs, label)
    nrot = length(c.rots)
    poses = Vector{JP.Pose{T}}(undef, nrot)
    t = @elapsed begin
        step = max(1, cld(nrot, 20))
        for lo in 1:step:nrot
            r = lo:min(lo + step - 1, nrot)
            gp = Vector{JP.Pose{T}}(undef, length(r))
            dock!(gp, c.ligS, view(c.rots.rots, r), rs, c.pool)
            @inbounds for (j, i) in enumerate(r)
                p = gp[j]
                poses[i] = JP.Pose{T}(p.energy, Int32(i), p.tx, p.ty, p.tz)
            end

            if stdout isa Base.TTY
                @printf("\r    %-24s %6d / %d", label, r[end], nrot); flush(stdout)
            end
        end
    end
    stdout isa Base.TTY && print("\r", " "^60, "\r")
    return (poses, t)
end


function arm_mask(nrot)
    hr("arm A -- non-CDR masking:  1FGN (Fab 5G9, receptor) + " *
       "1TFH (tissue factor, ligand) vs 1AHW")
    println("  CONTEXT, not a target (2017 protocol published p.273): automatic " *
            "non-CDR masking gave a")
    println("  4.7 A I-RMSD model as the centre of the 6th-largest cluster (48 " *
            "members); manual masking")
    println("  gave 4.8 A at rank 2 (73 members). Both arms ran Antibody Mode " *
            "(Brenke et al. 2012),")
    println("  which we do not hold -- so neither number is reachable and " *
            "neither is being aimed at.")
    c = setup_case(recfile = joinpath(STRUCT, "1FGN.pdb"), recchains = ["H", "L"],
                   ligfile = joinpath(STRUCT, "1TFH.pdb"), ligchains = "A",
                   natfile = joinpath(STRUCT, "1AHW.pdb"),
                   recmap = ["H" => "B", "L" => "A"], ligmap = ["A" => "C"],
                   nrot = nrot)
    cdr, loops = fab_cdr_residues(c.rec; heavy = "H", light = "L")
    println("\n  CDR annotation (union of Kabat, AbM, Chothia; anchors asserted, " *
            "see scripts/cdr.jl):")
    for (nm, v) in pairs(loops)
        @printf("    %-3s %4d-%-4d %2d  %s\n", String(nm), v[1][1], v[end][1],
                length(v), String([x[3] for x in v]))
    end
    sel = noncdr_selection(c.rec, cdr)
    nres_masked = 0
    @printf("  non-CDR atoms %d of %d (%.1f%%)\n",
            count(sel), length(c.rec), 100count(sel) / length(c.rec))
    xover = mask_crossover(W1)
    @printf("  mask_crossover(w1 = %.1f) = %.3f  <- below this a 'repulsive' mask is net ATTRACTIVE\n",
            W1, xover)

    runs = SubRun[]
    sub("unmasked control")
    rs0 = build_spectra(c)
    p0, t0 = dock_plain(c, rs0, "unmasked")
    r0 = analyse("unmasked", p0, c.rots, c.spec, c.rec, c.lig, c.native, c.camask;
                 t_dock = t0, extra = ["repulsion" => 0.0])
    push!(runs, r0); show_subrun(r0)

    for m in STRENGTHS
        sub(@sprintf("non-CDR repulsion x%.3g%s", m,
                     m < xover ? "   (BELOW CROSSOVER -- expected to attract)" : ""))
        mask = spatial_mask(c.rec, sel; repulsion = m)
        nres_masked = masked_residue_count(c.rec, mask)
        rs = build_spectra(c; mask = mask)
        p, t = dock_plain(c, rs, @sprintf("mask x%.3g", m))
        r = analyse(@sprintf("mask x%.3g", m), p, c.rots, c.spec, c.rec, c.lig,
                    c.native, c.camask; t_dock = t,
                    extra = ["repulsion" => m, "below_crossover" => m < xover])
        push!(runs, r); show_subrun(r)
    end
    return (runs, (; case = "1FGN+1TFH vs 1AHW", nrot = length(c.rots),
                     ncells = prod(c.spec.n), grid = collect(c.spec.n),
                     search_floor = c.floor.rmsd,
                     rec_superposition_rmsd = c.diag.rec_rmsd,
                     rec_ca_matched = c.diag.rec_matched,
                     lig_unbound_rmsd = c.diag.lig_unbound_rmsd,
                     lig_ca_matched = c.diag.lig_matched,
                     cdr_residues = length(cdr), masked_residues = nres_masked,
                     noncdr_atom_frac = count(sel) / length(c.rec),
                     mask_crossover = xover,
                     loops = [String(nm) => (v[1][1], v[end][1]) for (nm, v) in pairs(loops)]))
end


function arm_restraint(nrot)
    hr("arm B -- NOE/AIR distance restraints:  1F3G (E2A, receptor) + " *
       "1POH (HPr, ligand) vs 1GGR")
    println("  CONTEXT, not a target (2017 protocol published p.273): unrestrained, " *
            "the near-native model")
    println("  is the centre of the RANK-2 cluster (221 members) at 3.78 A I-RMSD, " *
            "the rank-1 cluster being")
    println("  >20 A away; with the restraints it becomes RANK 1 (268 members) at " *
            "2.88 A. Different scoring")
    println("  function, different grid spacing -- the direction is the comparable " *
            "part, not the values.")
    c = setup_case(recfile = joinpath(STRUCT, "1F3G.pdb"), recchains = "A",
                   ligfile = joinpath(STRUCT, "1POH.pdb"), ligchains = "A",
                   natfile = joinpath(STRUCT, "1GGR.pdb"),
                   recmap = ["A" => "A"], ligmap = ["A" => "B"], nrot = nrot)
    groups, meta = load_restraint_groups(RESTFILE)
    @printf("\n  restraints %d groups, %d residue pairs, from %s\n",
            meta.n, sum(length(g.rec) * length(g.lig) for g in groups),
            basename(RESTFILE))
    @printf("  source AIR sha256 %s\n", meta.air_sha)
    @printf("  distance window [%.1f, %.1f] A, per-group cutoff %d\n",
            groups[1].dmin, groups[1].dmax, groups[1].cutoff)

    runs = SubRun[]
    sub("unrestrained control")
    rs = build_spectra(c)
    p0, t0 = dock_plain(c, rs, "unrestrained")
    r0 = analyse("unrestrained", p0, c.rots, c.spec, c.rec, c.lig, c.native, c.camask;
                 t_dock = t0, extra = ["set_cutoff" => -1])
    push!(runs, r0); show_subrun(r0)

    recg = c.rec.coords
    for k in SETCUTS
        k < length(groups) || continue
        sub(@sprintf("restrained, set cutoff %d  (more than %d of %d groups satisfied)",
                     k, k, length(groups)))
        set = build_restraint_set(groups, c.rec, c.lig, k)
        poses = Vector{JP.Pose{T}}(undef, length(c.rots))
        t = @elapsed (_, rep) = dock_restrained!(poses, c.ligS, c.rots.rots, rs,
                                                 c.pool, set, recg)
        println("    ", rep)
        if rep.nskipped == length(c.rots)
            @printf("    EVERY rotation had an empty feasible set at cutoff %d -- not scoreable,\n    and that is a result: the set is unsatisfiable on this grid at this strictness.\n", k)
            push!(runs, SubRun(@sprintf("restrained k>%d", k), length(c.rots), t,
                               T(NaN), 0, T(NaN), 0, T(NaN), 0, 0, T(NaN), 0, T(NaN),
                               0, T(NaN), 0, skip_rate(rep), feasible_fraction(rep),
                               rep.t_feasible, ["set_cutoff" => k,
                                                "unsatisfiable" => true]))
            continue
        end
        r = analyse(@sprintf("restrained k>%d", k), poses, c.rots, c.spec, c.rec,
                    c.lig, c.native, c.camask; t_dock = t,
                    skip_rate = skip_rate(rep), feasible_frac = feasible_fraction(rep),
                    t_feasible = rep.t_feasible,
                    extra = ["set_cutoff" => k, "unsatisfiable" => false,
                             "nskipped" => rep.nskipped])
        push!(runs, r); show_subrun(r)
    end
    return (runs, (; case = "1F3G+1POH vs 1GGR", nrot = length(c.rots),
                     ncells = prod(c.spec.n), grid = collect(c.spec.n),
                     search_floor = c.floor.rmsd,
                     rec_superposition_rmsd = c.diag.rec_rmsd,
                     rec_ca_matched = c.diag.rec_matched,
                     lig_unbound_rmsd = c.diag.lig_unbound_rmsd,
                     lig_ca_matched = c.diag.lig_matched,
                     ngroups = length(groups),
                     air_sha256 = meta.air_sha,
                     dmin = groups[1].dmin, dmax = groups[1].dmax))
end


function verdict(name, runs, control_label)
    hr("Verdict -- $name")
    ctl = findfirst(r -> r.label == control_label, runs)
    ctl === nothing && (println("no control arm"); return (improved = false,))
    c = runs[ctl]
    @printf("  control (%s): near-native cluster rank %d, I-RMSD %.3f A, pop %d\n",
            c.label, c.near_rank, c.near_irmsd, c.near_pop)
    best = nothing
    for r in runs
        r === c && continue
        isnan(r.near_irmsd) && continue
        drank = c.near_rank - r.near_rank
        dirms = c.near_irmsd - r.near_irmsd
        @printf("  %-26s rank %2d (%+d)   I-RMSD %6.3f (%+.3f)   pop %4d (%+d)%s\n",
                r.label, r.near_rank, drank, r.near_irmsd, dirms, r.near_pop,
                r.near_pop - c.near_pop,
                (drank > 0 || (drank == 0 && dirms > 0)) ? "   <-- improved" : "")
        if best === nothing || (r.near_rank, r.near_irmsd) < (best.near_rank, best.near_irmsd)
            best = r
        end
    end
    improved = best !== nothing &&
               ((best.near_rank < c.near_rank) ||
                (best.near_rank == c.near_rank && best.near_irmsd < c.near_irmsd))
    @printf("\n  EXIT CRITERION: near-native cluster rank and/or I-RMSD improves\n  relative to the control arm of this same run -- %s\n",
            improved ? "MET" : "NOT MET")
    best !== nothing && @printf("  best arm: %s (rank %d, I-RMSD %.3f A)\n",
                                best.label, best.near_rank, best.near_irmsd)
    return (improved = improved, best = best === nothing ? "" : best.label)
end

jq(s) = string('"', replace(string(s), '\\' => "\\\\", '"' => "\\\"", '\n' => " "), '"')
jval(x::Real) = isfinite(x) ? string(x) : "null"
jval(x::Bool) = x ? "true" : "false"
jval(x::AbstractString) = jq(x)
jval(x) = jq(string(x))

function run_json(r::SubRun)
    fields = ["label" => jq(r.label), "nrot" => r.nrot, "t_dock_s" => jval(r.t_dock),
              "energy_rank1" => jval(r.e_rank1), "rank_hit" => r.rank_hit,
              "best_lrmsd_ca" => jval(r.best_rmsd), "best_rank" => r.best_rank,
              "irmsd_rank1" => jval(r.irmsd_rank1), "nclusters" => r.nclusters,
              "near_cluster_rank" => r.near_rank, "near_cluster_irmsd" => jval(r.near_irmsd),
              "near_cluster_pop" => r.near_pop, "near_cluster_lrmsd_ca" => jval(r.near_lrmsd),
              "top_cluster_pop" => r.top_pop, "top_cluster_irmsd" => jval(r.top_irmsd),
              "hit_cluster_rank" => r.hit_cluster_rank, "skip_rate" => jval(r.skip_rate),
              "feasible_frac" => jval(r.feasible_frac), "t_feasible_s" => jval(r.t_feasible)]
    for (k, v) in r.extra
        push!(fields, k => jval(v))
    end
    return "{" * join(["$(jq(k)): $v" for (k, v) in fields], ", ") * "}"
end

function write_json(path, blocks)
    open(path, "w") do io
        println(io, "{")
        println(io, "  \"schema\": \"jupiper-constraints/1\",")
        println(io, "  \"generated\": ", jq(string(now())), ",")
        println(io, "  \"machine\": ", jq(MACHINE), ",")
        println(io, "  \"tag\": ", jq(TAG), ",")
        println(io, "  \"julia\": ", jq(string(VERSION)), ",")
        println(io, "  \"threads\": ", Threads.nthreads(), ",")
        println(io, "  \"w1\": ", W1, ", \"w1_status\": ",
                jq("[ASSUMED]; 2006 publishes 2.1, 2017's parameterisation " *
                   "is a different energy"), ",")
        println(io, "  \"cluster_radius\": ", CLUSTRAD,
                ", \"cluster_metric\": ", jq("ligand all-atom RMSD (closed form)"),
                ", \"ncluster\": ", NCLUSTER, ",")
        println(io, "  \"hit_threshold_ca\": ", HIT, ",")
        println(io, "  \"claims\": ", jq("Within-run A/B only. No agreement with the " *
                "reference's published ranks is claimed or sought: both arms of its " *
                "masking example run Antibody Mode (Brenke et al. 2012, not held) and " *
                "its absolute I-RMSDs come from a different scoring function on a 1.0 A " *
                "grid."), ",")
        println(io, "  \"arms\": [")
        for (bi, b) in enumerate(blocks)
            println(io, "    {")
            println(io, "      \"arm\": ", jq(b.arm), ",")
            println(io, "      \"meta\": {", join(["$(jq(k)): $(jval(v))"
                        for (k, v) in pairs(b.meta) if !(v isa AbstractVector)], ", "), "},")
            for (k, v) in pairs(b.meta)
                v isa AbstractVector || continue
                println(io, "      ", jq(String(k)), ": [",
                        join([v[i] isa Pair ? "{$(jq(v[i].first)): [$(join(v[i].second, ", "))]}" : string(v[i])
                              for i in eachindex(v)], ", "), "],")
            end
            println(io, "      \"exit_criterion_met\": ", jval(b.verdict.improved), ",")
            println(io, "      \"best_arm\": ", jq(get(b.verdict, :best, "")), ",")
            println(io, "      \"runs\": [")
            for (i, r) in enumerate(b.runs)
                println(io, "        ", run_json(r), i == length(b.runs) ? "" : ",")
            end
            println(io, "      ]")
            println(io, "    }", bi == length(blocks) ? "" : ",")
        end
        println(io, "  ]")
        println(io, "}")
    end
end


hr("biological validation of masking and restraints")
@printf("threads   %d\nJulia     %s\nw1        %.2f  [ASSUMED",
        Threads.nthreads(), VERSION, W1)
@printf("clustering %d poses, %.1f A radius, ligand all-atom RMSD, max 30 clusters\n",
        NCLUSTER, CLUSTRAD)

blocks = NamedTuple[]
if ARM in ("restraint", "both")
    runs, meta = arm_restraint(NROT_BOTH > 0 ? NROT_BOTH : NROT_REST)
    v = verdict("arm B, distance restraints", runs, "unrestrained")
    push!(blocks, (arm = "restraint", runs = runs, meta = meta, verdict = v))
end
if ARM in ("mask", "both")
    runs, meta = arm_mask(NROT_BOTH > 0 ? NROT_BOTH : NROT_MASK)
    v = verdict("arm A, non-CDR masking", runs, "unmasked")
    push!(blocks, (arm = "mask", runs = runs, meta = meta, verdict = v))
end

if JSONOUT !== nothing
    write_json(JSONOUT, blocks)
    println("\nwrote ", JSONOUT)
end
