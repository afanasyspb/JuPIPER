#!/usr/bin/env julia


using JuPIPER, BioStructures, StaticArrays, FFTW, LinearAlgebra, Printf

const JP = JuPIPER


function argval(flag::AbstractString, default)
    i = findfirst(==(flag), ARGS)
    i === nothing && return default
    i == length(ARGS) && error("$flag needs a value")
    return ARGS[i+1]
end
argflag(flag::AbstractString) = findfirst(==(flag), ARGS) !== nothing
argint(flag, d) = (v = argval(flag, nothing); v === nothing ? d : parse(Int, v))
argnum(flag, d) = (v = argval(flag, nothing); v === nothing ? d : parse(Float64, v))

const T = Float64

const CASE        = argval("--case", "1GGR")
const COMPLEX     = argval("--complex", joinpath(@__DIR__, "..", "data", "structures", CASE * ".pdb"))
const REC_CHAIN   = argval("--rec-chain", "A")
const LIG_CHAIN   = argval("--lig-chain", "B")
const NROT        = argint("--nrot", 70_000)
const SPACING     = argnum("--spacing", JP.PIPER_SPACING)
const PAD         = argnum("--pad", 2.0)
const CHARGES     = argval("--charges", "zero")
const GBMODE      = argval("--gb", "capped")
const W1          = argnum("--w1", JP.DEFAULT_W_REP)
const HIT         = argnum("--hit", 10.0)
const TOPK        = argint("--topk", 2000)
const NCLUSTER    = argint("--ncluster", 1000)
const CLUSTRAD    = argnum("--cluster-radius", 10.0)
const IFACE_CUT   = argnum("--interface-cutoff", 10.0)
const BATCH       = argint("--batch", 1)
const TAG         = argval("--tag", "")
const JSONOUT     = argval("--json", nothing)
const PARG        = argint("--P", 0)
const CTERM_COOH  = argflag("--terminal-carboxylate")

charge_model(s) = s == "zero"   ? ZeroCharges()   :
                  s == "formal" ? FormalCharges() :
                  error("--charges must be zero or formal, got $s")
gb_model(s)     = s == "capped"    ? GBCapped()    :
                  s == "truncated" ? GBTruncated() :
                  error("--gb must be capped or truncated, got $s")


hr(title) = (println(); println(title); println(repeat("-", length(title))))

hr("bound-bound validation")
@printf("case            %s (%s)\n", CASE, basename(COMPLEX))
@printf("receptor chain  %s\n", REC_CHAIN)
@printf("ligand chain    %s\n", LIG_CHAIN)
@printf("threads         %d\n", Threads.nthreads())

isfile(COMPLEX) || error("""
    structure not found: $COMPLEX
    fetch it first:  bash scripts/fetch_structures.sh $CASE""")

rec = read_atoms(COMPLEX; T = T, chains = REC_CHAIN)
lig = read_atoms(COMPLEX; T = T, chains = LIG_CHAIN)
println()
println("receptor  ", rec.report)
println("ligand    ", lig.report)

length(rec) >= length(lig) || @warn """
    the ligand has more atoms than the receptor. The preprint fixes the frame on
    "the unbound receptor (the larger protein)" (PDF p.22); swapping the roles
    changes which molecule the RMSD is reported for."""


native = native_ligand_coords(rec, lig)
camask = ca_mask(lig)
@printf("\nC-alpha atoms in ligand   %d of %d\n", count(camask), length(lig))


tbl = load_acp_types()
pot = load_dars()
P   = PARG > 0 ? PARG : dars_rank(pot)
cm  = charge_model(CHARGES)
gbm = gb_model(GBMODE)
w   = PiperWeights{T}(T(W1), T(PIPER_WEIGHTS_ENZYME_INHIBITOR.w2),
                      T(PIPER_WEIGHTS_ENZYME_INHIBITOR.w3))

par_rec = parametrize(tbl, rec; charges = cm, terminal_carboxylate = CTERM_COOH)
par_lig = parametrize(tbl, lig; charges = cm, terminal_carboxylate = CTERM_COOH)

hr("Scoring function")
_, trunc_rel = truncation_error(pot, P)
@printf("P               %d  (dars_rank = %d, rank-%d Frobenius error %.1f%%)\n",
        P, dars_rank(pot), P, 100 * trunc_rel)
@printf("weights         w1 = %.4g  w2 = %.4g  w3 = %.4g\n", w.w1, w.w2, w.w3)
@printf("charges         %s (receptor Q = %+.3f, ligand Q = %+.3f)\n",
        CHARGES, par_rec.net_charge, par_lig.net_charge)
@printf("GB truncation   %s\n", GBMODE)
@printf("terms           %d  (%d transforms per rotation)\n", 2 + P, 3 + P)


spec = gridspec_for(rec, lig; spacing = SPACING, pad = PAD)
fit_rec = grid_fit(spec, rec.coords)
fit_lig = grid_fit_rotated(spec, lig.coords)

hr("Grid")
@printf("n               %s  (spacing %.2f A, %.1f A per side)\n",
        string(spec.n), spec.spacing, spec.n[1] * spec.spacing)
println("receptor fit    ", fit_rec)
println("ligand fit      ", fit_lig, "   (worst case over all rotations)")
fit_rec.n_leaking == 0 || @warn "receptor mass is leaking off the grid" fit_rec
fit_rec.margin_angstrom >= JP.PIPER_D || @warn """
    receptor margin $(round(fit_rec.margin_angstrom; digits=2)) A is below the \
    $(JP.PIPER_D) A the receptor field extends, so the field itself is clipped \
    even though every atom landed (see GridFit)."""


FFTW.set_num_threads(1)
BLAS.set_num_threads(1)

rots = rotations_hopf_balanced(T, NROT)
nrot = length(rots)
hr("Search space")
@printf("rotations       %d (%s, %d directions x %d spins)\n",
        nrot, rots.kind, rots.ndir, rots.nspin)
@printf("translations    %d per rotation\n", prod(spec.n))

t_setup = @elapsed begin
    rs = piper_receptor_spectra(JP.CPUBackend(), rec, par_rec, pot, spec;
                                P = P, w = w, truncation = gbm, flags = FFTW.MEASURE)
    ligS = piper_structure(lig, par_lig, pot; P = P)
    nchunks = Threads.nthreads()
    pool = [workspace(JP.CPUBackend(), spec, 2 + P; batch = BATCH, flags = FFTW.MEASURE)
            for _ in 1:nchunks]
end
@printf("setup           %.1f s (%d workspaces, %.1f MiB each)\n",
        t_setup, length(pool), JP.workspace_bytes(first(pool)) / 2^20)


t_floor = @elapsed floor_ca = search_floor(rots, spec, lig.coords, native; mask = camask)
@printf("search floor    %.3f A C-alpha (rotation %d, %.2f deg off, cells %s) [%.1f s]\n",
        floor_ca.rmsd, floor_ca.rot, floor_ca.angle, string(floor_ca.cells), t_floor)


hr("Docking")
poses = Vector{JP.Pose{T}}(undef, nrot)
ngroups = max(1, min(20, cld(nrot, 1000)))


bounds = unique!([BATCH * fld(round(Int, b), BATCH)
                  for b in range(0, nrot; length = ngroups + 1)])
bounds[end] == nrot || push!(bounds, nrot)
ngroups = length(bounds) - 1
t0 = time()
for g in 1:ngroups
    r = (bounds[g]+1):bounds[g+1]
    isempty(r) && continue
    gp = Vector{JP.Pose{T}}(undef, length(r))
    dock!(gp, ligS, view(rots.rots, r), rs, pool)

    @inbounds for (j, i) in enumerate(r)
        p = gp[j]
        poses[i] = JP.Pose{T}(p.energy, Int32(i), p.tx, p.ty, p.tz)
    end
    el = time() - t0
    @printf("\r  %6d / %d rotations   %5.1f s elapsed, %5.1f s remaining",
            bounds[g+1], nrot, el, el * (nrot - bounds[g+1]) / max(1, bounds[g+1]))
    flush(stdout)
end
t_dock = time() - t0
@printf("\n  done in %.1f s (%.2f ms per rotation, %d threads)\n",
        t_dock, 1000 * t_dock / nrot, Threads.nthreads())


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

function pose_report(rank::Integer)
    idx = order[rank]
    p = poses[idx]
    R, t = pose_transform(p, rots.rots, spec)
    pose_coords!(placed, lig.coords, R, t)
    lrms = ligand_rmsd(placed, native; mask = camask)
    lall = ligand_rmsd(placed, native)
    irms = interface_rmsd(rec, lig, native, placed; cutoff = IFACE_CUT)
    return (rank = rank, energy = p.energy, rot = Int(p.rot),
            cells = (minimal_image(p.tx, spec.n[1]), minimal_image(p.ty, spec.n[2]),
                     minimal_image(p.tz, spec.n[3])),
            lrmsd_ca = lrms, lrmsd_all = lall, irmsd = irms)
end

hr("Result")
@printf("hit threshold   %.1f A C-alpha ligand RMSD (PDF p.22)\n", HIT)
@printf("search floor    %.3f A  <- the best this discrete space contains\n", floor_ca.rmsd)
if rank_hit == 0
    @printf("NATIVE POSE     not found: no pose within %.1f A among all %d\n", HIT, nrot)
else
    @printf("NATIVE POSE     rank %d of %d\n", rank_hit, nrot)
end
@printf("best RMSD       %.3f A at rank %d\n", best_rmsd, best_at)

top = pose_report(1)
@printf("\nrank 1          E = %.4f   C-alpha L-RMSD = %.3f A   all-atom = %.3f A   I-RMSD = %.3f A\n",
        top.energy, top.lrmsd_ca, top.lrmsd_all, top.irmsd)
hitrep = rank_hit == 0 ? nothing : pose_report(rank_hit)
if hitrep !== nothing && rank_hit != 1
    @printf("rank %-10d E = %.4f   C-alpha L-RMSD = %.3f A   all-atom = %.3f A   I-RMSD = %.3f A\n",
            rank_hit, hitrep.energy, hitrep.lrmsd_ca, hitrep.lrmsd_all, hitrep.irmsd)
end

println("\ntop 10 by energy:")
println("  rank      energy   L-RMSD(CA)   I-RMSD      cells        rot")
for r in 1:min(10, nrot)
    pr = pose_report(r)
    @printf("  %4d  %10.4f   %8.3f   %8.3f   %-14s %6d%s\n", pr.rank, pr.energy,
            pr.lrmsd_ca, pr.irmsd, string(pr.cells), pr.rot,
            pr.lrmsd_ca <= HIT ? "  <-- hit" : "")
end

nhits_topk = count(<=(HIT), @view rmsds[1:min(TOPK, nrot)])
@printf("\nhits in top %d   %d\n", min(TOPK, nrot), nhits_topk)


nclu = min(NCLUSTER, nrot)
cl_idx = order[1:nclu]
Rs, ts = pose_transforms(view(poses, cl_idx), rots.rots, spec)
S = JP.second_moment(lig.coords)
D = Matrix{T}(undef, nclu, nclu)
pairwise_rmsd!(D, Rs, ts, S, length(lig))
clusters = greedy_cluster(D, CLUSTRAD; maxclusters = 30)
bypop = sortperm(clusters; by = length, rev = true)

hr("Clustering (top $nclu by energy, $(CLUSTRAD) A ligand RMSD radius)")
println("  cluster  members   centre L-RMSD(CA)   centre rank   best member RMSD")


function report_clusters(clusters, bypop, rmsds, hit)
    nc = 0
    for (ci, c) in enumerate(bypop)
        members = clusters[c]
        centre = members[1]
        crms = rmsds[centre]
        bestm = minimum(rmsds[m] for m in members)
        crms <= hit && nc == 0 && (nc = ci)
        ci <= 10 && @printf("  %7d  %7d   %15.3f   %11d   %16.3f%s\n",
                            ci, length(members), crms, centre, bestm,
                            crms <= hit ? "  <-- near-native" : "")
    end
    return nc
end
native_cluster = report_clusters(clusters, bypop, rmsds, HIT)
if native_cluster == 0
    println("  no cluster centre within the hit threshold")
else
    @printf("\n  near-native cluster is rank %d by population (%d members)\n",
            native_cluster, length(clusters[bypop[native_cluster]]))
end


hr("Verdict")
passed = rank_hit == 1
if passed
    @printf("PASS -- the native pose is rank 1 (%.3f A C-alpha, floor %.3f A)\n",
            rmsds[1], floor_ca.rmsd)
elseif rank_hit > 0
    @printf("PARTIAL -- near-native at rank %d, not rank 1 (%.3f A C-alpha)\n",
            rank_hit, rmsds[rank_hit])
else
    @printf("FAIL -- no pose within %.1f A anywhere in the search\n", HIT)
end


println("SUMMARY case=$CASE nrot=$nrot P=$P w1=$W1 charges=$CHARGES gb=$GBMODE " *
        @sprintf("floor=%.3f ", floor_ca.rmsd) *
        "rank=$rank_hit " *
        @sprintf("rank1_lrmsd=%.3f rank1_irmsd=%.3f best_lrmsd=%.3f ",
                 top.lrmsd_ca, top.irmsd, best_rmsd) *
        "best_rank=$best_at hits_in_top$(min(TOPK, nrot))=$nhits_topk " *
        "cluster=$native_cluster " * @sprintf("dock_s=%.1f", t_dock))


_j(::Nothing) = "null"
_j(x::Bool) = x ? "true" : "false"
_j(x::Integer) = string(x)
_j(x::AbstractFloat) = isfinite(x) ? string(x) : "null"
_j(x::AbstractString) = '"' * replace(String(x), '\\' => "\\\\", '"' => "\\\"") * '"'
_j(x::Symbol) = _j(String(x))
_j(x::Union{AbstractVector,Tuple}) = "[" * join((_j(v) for v in x), ",") * "]"
_j(x::NamedTuple) = "{" * join((_j(String(k)) * ":" * _j(v) for (k, v) in pairs(x)), ",") * "}"

if JSONOUT !== nothing
    doc = (
        kind = "validation-bound-bound",
        is_benchmark = false,
        note = "Metric: C-alpha ligand RMSD " *
               "in the fixed receptor frame, PDF p.22.",

        date = Base.Libc.strftime("%Y-%m-%d", time()),
        machine = Sys.CPU_NAME,
        julia = string(VERSION),
        threads = Threads.nthreads(),
        tag = TAG,
        case = CASE,
        structure = basename(COMPLEX),
        rec_chain = REC_CHAIN,
        lig_chain = LIG_CHAIN,
        rec_atoms = length(rec),
        lig_atoms = length(lig),
        lig_ca = count(camask),
        eltype = string(T),
        P = P,
        dars_rank = dars_rank(pot),
        truncation_rel = trunc_rel,
        w1 = w.w1, w2 = w.w2, w3 = w.w3,
        charges = CHARGES,
        rec_qtotal = par_rec.net_charge,
        lig_qtotal = par_lig.net_charge,
        gb_truncation = GBMODE,
        terminal_carboxylate = CTERM_COOH,
        grid_n = collect(spec.n),
        spacing = spec.spacing,
        rec_leak_atoms = fit_rec.leak_total,
        rec_margin_a = fit_rec.margin_angstrom,
        lig_margin_a = fit_lig.margin_angstrom,
        nrot = nrot,
        rot_kind = string(rots.kind),
        batch = BATCH,
        hit_threshold = HIT,
        search_floor_ca = floor_ca.rmsd,
        search_floor_rot = floor_ca.rot,
        search_floor_angle_deg = floor_ca.angle,
        native_rank = rank_hit,
        best_rmsd_ca = best_rmsd,
        best_rmsd_rank = best_at,
        rank1 = (energy = top.energy, lrmsd_ca = top.lrmsd_ca,
                 lrmsd_all = top.lrmsd_all, irmsd = top.irmsd,
                 rot = top.rot, cells = collect(top.cells)),
        hit = hitrep === nothing ? nothing :
              (rank = hitrep.rank, energy = hitrep.energy, lrmsd_ca = hitrep.lrmsd_ca,
               lrmsd_all = hitrep.lrmsd_all, irmsd = hitrep.irmsd,
               rot = hitrep.rot, cells = collect(hitrep.cells)),
        topk = min(TOPK, nrot),
        hits_in_topk = nhits_topk,
        cluster_radius = CLUSTRAD,
        cluster_pool = nclu,
        cluster_sizes = [length(clusters[c]) for c in bypop],
        cluster_centre_rmsd = [rmsds[clusters[c][1]] for c in bypop],
        native_cluster_rank = native_cluster,
        top_energies = [poses[order[r]].energy for r in 1:min(100, nrot)],
        top_rmsds = [rmsds[r] for r in 1:min(100, nrot)],
        wall_setup_s = t_setup,
        wall_dock_s = t_dock,
        passed = passed,
    )
    open(JSONOUT, "w") do io
        write(io, _j(doc))
    end
    println("\nwrote ", JSONOUT)
end

exit(rank_hit == 0 ? 1 : 0)
