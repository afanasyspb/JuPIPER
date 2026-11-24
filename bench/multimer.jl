#!/usr/bin/env julia


using JuPIPER, BioStructures, StaticArrays, FFTW, LinearAlgebra, Printf

const JP = JuPIPER
const _ROOT = normpath(joinpath(@__DIR__, ".."))


function argval(flag::AbstractString, default)
    i = findfirst(==(flag), ARGS)
    i === nothing && return default
    i == length(ARGS) && error("$flag needs a value")
    return ARGS[i+1]
end
argflag(flag::AbstractString) = findfirst(==(flag), ARGS) !== nothing
argint(flag, d) = (v = argval(flag, nothing); v === nothing ? d : parse(Int, v))
argnum(flag, d) = (v = argval(flag, nothing); v === nothing ? d : parse(Float64, v))

const T           = Float64
const CASE        = argval("--case", "4XAU")
const STRUCT      = argval("--structure",
                           joinpath(_ROOT, "data", "structures", CASE * ".pdb"))
const CHAIN_A     = argval("--chain-a", "A")
const CHAIN_B     = argval("--chain-b", "B")
const ORDER       = argint("--order", 2)
const NROT        = argint("--nrot", 70_000)
const TOL         = argnum("--tol", JP.CLUSPRO_SYM_TOL)
const SPACING     = argnum("--spacing", JP.PIPER_SPACING)
const PAD         = argnum("--pad", 2.0)
const W1          = argnum("--w1", 7.0)
const HIT         = argnum("--hit", 10.0)
const TOPK        = argint("--topk", 100)
const PARG        = argint("--P", 0)
const NOFULL      = argflag("--no-full")
const TAG         = argval("--tag", "")
const JSONOUT     = argval("--json", nothing)

scrub(s::AbstractString) = begin
    t = replace(String(s), '\\' => '/')
    r = replace(_ROOT, '\\' => '/'); endswith(r, '/') || (r *= "/")
    replace(t, r => "")
end

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

hr(title) = (println(); println(title); println(repeat("-", length(title))))


"""
    match_ca(a, b) -> (ia, ib)

Indices of C-alpha atoms present in both chains, aligned on `(resnum, inscode)`
and returned in ascending residue order.

Two chains of one deposition are *not* guaranteed to model the same residues --
4XAU's A and B differ by six -- so a symmetry fit that zipped them positionally
would superimpose different residues and report a defect that is a bookkeeping
error wearing the clothes of a crystallographic one.
"""
function match_ca(a::JP.AtomTable, b::JP.AtomTable)
    key(at, i) = (at.resnum[i], at.inscode[i])
    da = Dict{Tuple{Int32,Char},Int}()
    for i in 1:length(a)
        (a.atomname[i] == "CA" && a.element[i] == "C") && (da[key(a, i)] = i)
    end
    db = Dict{Tuple{Int32,Char},Int}()
    for i in 1:length(b)
        (b.atomname[i] == "CA" && b.element[i] == "C") && (db[key(b, i)] = i)
    end
    ks = sort!(collect(intersect(keys(da), keys(db))))
    return ([da[k] for k in ks], [db[k] for k in ks])
end

hr("Cn symmetric docking vs unconstrained 6D")
@printf("case            %s chains %s / %s   (C%d)\n", CASE, CHAIN_A, CHAIN_B, ORDER)
@printf("threads         %d\n", Threads.nthreads())
isfile(STRUCT) || error("""
    structure not found: $(scrub(STRUCT))
    fetch it first:  bash scripts/fetch_structures.sh $CASE""")

A = read_atoms(STRUCT; T = T, chains = CHAIN_A)
B = read_atoms(STRUCT; T = T, chains = CHAIN_B)
println("subunit A       ", A.report)
println("subunit B       ", B.report)

ia, ib = match_ca(A, B)
length(ia) >= 3 || error("only $(length(ia)) matched C-alpha pairs -- the two " *
                         "chains do not model the same residues")


shift  = B.centroid - A.centroid
nativeB = [B.coords[j] + shift for j in ib]
Pa = [A.coords[i] for i in ia]

fit = symmetry_fit(Pa, nativeB)
hr("Deposited assembly (before any docking)")
@printf("matched C-alpha %d of %d / %d\n", length(ia),
        count(i -> A.atomname[i] == "CA" && A.element[i] == "C", 1:length(A)),
        count(i -> B.atomname[i] == "CA" && B.element[i] == "C", 1:length(B)))
println("symmetry fit    ", fit)
if fit.order != ORDER
    @warn """the deposited A->B operation is C$(fit.order), not the C$(ORDER) being
             searched. The ranks below are then measured against a target the
             restricted space does not contain."""
end
@printf("rise            %+.4f A   <-- a closed C%d ring has rise 0 (file header)\n",
        fit.rise, ORDER)
@printf("angular defect  %.4f deg from exactly %.1f\n", fit.defect, 360 / ORDER)


camask = falses(length(A))
native = copy(A.coords)
for (k, i) in enumerate(ia)
    camask[i] = true
    native[i] = nativeB[k]
end


tbl = load_acp_types()
pot = load_dars()
P   = PARG > 0 ? PARG : dars_rank(pot)
w   = PiperWeights{T}(T(W1), T(PIPER_WEIGHTS_ENZYME_INHIBITOR.w2),
                      T(PIPER_WEIGHTS_ENZYME_INHIBITOR.w3))
par = parametrize(tbl, A)


spec = gridspec_for(A, A; spacing = SPACING, pad = PAD)
hr("Scoring function and grid")
@printf("P               %d  (%d terms, %d transforms per rotation)\n", P, 2 + P, 3 + P)
@printf("weights         w1 = %.4g  w2 = %.4g  w3 = %.4g\n",
        w.w1, w.w2, w.w3)
@printf("grid            %s at %.2f A = %d cells (%.1f A per side)\n",
        string(spec.n), spec.spacing, prod(spec.n), spec.n[1] * spec.spacing)
println("receptor fit    ", grid_fit(spec, A.coords))

FFTW.set_num_threads(1)
BLAS.set_num_threads(1)

rots = rotations_hopf_balanced(T, NROT)
sym  = cyclic_symmetry(ORDER, rots; tol = TOL)
nfull = length(rots)
naxes = length(sym)

hr("Search space")
@printf("unconstrained   %d rotations (%d directions x %d spins) x %d cells\n",
        nfull, rots.ndir, rots.nspin, prod(spec.n))
@printf("C%d symmetric    %d axes (the SAME S^2 base) x the %.1f A plane slab\n",
        ORDER, naxes, TOL)
@printf("rotation ratio  %.1fx   <-- transforms; this is the whole speedup\n",
        nfull / naxes)
@printf("tolerance floor %.2f A (sym_tol_floor at %.2f A spacing); running %.2f A\n",
        sym_tol_floor(spec.spacing), spec.spacing, TOL)

t_setup = @elapsed begin
    rs   = piper_receptor_spectra(JP.CPUBackend(), A, par, pot, spec;
                                  P = P, w = w, flags = FFTW.MEASURE)
    ligS = piper_structure(A, par, pot; P = P)
    pool = [workspace(JP.CPUBackend(), spec, 2 + P; batch = 1, flags = FFTW.MEASURE)
            for _ in 1:Threads.nthreads()]
end
@printf("setup           %.1f s (%d workspaces, %.1f MiB each)\n",
        t_setup, length(pool), JP.workspace_bytes(first(pool)) / 2^20)


pc = [length(plane_cells(sym.axes[i], spec, TOL)) for i in 1:min(naxes, 256)]
@printf("plane slab      %.3f%% of cells on average (%.3f-%.3f%%), %.0f cells\n",
        100 * sum(pc) / (length(pc) * prod(spec.n)),
        100 * minimum(pc) / prod(spec.n), 100 * maximum(pc) / prod(spec.n),
        sum(pc) / length(pc))


hr("Search floors (no energy consulted)")
t_symfloor = @elapsed sfloor = symmetric_search_floor(sym, spec, A.coords, native;
                                                      mask = camask)
@printf("C%d restricted    %.3f A C-alpha  (axis %d, cells %s) [%.1f s]\n",
        ORDER, sfloor.rmsd, sfloor.rot, string(sfloor.cells), t_symfloor)
t_ufloor = @elapsed ufloor = search_floor(rots, spec, A.coords, native; mask = camask)
@printf("unconstrained   %.3f A C-alpha  (rotation %d, %.2f deg off) [%.1f s]\n",
        ufloor.rmsd, ufloor.rot, ufloor.angle, t_ufloor)
@printf("subunit RMSD    %.3f A  <-- the conformational floor neither can beat\n",
        fit.rmsd)


hr("Plane tolerance sweep (restricted floor vs tol)")
println("   tol (A)   floor (A)   mean cells   % of grid")
tolsweep = Any[]
for tol in (0.3, 0.6, 1.2, 2.0, 2.4, 4.0)
    s2 = cyclic_symmetry(ORDER, sym.axes; tol = tol)
    f2 = symmetric_search_floor(s2, spec, A.coords, native; mask = camask)
    cc = sum(length(plane_cells(s2.axes[i], spec, tol)) for i in 1:min(naxes, 128)) /
         min(naxes, 128)
    @printf("   %7.2f   %9.3f   %10.0f   %8.3f%%\n", tol, f2.rmsd, cc,
            100 * cc / prod(spec.n))
    push!(tolsweep, (tol = tol, floor_ca = f2.rmsd, mean_cells = cc,
                     frac = cc / prod(spec.n)))
end


placed = Vector{SVector{3,T}}(undef, length(A))

"""Rank the native placement in an energy-ordered pose list, and the best RMSD."""
function score_arm(poses, rotlist, spec, y, native, camask, hit, topk)
    order = sortperm(poses; by = p -> p.energy)
    n = length(order)
    rmsds = Vector{T}(undef, n)
    pl = Vector{SVector{3,T}}(undef, length(y))
    for (rank, idx) in enumerate(order)
        R, t = pose_transform(poses[idx], rotlist, spec)
        pose_coords!(pl, y, R, t)
        rmsds[rank] = ligand_rmsd(pl, native; mask = camask)
    end
    best, bestat = findmin(rmsds)
    k = min(topk, n)
    return (order = order, rmsds = rmsds, rank = hit_rank(rmsds, hit),
            best = best, best_rank = bestat,
            hits_topk = count(<=(hit), @view rmsds[1:k]), topk = k,
            e1 = poses[order[1]].energy, r1 = rmsds[1])
end

hr("Arm 1 -- C$(ORDER) symmetric")
sposes = Vector{JP.Pose{T}}(undef, naxes)


dock_symmetric!(view(sposes, 1:1), ligS,
                cyclic_symmetry(ORDER, sym.axes[1:1]; tol = TOL), rs, pool[1:1])
t_sym = @elapsed (_, srep) = dock_symmetric!(sposes, ligS, sym, rs, pool;
                                             nrot_full = nfull)
@printf("wall            %.2f s  (%.3f ms per axis)\n", t_sym, 1000 * t_sym / naxes)
println("report          ", srep)
ssc = score_arm(sposes, sym.rots, spec, A.coords, native, camask, HIT, TOPK)
@printf("native rank     %s of %d\n", ssc.rank == 0 ? "not found" : string(ssc.rank), naxes)
@printf("best RMSD       %.3f A at rank %d   (restricted floor %.3f A)\n",
        ssc.best, ssc.best_rank, sfloor.rmsd)
@printf("rank 1          E = %.4f, %.3f A\n", ssc.e1, ssc.r1)
println("\ntop 10 by energy:")
println("   rank      energy   L-RMSD(CA)   axis")
for r in 1:min(10, naxes)
    idx = ssc.order[r]
    @printf("   %4d  %10.4f   %9.3f   %5d%s\n", r, sposes[idx].energy, ssc.rmsds[r],
            Int(sposes[idx].rot), ssc.rmsds[r] <= HIT ? "  <-- hit" : "")
end


t_full = NaN; fsc = nothing
if !NOFULL
    hr("Arm 2 -- unconstrained 6D")
    fposes = Vector{JP.Pose{T}}(undef, nfull)
    t0 = time()
    ngroups = max(1, min(20, cld(nfull, 1000)))
    bounds = unique!([round(Int, b) for b in range(0, nfull; length = ngroups + 1)])
    for g in 1:(length(bounds) - 1)
        r = (bounds[g]+1):bounds[g+1]
        isempty(r) && continue
        gp = Vector{JP.Pose{T}}(undef, length(r))
        dock!(gp, ligS, view(rots.rots, r), rs, pool)
        @inbounds for (j, i) in enumerate(r)
            p = gp[j]
            fposes[i] = JP.Pose{T}(p.energy, Int32(i), p.tx, p.ty, p.tz)
        end
        el = time() - t0
        @printf("\r  %6d / %d rotations   %6.1f s elapsed, %6.1f s remaining",
                bounds[g+1], nfull, el, el * (nfull - bounds[g+1]) / max(1, bounds[g+1]))
        flush(stdout)
    end
    t_full = time() - t0
    @printf("\nwall            %.2f s  (%.3f ms per rotation)\n",
            t_full, 1000 * t_full / nfull)
    global fsc = score_arm(fposes, rots.rots, spec, A.coords, native, camask, HIT, TOPK)
    @printf("native rank     %s of %d\n",
            fsc.rank == 0 ? "not found" : string(fsc.rank), nfull)
    @printf("best RMSD       %.3f A at rank %d   (unconstrained floor %.3f A)\n",
            fsc.best, fsc.best_rank, ufloor.rmsd)
    @printf("rank 1          E = %.4f, %.3f A\n", fsc.e1, fsc.r1)
end


hr("Verdict")
@printf("rotations       %d -> %d = %.1fx fewer FFT schedules\n",
        nfull, naxes, rotation_reduction(srep))
@printf("translations    %d -> %.0f = %.1fx fewer cells scanned, worth NO transform work\n",
        prod(spec.n), srep.plane_total / naxes, translation_reduction(srep))
if !NOFULL
    @printf("wall clock      %.2f s -> %.2f s = %.1fx  (same process, same grid, same pool)\n",
            t_full, t_sym, t_full / t_sym)
    @printf("per rotation    %.3f -> %.3f ms  (%.2fx -- the residue is the scan)\n",
            1000 * t_full / nfull, 1000 * t_sym / naxes,
            (t_full / nfull) / (t_sym / naxes))
end
ok_sym = ssc.rank == 1
@printf("\nC%d native rank   %s\n", ORDER,
        ssc.rank == 0 ? "NOT FOUND" : string(ssc.rank) * " of " * string(naxes))
println(ok_sym ? "VERDICT: exit criterion MET -- native interface at rank 1 of the restricted space." :
        ssc.rank == 0 ?
        "VERDICT: exit criterion NOT met -- no near-native pose in the restricted space." :
        "VERDICT: exit criterion PARTIALLY met -- near-native found, not at rank 1.")

if JSONOUT !== nothing
    doc = (kind = "multimer",
           note = "Symmetric vs unconstrained docking, both arms in one process. " *
                  "The rotation reduction removes transform work; the translation " *
                  "reduction removes none.",
           date = Base.Libc.strftime("%Y-%m-%d", time()),
           tag = TAG, machine = Sys.CPU_NAME, julia = string(VERSION),
           threads = Threads.nthreads(), arm = "cpu",
           case = CASE, structure = basename(STRUCT),
           chain_a = CHAIN_A, chain_b = CHAIN_B, order = ORDER,
           matched_ca = length(ia), rec_atoms = length(A), lig_atoms = length(A),
           deposited = (order = fit.order, angle_deg = fit.angle,
                        defect_deg = fit.defect, rise_a = fit.rise,
                        subunit_rmsd = fit.rmsd,
                        axis = (fit.axis[1], fit.axis[2], fit.axis[3])),
           eltype = string(T), P = P, nterms = 2 + P,
           w1 = w.w1, w2 = w.w2, w3 = w.w3,
           w1_status = "ASSUMED), not the published 2.1",
           grid_n = collect(spec.n), spacing = spec.spacing, ncells = prod(spec.n),
           nrot_full = nfull, naxes = naxes,
           rot_kind = string(rots.kind), ndir = rots.ndir, nspin = rots.nspin,
           tol = TOL, tol_floor = sym_tol_floor(spec.spacing),
           tol_source = "2017 ClusPro protocol p.9, published at 2 A on a 1.0 A grid",
           plane_cells_mean = srep.plane_total / naxes,
           rotation_reduction = rotation_reduction(srep),
           translation_reduction = translation_reduction(srep),
           translation_reduction_is_transform_work = false,
           search_floor_sym = sfloor.rmsd, search_floor_full = ufloor.rmsd,
           tol_sweep = tolsweep,
           t_setup_s = t_setup, t_sym_s = t_sym, t_full_s = t_full,
           t_cells_s = srep.t_cells,
           ms_per_axis = 1000 * t_sym / naxes,
           ms_per_rot_full = NOFULL ? nothing : 1000 * t_full / nfull,
           wall_speedup = NOFULL ? nothing : t_full / t_sym,
           hit_threshold = HIT,
           sym = (native_rank = ssc.rank, best_rmsd = ssc.best,
                  best_rank = ssc.best_rank, hits_in_topk = ssc.hits_topk,
                  topk = ssc.topk, rank1_energy = ssc.e1, rank1_rmsd = ssc.r1),
           full = fsc === nothing ? nothing :
                  (native_rank = fsc.rank, best_rmsd = fsc.best,
                   best_rank = fsc.best_rank, hits_in_topk = fsc.hits_topk,
                   topk = fsc.topk, rank1_energy = fsc.e1, rank1_rmsd = fsc.r1),
           passed = ok_sym)
    open(JSONOUT, "w") do io
        write(io, _j(doc))
    end
    println("\nwrote ", scrub(JSONOUT))
end
