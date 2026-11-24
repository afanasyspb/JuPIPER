using Test, JuPIPER, StaticArrays, FFTW, LinearAlgebra, Random
using JuPIPER: GridSpec, Structure, CPUBackend, Workspace, workspace,
               spectra_from_grids, correlate!, receptor_weights


function fibre_case(::Type{T}, n::Int, nterms::Int, natoms::Int;
                    seed = 4242) where {T}
    rng = MersenneTwister(seed)
    spec = GridSpec{T}((n, n, n), T(1.2), SVector{3,T}(-n * 0.6, -n * 0.6, -n * 0.6))
    y = [SVector{3,T}(2randn(rng), 2randn(rng), 2randn(rng)) for _ in 1:natoms]
    y .-= (sum(y) / natoms,)
    W = Matrix{T}(undef, natoms, nterms)
    @views W[:, 1] .= one(T)
    for p in 2:nterms
        @views W[:, p] .= randn(rng, natoms)
    end
    lig = Structure(y, W)
    grids = [T.(randn(rng, n, n, n)) for _ in 1:nterms]

    weights = T[1, 3.5, -2.25, 0.75][1:nterms]
    return (spec, lig, grids, weights)
end

@testset "fibre bound" begin

@testset "sign split reconstructs the energy spectra exactly" begin

    T = Float64
    spec, _, grids, weights = fibre_case(T, 8, 4, 25)
    rs = spectra_from_grids(CPUBackend(), grids, spec, weights; flags = FFTW.ESTIMATE)
    fb = fibre_bound(CPUBackend(), grids, spec, weights; flags = FFTW.ESTIMATE)
    n = length(grids)
    @test fb.nterms == n
    @test length(fb.rs.data) == 2n
    for p in 1:n
        @test fb.rs.data[p] .+ fb.rs.data[n+p] ≈ rs.data[p] atol = 1e-14
    end
end

@testset "envelopes bracket every member of the group" begin

    T = Float64
    spec, lig, grids, weights = fibre_case(T, 10, 3, 30)
    fb = fibre_bound(CPUBackend(), grids, spec, weights; flags = FFTW.ESTIMATE)
    rset = rotations_hopf(T, 3, 6)
    rots = rset.rots[fiber(rset, 2)]
    envelopes!(fb, lig, rots)
    n = fb.nterms
    tmp = Array{T,4}(undef, spec.n..., 1)
    v = reshape(tmp, spec.n...)
    worst_lo = worst_hi = 0.0
    for p in 1:n, r in eachindex(rots)
        JuPIPER.voxelize!(tmp, lig, view(rots, r:r), p, spec)
        @test all(fb.envelopes[p] .<= v .+ 1e-12)
        @test all(v .<= fb.envelopes[n+p] .+ 1e-12)
        worst_lo = max(worst_lo, maximum(fb.envelopes[p] .- v))
        worst_hi = max(worst_hi, maximum(v .- fb.envelopes[n+p]))
    end
    @test worst_lo <= 1e-12
    @test worst_hi <= 1e-12

    @test any(!iszero, fb.envelopes[n+1])
end

@testset "B(a) <= E(a, psi) for every rotation and every translation" begin

    T = Float64
    for (n, nterms, natoms, ndir, nspin) in ((8, 3, 20, 2, 6), (10, 4, 35, 3, 8),
                                             (12, 2, 15, 2, 12))
        spec, lig, grids, weights = fibre_case(T, n, nterms, natoms; seed = n * 7)
        rset = rotations_hopf(T, ndir, nspin)
        rs = spectra_from_grids(CPUBackend(), grids, spec, weights; flags = FFTW.ESTIMATE)
        fb = fibre_bound(CPUBackend(), grids, spec, weights; flags = FFTW.ESTIMATE)
        ws = workspace(CPUBackend(), spec, nterms; batch = 1, flags = FFTW.ESTIMATE)

        for i in 1:ndir
            rots = rset.rots[fiber(rset, i)]
            envelopes!(fb, lig, rots)
            B = copy(bound_landscape!(fb))
            violation = -Inf
            slack = Inf
            for r in eachindex(rots)
                E = correlate!(ws, lig, view(rots, r:r), rs)
                Ev = reshape(E, spec.n...)

                violation = max(violation, maximum(B .- Ev))
                slack = min(slack, minimum(Ev .- B))
            end
            @test violation <= 1e-9
            @test slack >= -1e-9
            @test isfinite(minimum(B))
        end
    end
end

@testset "the bound tracks the group it is given" begin

    T = Float64
    spec, lig, grids, weights = fibre_case(T, 10, 3, 30)
    rset = rotations_hopf(T, 4, 5)
    fb = fibre_bound(CPUBackend(), grids, spec, weights; flags = FFTW.ESTIMATE)
    b1 = fibre_bound_min(fb, lig, rset.rots[fiber(rset, 1)])
    b2 = fibre_bound_min(fb, lig, rset.rots[fiber(rset, 3)])
    @test b1 != b2

    rs = spectra_from_grids(CPUBackend(), grids, spec, weights; flags = FFTW.ESTIMATE)
    ws = workspace(CPUBackend(), spec, 3; batch = 1, flags = FFTW.ESTIMATE)
    single = rset.rots[1:1]
    bs = fibre_bound_min(fb, lig, single)
    E = minimum(correlate!(ws, lig, single, rs))
    @test bs <= E + 1e-9

    @test fibre_bound_min(fb, lig, rset.rots[fiber(rset, 1)]) <= bs + 1e-9
end

@testset "core of revolution" begin

    T = Float64
    spec, lig, _, _ = fibre_case(T, 10, 3, 30)
    rset = rotations_hopf(T, 3, 8)
    rots = rset.rots[fiber(rset, 1)]
    core = Array{T,3}(undef, spec.n...)
    tmp = Array{T,4}(undef, spec.n..., 1)
    fibre_core!(core, tmp, lig, rots, spec)

    ref = fill(T(Inf), spec.n...)
    v = reshape(tmp, spec.n...)
    for r in eachindex(rots)
        JuPIPER.voxelize!(tmp, lig, view(rots, r:r), 1, spec)
        @. ref = min(ref, v)
    end
    @test core == ref
    @test all(core .>= 0)

    @test count(>(1e-12), core) < 0.02 * length(core)
    @test_throws ArgumentError fibre_core!(core, tmp, lig, rots[1:0], spec)
end

end


@testset "pruned driver" begin

@testset "TopK incumbent" begin
    t = JuPIPER.TopK{Float64}(3)
    @test JuPIPER.threshold(t) == Inf
    for v in (5.0, 1.0)
        JuPIPER.push_topk!(t, v)
    end
    @test JuPIPER.threshold(t) == Inf
    JuPIPER.push_topk!(t, 3.0)
    @test JuPIPER.threshold(t) == 5.0
    JuPIPER.push_topk!(t, 2.0)
    @test JuPIPER.threshold(t) == 3.0
    JuPIPER.push_topk!(t, 99.0)
    @test JuPIPER.threshold(t) == 3.0
    @test t.vals[1:3] == [1.0, 2.0, 3.0]

    Random.seed!(31)
    for K in (1, 4, 17)
        xs = randn(200)
        tt = JuPIPER.TopK{Float64}(K)
        for x in xs; JuPIPER.push_topk!(tt, x); end
        @test tt.vals[1:K] == sort(xs)[1:K]
    end
end


function prunable_case(::Type{T}; n = 16, nterms = 2, natoms = 24, seed = 77) where {T}
    rng = MersenneTwister(seed)
    spec = GridSpec{T}((n, n, n), T(1.0), SVector{3,T}(-n / 2, -n / 2, -n / 2))
    y = [SVector{3,T}(1.5randn(rng), 1.5randn(rng), 1.5randn(rng)) for _ in 1:natoms]
    y .-= (sum(y) / natoms,)
    lig = Structure(y, ones(T, natoms, nterms))

    grids = Vector{Array{T,3}}(undef, nterms)
    for p in 1:nterms
        g = Array{T,3}(undef, n, n, n)
        for k in 1:n, j in 1:n, i in 1:n
            d2 = (i - n / 2)^2 + (j - n / 2)^2 + (k - n / 2)^2
            g[i, j, k] = exp(-d2 / (2 * (n / 5)^2)) * (1 + 0.3rand(rng))
        end
        grids[p] = g
    end
    return (spec, lig, grids, ones(T, nterms))
end

@testset "pruning fires, and the answer does not change" begin
    T = Float64
    spec, lig, grids, weights = prunable_case(T)
    rset = rotations_hopf(T, 12, 4)
    rs = spectra_from_grids(CPUBackend(), grids, spec, weights; flags = FFTW.ESTIMATE)
    fb = fibre_bound(CPUBackend(), grids, spec, weights; flags = FFTW.ESTIMATE)
    pool = [workspace(CPUBackend(), spec, length(grids); batch = 1, flags = FFTW.ESTIMATE)]

    ref = Vector{JuPIPER.Pose{T}}(undef, length(rset))
    dock!(ref, lig, rset.rots, rs, pool)

    fired = false
    for K in (1, 5, 20), g in (1, 2, 4)
        got = Vector{JuPIPER.Pose{T}}(undef, length(rset))
        rep = dock_pruned!(got, lig, rset, rs, pool, fb;
                           K = K, group = g, mode = PerCellExact())
        rep.npruned > 0 && (fired = true)

        oref = sortperm(ref; by = p -> (p.energy, p.rot))
        ogot = sortperm(got; by = p -> (p.energy, p.rot))
        for r in 1:K
            a = ref[oref[r]]; b = got[ogot[r]]
            @test a.energy === b.energy
            @test a.rot == b.rot
            @test (a.tx, a.ty, a.tz) == (b.tx, b.ty, b.tz)
        end

        @test count(p -> p.energy == typemax(T), got) == rep.nrot_pruned
        @test rep.nrot == length(rset)
        @test 0 <= prune_rate(rep) <= 1
    end

    @test fired
end

@testset "a group may not straddle two fibres" begin
    T = Float64
    spec, lig, grids, weights = prunable_case(T)
    rset = rotations_hopf(T, 6, 4)
    rs = spectra_from_grids(CPUBackend(), grids, spec, weights; flags = FFTW.ESTIMATE)
    fb = fibre_bound(CPUBackend(), grids, spec, weights; flags = FFTW.ESTIMATE)
    pool = [workspace(CPUBackend(), spec, length(grids); batch = 1, flags = FFTW.ESTIMATE)]
    poses = Vector{JuPIPER.Pose{T}}(undef, length(rset))
    @test_throws ArgumentError dock_pruned!(poses, lig, rset, rs, pool, fb; group = 5)
    @test_throws ArgumentError dock_pruned!(poses, lig, rset, rs, pool, fb; K = 0)

    fset = rotations_fibonacci(T, 24)
    @test_throws ArgumentError dock_pruned!(poses, lig, fset, rs, pool, fb)

    bpool = [workspace(CPUBackend(), spec, length(grids); batch = 2, flags = FFTW.ESTIMATE)]
    @test_throws ArgumentError dock_pruned!(poses, lig, rset, rs, bpool, fb)
end

@testset "with an unreachable threshold the run is exhaustive and identical" begin

    T = Float64
    spec, lig, grids, weights = prunable_case(T; seed = 5)
    rset = rotations_hopf(T, 8, 3)
    rs = spectra_from_grids(CPUBackend(), grids, spec, weights; flags = FFTW.ESTIMATE)
    fb = fibre_bound(CPUBackend(), grids, spec, weights; flags = FFTW.ESTIMATE)
    pool = [workspace(CPUBackend(), spec, length(grids); batch = 1, flags = FFTW.ESTIMATE)]
    ref = Vector{JuPIPER.Pose{T}}(undef, length(rset))
    dock!(ref, lig, rset.rots, rs, pool)
    got = Vector{JuPIPER.Pose{T}}(undef, length(rset))
    rep = dock_pruned!(got, lig, rset, rs, pool, fb; K = 10^6)
    @test rep.npruned == 0
    @test rep.nrot_pruned == 0
    for r in eachindex(ref)
        @test ref[r].energy === got[r].energy
        @test (ref[r].rot, ref[r].tx, ref[r].ty, ref[r].tz) ===
              (got[r].rot, got[r].tx, got[r].ty, got[r].tz)
    end
end

end
