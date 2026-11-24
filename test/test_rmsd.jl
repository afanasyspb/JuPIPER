"""
Closed-form pairwise pose RMSD vs the definition.

    RMSD^2(i,j) = ||t_i - t_j||^2 + (2/M)*( tr(S) - tr(R_i' R_j S) )

valid because docking RMSD is computed in the fixed receptor frame with no
re-superposition, and every pose is the same rigid body.
"""

using Test, JuPIPER, StaticArrays, LinearAlgebra, Random
const JP = JuPIPER

randrot(T) = begin
    q = normalize(SVector{4,T}(randn(), randn(), randn(), randn()))
    w, x, y, z = q
    SMatrix{3,3,T,9}(1-2(y^2+z^2), 2(x*y+w*z),   2(x*z-w*y),
                     2(x*y-w*z),   1-2(x^2+z^2), 2(y*z+w*x),
                     2(x*z+w*y),   2(y*z-w*x),   1-2(x^2+y^2))
end

function rmsd_naive!(D, rots, trans, y, M)
    N = length(rots)
    @inbounds for j in 1:N, i in 1:N
        Ri, Rj, ti, tj = rots[i], rots[j], trans[i], trans[j]
        s = zero(eltype(D))
        for k in 1:M
            d = (Ri * y[k] + ti) - (Rj * y[k] + tj)
            s += dot(d, d)
        end
        D[i, j] = sqrt(s / M)
    end
    D
end

@testset "closed-form RMSD == naive" begin
    Random.seed!(31415)
    T = Float64
    for (M, spread, tspread) in ((400, 8.0, 20.0), (2000, 18.0, 60.0), (2000, 18.0, 3.0))
        N = 40
        raw = [SVector{3,T}(randn(), randn(), randn()) .* spread for _ in 1:M]
        c = sum(raw) / M
        y = [p - c for p in raw]
        rots = [randrot(T) for _ in 1:N]
        trans = [SVector{3,T}(randn(), randn(), randn()) .* tspread for _ in 1:N]

        D = Matrix{T}(undef, N, N)
        S = JP.second_moment(y)
        JP.pairwise_rmsd!(D, rots, trans, S, M)
        Dn = rmsd_naive!(Matrix{T}(undef, N, N), rots, trans, y, M)

        bound = max(10 * sqrt(2 * eps(T) * tr(S) / M), 1e-9)
        @test maximum(abs, D .- Dn) < bound
        @test maximum(abs, diag(D)) < bound
        @test maximum(abs, D .- D') < 1e-10
    end
end

@testset "greedy clustering" begin

    T = Float64
    N = 30
    D = fill(T(50), N, N)
    for i in 1:N, j in 1:N
        gi, gj = (i - 1) ÷ 10, (j - 1) ÷ 10
        D[i, j] = gi == gj ? T(2) : T(50)
    end
    for i in 1:N; D[i, i] = 0; end
    cl = JP.greedy_cluster(D, 9.0)
    @test length(cl) == 3
    @test all(c -> length(c) == 10, cl)
    @test sort(vcat(cl...)) == 1:N
end

@testset "RMSD and clustering input dimensions" begin
    rot = one(SMatrix{3,3,Float64,9})
    shift = zero(SVector{3,Float64})
    D = zeros(1, 1)
    @test_throws ArgumentError pairwise_rmsd!(D, [rot], [shift], rot, 0)
    @test_throws DimensionMismatch pairwise_rmsd!(D, [rot], [shift, shift], rot, 1)
    @test_throws DimensionMismatch pairwise_rmsd!(D, [rot], typeof(shift)[], rot, 1)
    @test_throws DimensionMismatch pairwise_rmsd!(zeros(1, 2), [rot], [shift], rot, 1)
    @test_throws DimensionMismatch greedy_cluster(zeros(2, 1), 1.0)
end
