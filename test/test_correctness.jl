"""
The direct periodic correlation on a small grid checks the sign, conjugation,
and reflection conventions of the FFT implementation (PIPER preprint, p.8).

    E(a,b,c) = sum_p w_p * sum_ijk R_p[i,j,k] * L_p[i+a, j+b, k+c]

with periodic wrap, which is what the FFT computes.
"""

using Test, JuPIPER, StaticArrays, FFTW, LinearAlgebra, Random
const JP = JuPIPER

@testset "FFT correlation == direct O(N^6)" begin
    Random.seed!(1234)
    T = Float64
    n = 8
    P = 2
    spec = JP.GridSpec{T}((n, n, n), T(1.2), SVector{3,T}(-4.8, -4.8, -4.8))
    bk = JP.CPUBackend()

    mk(na) = JP.Structure([SVector{3,T}(randn(), randn(), randn()) .* 1.5 for _ in 1:na],
                          rand(T, na, P) .- 0.5)
    rec, lig = mk(12), mk(7)
    w = T[0.7, -1.3]

    rs = JP.receptor_spectra(bk, rec, spec, w; flags = FFTW.ESTIMATE)
    ws = JP.workspace(bk, spec, P; batch = 1, flags = FFTW.ESTIMATE)

    th = 0.7
    rots = [SMatrix{3,3,T,9}(cos(th), sin(th), 0, -sin(th), cos(th), 0, 0, 0, 1)]
    got = copy(JP.correlate!(ws, lig, rots, rs))

    Rg = Array{T}(undef, n, n, n, 1)
    Lg = Array{T}(undef, n, n, n, 1)
    ident = [one(SMatrix{3,3,T,9})]
    want = zeros(T, n, n, n)
    for p in 1:P
        JP.voxelize!(Rg, rec, ident, p, spec)
        JP.voxelize!(Lg, lig, rots, p, spec)
        for c in 0:n-1, b in 0:n-1, a in 0:n-1
            s = zero(T)
            for k in 0:n-1, j in 0:n-1, i in 0:n-1
                s += Rg[i+1, j+1, k+1, 1] *
                     Lg[mod(i+a, n)+1, mod(j+b, n)+1, mod(k+c, n)+1, 1]
            end
            want[a+1, b+1, c+1] += w[p] * s
        end
    end

    @test maximum(abs, got[:, :, :, 1] .- want) / maximum(abs, want) < 1e-12
end

@testset "translation index convention" begin

    T = Float64
    n = 16
    spec = JP.GridSpec{T}((n, n, n), one(T), SVector{3,T}(0, 0, 0))
    bk = JP.CPUBackend()

    rec = JP.Structure([SVector{3,T}(3, 4, 5)], ones(T, 1, 1))
    lig = JP.Structure([SVector{3,T}(0, 0, 0)], ones(T, 1, 1))
    rs = JP.receptor_spectra(bk, rec, spec, T[-1.0]; flags = FFTW.ESTIMATE)
    ws = JP.workspace(bk, spec, 1; batch = 1, flags = FFTW.ESTIMATE)
    sc = JP.correlate!(ws, lig, [one(SMatrix{3,3,T,9})], rs)
    lin = argmin(vec(sc))
    x, y, z = JP.unravel(spec, lin)

    @test (x, y, z) == (n - 3, n - 4, n - 5)
end

@testset "grid sizing" begin
    @test JP.nextfftsize(127) == 128
    @test JP.nextfftsize(128) == 128
    @test JP.nextfftsize(129) == 130
    @test JP.isfftfriendly(128) && JP.isfftfriendly(120) && JP.isfftfriendly(126)
    @test !JP.isfftfriendly(254)
    @test !JP.isfftfriendly(127)

    for n in 2:2:400
        m = JP.nextfftsize(n)
        @test m >= n && JP.isfftfriendly(m)
    end
end

@testset "DARS potential integrity" begin
    pot = JP.load_dars()
    @test size(pot.eps) == (18, 18)

    @test pot.eps == pot.eps'
    @test length(pot.labels) == 18

    @test issorted(abs.(pot.lambda); rev = true)
    a, r = JP.truncation_error(pot, 18)
    @test a < 1e-10

    i(l) = findfirst(==(l), pot.labels)
    @test pot.eps[i("DOd"), i("DOd")] > 0
    @test pot.eps[i("DOd"), i("RNh")] < 0
    @test pot.eps[i("RNh"), i("RNh")] < 0
    @test pot.eps[i("CSg"), i("CSg")] < -5
end

@testset "Potential element-type conversion narrows, never re-decomposes" begin

    pot = JP.load_dars()
    p32 = JP.Potential{Float32}(pot)
    @test p32 isa JP.Potential{Float32}
    @test p32.labels == pot.labels
    @test p32.eps == Float32.(pot.eps)

    @test p32.lambda == Float32.(pot.lambda)
    @test p32.U == Float32.(pot.U)

    native32 = JP.load_dars(; T = Float32)
    @test native32.lambda != p32.lambda ||
          native32.U != p32.U

    @test JP.dars_rank(p32) == JP.dars_rank(pot)

    @test JP.Potential{Float64}(p32).eps == Float64.(p32.eps)
    @test convert(JP.Potential{Float64}, pot) === pot
    @test convert(JP.Potential{Float32}, pot).lambda == p32.lambda

    @test p32.labels !== pot.labels
end
