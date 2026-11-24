"""
End-to-end driver: `dock!` under threads.

The point of these tests is not throughput, it is that the threaded driver
produces *exactly* the same answer as a single-threaded run. A workspace shared
between two live tasks corrupts results silently, which is why `dock!` asserts
its chunking invariant and why this comparison exists.
"""

using Test, JuPIPER, StaticArrays, FFTW, Random, LinearAlgebra
const JP = JuPIPER

randrot_drv(T) = begin
    q = normalize(SVector{4,T}(randn(), randn(), randn(), randn()))
    w, x, y, z = q
    SMatrix{3,3,T,9}(1-2(y^2+z^2), 2(x*y+w*z),   2(x*z-w*y),
                     2(x*y-w*z),   1-2(x^2+z^2), 2(y*z+w*x),
                     2(x*z+w*y),   2(y*z-w*x),   1-2(x^2+y^2))
end

@testset "dock! threaded == serial" begin
    Random.seed!(2718)
    T = Float32
    n = 24
    P = 3
    nrot = 24
    spec = JP.GridSpec{T}((n, n, n), T(1.2), SVector{3,T}(-14.4, -14.4, -14.4))
    bk = JP.CPUBackend()
    rec = JP.Structure([SVector{3,T}(randn(), randn(), randn()) .* T(4) for _ in 1:600],
                       rand(T, 600, P))
    lig = JP.Structure([SVector{3,T}(randn(), randn(), randn()) .* T(2.5) for _ in 1:250],
                       rand(T, 250, P))
    rs = JP.receptor_spectra(bk, rec, spec, ones(T, P); flags = FFTW.ESTIMATE)
    rots = [randrot_drv(T) for _ in 1:nrot]

    serial = Vector{JP.Pose{T}}(undef, nrot)
    JP.dock!(serial, lig, rots, rs,
             [JP.workspace(bk, spec, P; batch = 1, flags = FFTW.ESTIMATE)])

    for nw in (2, 3, 4)
        par = Vector{JP.Pose{T}}(undef, nrot)
        pool = [JP.workspace(bk, spec, P; batch = 1, flags = FFTW.ESTIMATE) for _ in 1:nw]
        JP.dock!(par, lig, rots, rs, pool)
        @test [p.energy for p in par] == [p.energy for p in serial]
        @test [(p.tx, p.ty, p.tz) for p in par] == [(p.tx, p.ty, p.tz) for p in serial]
    end
end

@testset "dock! batching agrees with batch=1" begin
    Random.seed!(161)
    T = Float32
    n = 16
    P = 2
    nrot = 12
    spec = JP.GridSpec{T}((n, n, n), T(1.2), SVector{3,T}(-9.6, -9.6, -9.6))
    bk = JP.CPUBackend()
    rec = JP.Structure([SVector{3,T}(randn(), randn(), randn()) .* T(3) for _ in 1:300],
                       rand(T, 300, P))
    lig = JP.Structure([SVector{3,T}(randn(), randn(), randn()) .* T(2) for _ in 1:120],
                       rand(T, 120, P))
    rs = JP.receptor_spectra(bk, rec, spec, ones(T, P); flags = FFTW.ESTIMATE)
    rots = [randrot_drv(T) for _ in 1:nrot]

    ref = Vector{JP.Pose{T}}(undef, nrot)
    JP.dock!(ref, lig, rots, rs, [JP.workspace(bk, spec, P; batch = 1, flags = FFTW.ESTIMATE)])

    for B in (2, 3, 4, 6)
        got = Vector{JP.Pose{T}}(undef, nrot)
        JP.dock!(got, lig, rots, rs,
                 [JP.workspace(bk, spec, P; batch = B, flags = FFTW.ESTIMATE)])

        @test all(isapprox(a.energy, b.energy; rtol = 1e-4) for (a, b) in zip(got, ref))
        @test [(p.tx, p.ty, p.tz) for p in got] == [(p.tx, p.ty, p.tz) for p in ref]
    end
end

@testset "dock! refuses a ragged batch rather than dropping rotations" begin
    T = Float32
    n = 16
    spec = JP.GridSpec{T}((n, n, n), T(1.2), SVector{3,T}(-9.6, -9.6, -9.6))
    bk = JP.CPUBackend()
    s = JP.Structure([SVector{3,T}(randn(), randn(), randn()) for _ in 1:10], rand(T, 10, 1))
    rs = JP.receptor_spectra(bk, s, spec, ones(T, 1); flags = FFTW.ESTIMATE)
    pool = [JP.workspace(bk, spec, 1; batch = 4, flags = FFTW.ESTIMATE)]
    rots = [one(SMatrix{3,3,T,9}) for _ in 1:7]
    poses = Vector{JP.Pose{T}}(undef, 7)
    @test_throws ArgumentError JP.dock!(poses, s, rots, rs, pool)
end
