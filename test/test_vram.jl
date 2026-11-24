"""
Device-memory budgeting.

The point of this file is that the batch-sizing arithmetic -- the thing that
actually decides whether a GPU run OOMs -- is validated on hardware we have.
There is no NVIDIA GPU on the development machine, so if this logic lived in the
CUDA extension it could not be tested at all. It lives in `src/vram.jl` instead,
generic over `Backend`, and everything below runs on `CPUBackend`.

What is left untested by construction: the three one-line methods in
`ext/JuPIPERCUDAExt.jl` that map `device_free_bytes` / `device_total_bytes` /
`device_collect!` onto CUDA. That is the intended residue -- three lines, not a
subsystem.
"""

using Test, JuPIPER, StaticArrays, FFTW
const JP = JuPIPER

gspec(T, n) = JP.GridSpec{T}((n, n, n), T(1.2), SVector{3,T}(0, 0, 0))

@testset "byte model agrees with the real workspace" begin

    for T in (Float32, Float64), n in (16, 32, 64), P in (1, 4, 6), B in (1, 2, 5)
        spec = gspec(T, n)
        m = memory_model(spec, P; T = T)
        ws = JP.workspace(JP.CPUBackend(), spec, P; batch = B, flags = FFTW.ESTIMATE)
        scan = m.nz * (m.elsize + 4)
        @test B * (bytes_per_batch(m) - scan) == JP.workspace_bytes(ws)
    end
end

@testset "byte model reproduces the documented 128^3 figure" begin

    m = memory_model(gspec(Float32, 128), 6; T = Float32)
    @test 32.0 < bytes_per_batch(m) / 2^20 < 32.5

    @test bytes_fixed(m) / 2^20 ≈ 6 * 8.125 rtol = 1e-3
end

@testset "fixed vs per-batch split" begin
    m = memory_model(gspec(Float32, 32), 4; natoms = 500, nrot = 1000, T = Float32)

    f = 1.0
    b1 = job_bytes(m, 1; fft_factor = f)
    b2 = job_bytes(m, 2; fft_factor = f)
    b7 = job_bytes(m, 7; fft_factor = f)
    @test b2 - b1 == bytes_per_batch(m)
    @test b7 == bytes_fixed(m) + 7 * bytes_per_batch(m)

    @test job_bytes(m, 4; fft_factor = 2.0) - job_bytes(m, 4; fft_factor = 1.0) ==
          4 * bytes_per_batch(m)
end

@testset "max_batch is sound and monotone" begin
    m = memory_model(gspec(Float32, 128), 6; T = Float32)
    reserve = 512 * 1024^2

    for gb in (2, 4, 8, 16, 24, 40, 80)
        free = gb * 1024^3
        b = max_batch(m, free; reserve = reserve)
        if b > 0
            @test job_bytes(m, b) <= free * 0.90 - reserve

            b < 1024 && @test job_bytes(m, b + 1) > free * 0.90 - reserve
        end
    end

    bs = [max_batch(m, gb * 1024^3) for gb in 1:2:64]
    @test issorted(bs)

    @test max_batch(m, 600 * 1024^2) == 0
    @test max_batch(memory_model(gspec(Float32, 256), 6; T = Float32), 1024^3) == 0

    @test max_batch(m, 16 * 1024^3; headroom = 0.0) >=
          max_batch(m, 16 * 1024^3; headroom = 0.25)
    @test_throws ArgumentError max_batch(m, 16 * 1024^3; headroom = 1.0)
end

@testset "calibrate_fft_factor inverts job_bytes" begin
    m = memory_model(gspec(Float32, 64), 4; natoms = 200, nrot = 500, T = Float32)
    for f in (0.5, 1.0, 1.7, 2.0, 3.25), B in (1, 4, 16)
        measured = bytes_fixed(m) + round(Int, f * bytes_per_batch(m) * B)

        atol = 0.5 / (bytes_per_batch(m) * B)
        @test calibrate_fft_factor(m, B, measured) ≈ f atol = atol
    end
    @test_throws ArgumentError calibrate_fft_factor(m, 0, 10^6)
end

@testset "safe_fft_factor clamps only downward" begin

    @test safe_fft_factor(-0.14048080336015145) == 1.0
    @test safe_fft_factor(0.0) == 1.0
    @test safe_fft_factor(0.999) == 1.0
    @test safe_fft_factor(1.0) == 1.0
    @test safe_fft_factor(1.5) == 1.5
    @test safe_fft_factor(3.25) == 3.25
    @test JP.FFT_FACTOR_FLOOR == 1.0

    m = memory_model(gspec(Float32, 128), 6; T = Float32)
    free = 4 * 1024^3
    @test 0 < max_batch(m, free; fft_factor = 1.0) < 1024
    for raw in (0.25, 0.5, 0.999)
        @test max_batch(m, free; fft_factor = safe_fft_factor(raw)) <
              max_batch(m, free; fft_factor = raw)
    end
    for raw in (1.0, 1.5, 4.0)
        @test max_batch(m, free; fft_factor = safe_fft_factor(raw)) ==
              max_batch(m, free; fft_factor = raw)
    end
end

@testset "backend seams are declared generically, not on a concrete backend" begin

    ext_loaded = Base.get_extension(JuPIPER, :JuPIPERCUDAExt) !== nothing
    for f in (JP.allocate, JP.adapt_rots, JP.adapt_backend, JP.device_free_bytes,
              JP.device_total_bytes, JP.device_collect!, JP.device_synchronize!)

        cuda_methods = filter(methods(f)) do meth
            params = Base.unwrap_unionall(meth.sig).parameters
            any(t -> t === JP.CUDABackend, params)
        end

        @test length(cuda_methods) == (ext_loaded ? 1 : 0)
        ext_loaded && @test all(m -> nameof(m.module) === :JuPIPERCUDAExt, cuda_methods)
    end

    bk = JP.CUDABackend()
    if !ext_loaded
        for call in (() -> JP.allocate(bk, Float32, 2, 2),
                     () -> JP.adapt_rots(bk, [1, 2]),
                     () -> JP.adapt_backend(bk, [1, 2]),
                     () -> JP.device_free_bytes(bk),
                     () -> JP.device_total_bytes(bk),
                     () -> JP.device_collect!(bk),
                     () -> JP.device_synchronize!(bk))
            err = try; call(); nothing; catch e; e; end
            @test err isa ErrorException
            @test occursin("--project=gpu", err.msg)
        end
    end

    @test JP.device_synchronize!(JP.CPUBackend()) === nothing
    @test JP._free_after_sync(JP.CPUBackend()) > 0
end

@testset "the autotune probe allocates the whole job, not just the workspace" begin

    cases = ((32, 4, 8,   500, 1024),
             (64, 6, 4,   3000, 70_000),
             (128, 6, 321, 3000, 70_000))
    for (n, P, B, na, nr) in cases
        m = memory_model(gspec(Float32, n), P; natoms = na, nrot = nr, T = Float32)
        err = bytes_fixed(m) / (bytes_per_batch(m) * B)

        b = min(B, 4)
        ws = JP.workspace(JP.CPUBackend(), gspec(Float32, n), P;
                          batch = b, flags = FFTW.ESTIMATE)
        pad = job_bytes(m, b; fft_factor = 1.0) - JP.workspace_bytes(ws)
        @test pad == bytes_fixed(m) + b * m.nz * (m.elsize + 4)
        @test pad > 0

        if n == 32

            @test 0.13 < err < 0.15
            @test calibrate_fft_factor(m, B, 0) ≈ -0.14048080336015145
            @test safe_fft_factor(calibrate_fft_factor(m, B, 0)) == 1.0
        elseif n == 128
            @test err < 0.006
        end
    end
end

@testset "guard interval is derived, not tuned" begin

    @test guard_interval(1024^3, 0) == typemax(Int)
    @test guard_interval(1024^3, -5) == typemax(Int)
    @test guard_interval(1000, 10) == 100
    @test guard_interval(1024^3, 1024) == 1024^2

    @test guard_interval(1, 10^9) == 1

    for H in (10^4, 10^6, 987654), a in (3, 17, 1024)
        k = guard_interval(H, a)
        @test k * a <= H
        @test (k + 1) * a > H
    end
end

@testset "memory guard fires only under pressure" begin
    bk = JP.CPUBackend()

    g = memory_guard(bk; bytes_per_rotation = 0, low_water = typemax(Int) >> 1)
    @test g.interval == typemax(Int)
    @test !any(_ -> maybe_collect!(g), 1:1000)
    @test g.collections == 0

    g2 = MemoryGuard(bk, 10, 0, 0, 0)
    @test !any(_ -> maybe_collect!(g2), 1:100)
    @test g2.collections == 0

    g3 = MemoryGuard(bk, 10, typemax(Int), 0, 0)
    fired = count(_ -> maybe_collect!(g3), 1:100)
    @test fired == 10
    @test g3.collections == 10

    @test maybe_collect!(nothing) == false
end

@testset "guard costs nothing on the default path" begin

    g = memory_guard(JP.CPUBackend(); bytes_per_rotation = 0)
    f_nothing() = maybe_collect!(nothing)
    f_guard(x) = maybe_collect!(x)
    f_nothing(); f_guard(g)
    @test (@allocated f_nothing()) == 0
    @test (@allocated f_guard(g)) == 0
end

@testset "autotune_batch on the CPU backend" begin

    T = Float32
    spec = gspec(T, 32)
    b, rep = autotune_batch(JP.CPUBackend(), spec, 4;
                            natoms = 100, nrot = 256, flags = FFTW.ESTIMATE,
                            candidates = [8, 4, 2, 1])
    @test b in (8, 4, 2, 1)
    @test rep.predicted_bytes > 0
    @test !isempty(rep.tried)
    @test first(rep.tried)[1] >= b
    @test all(t -> t[2] in (:ok, :oom), rep.tried)

    m = memory_model(spec, 4; natoms = 100, nrot = 256, T = T)
    @test rep.own_bytes == job_bytes(m, b; fft_factor = 1.0)
    @test rep.own_bytes < rep.predicted_bytes
    @test rep.fft_factor >= JP.FFT_FACTOR_FLOOR
    @test rep.fft_factor == safe_fft_factor(rep.fft_factor_raw)
    @test rep.fft_factor_raw ≈ calibrate_fft_factor(m, b, rep.measured_bytes)
    @test rep.resolved == (rep.fft_factor_raw >= JP.FFT_FACTOR_FLOOR)

    @test_throws ErrorException autotune_batch(JP.CPUBackend(), gspec(T, 32), 4;
                                               reserve = typemax(Int) >> 2,
                                               flags = FFTW.ESTIMATE)
end

@testset "descending candidate ladder" begin
    @test JP._descending_batches(1) == [1]
    @test JP._descending_batches(8) == [8, 4, 2, 1]
    @test JP._descending_batches(10) == [10, 8, 4, 2, 1]
    @test issorted(JP._descending_batches(37); rev = true)
    @test last(JP._descending_batches(37)) == 1
end

@testset "memory_report renders" begin
    s = memory_report(JP.CPUBackend(), gspec(Float32, 64), 6, 4; natoms = 100, nrot = 100)
    @test occursin("Device memory budget", s)
    @test occursin("max safe batch", s)

    @test occursin("MEASURED", s)
    @test occursin(string(JP.FFT_FACTOR_MEASURED), s)
end

@testset "dock! rejects a shared guard" begin

    T = Float32
    n, P, B = 16, 3, 1
    spec = gspec(T, n)
    bk = JP.CPUBackend()
    s = JP.Structure([SVector{3,T}(randn(), randn(), randn()) for _ in 1:20], rand(T, 20, P))
    rs = JP.receptor_spectra(bk, s, spec, ones(T, P); flags = FFTW.ESTIMATE)
    pool = [JP.workspace(bk, spec, P; batch = B, flags = FFTW.ESTIMATE) for _ in 1:2]
    rots = [one(SMatrix{3,3,T,9}) for _ in 1:4]
    poses = Vector{JP.Pose{T}}(undef, 4)

    @test_throws ArgumentError dock!(poses, s, rots, rs, pool;
                                     guards = [memory_guard(bk)])

    dock!(poses, s, rots, rs, pool)
    ref = copy(poses)
    fill!(poses, JP.Pose{T}(zero(T), 0, 0, 0, 0))
    dock!(poses, s, rots, rs, pool; guards = [memory_guard(bk) for _ in 1:2])
    @test all(i -> poses[i].energy == ref[i].energy && poses[i].rot == ref[i].rot &&
                   poses[i].tx == ref[i].tx && poses[i].ty == ref[i].ty &&
                   poses[i].tz == ref[i].tz, eachindex(poses))
end
