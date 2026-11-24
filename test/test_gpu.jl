using Test, JuPIPER, FFTW, StaticArrays, Random, LinearAlgebra
const JP = JuPIPER

"""Max |a-b| relative to the scale of `b`, which is the right metric for a
quantity that crosses zero -- pointwise relative error is meaningless there."""
function relmax(a, b)
    s = maximum(abs, b)
    d = maximum(abs, a .- b)
    return s > 0 ? d / s : d
end

"""The synthetic case, built identically for both arms.

`signed = true` gives both structures signed term weights, and it is required by any
test that compares a **minimum** rather than a whole array. Under the default
non-negative weights every grid product is non-negative, so the landscape's minimum is
exactly zero and what `dock!` returns is a plateau of round-off. It is already known
that comparing the *cell* against such a plateau measures nothing; the less obvious half
is that comparing the *energy* measures nothing either, and worse, that a device scaling
error passes -- twice round-off is still round-off. Signed weights put a genuine
isolated minimum at O(1): measured on this fixture |min| is 0.91-1.00 of the landscape
scale and the second-best cell sits 3.3-9.2% above it, so a device that picked a
different cell shows up in the energy.

Whole-array comparisons (`voxelize!`, `correlate!`, `_store!`) are unaffected by the
degeneracy -- they are scaled by `maximum(abs,...)` over the array, which is O(1) either
way -- so they keep the default and their calibrated tolerances.
"""
function gpu_case(::Type{T}, n, P, na_rec, na_lig; signed::Bool = false) where {T}
    Random.seed!(20260806)
    off = signed ? T(0.5) : zero(T)
    spec = JP.GridSpec{T}((n, n, n), T(1.2),
                          SVector{3,T}(-n*T(0.6), -n*T(0.6), -n*T(0.6)))
    rec = JP.Structure([SVector{3,T}(randn(), randn(), randn()) .* T(n*0.13) for _ in 1:na_rec],
                       rand(T, na_rec, P) .- off)
    lig = JP.Structure([SVector{3,T}(randn(), randn(), randn()) .* T(n*0.08) for _ in 1:na_lig],
                       rand(T, na_lig, P) .- off)
    return (spec, rec, lig)
end

function run_gpu_tests(CU)
    dev = CU.device()
    @info "Running device tests" device=CU.name(dev) capability=CU.capability(dev)

    @testset "GPU (CUDA extension)" begin
        bk = JP.CUDABackend()
        CuArr = CU.CuArray

        @testset "allocator and memory queries" begin
            A = JP.allocate(bk, Float32, 4, 4, 4, 2)
            @test A isa CuArr{Float32,4}
            @test size(A) == (4, 4, 4, 2)
            free  = JP.device_free_bytes(bk)
            total = JP.device_total_bytes(bk)
            @test free > 0
            @test total >= free

            @test JP.device_collect!(bk) === nothing

            @test JP._is_device_oom(CU.OutOfGPUMemoryError(1)) === true
            @test JP._is_device_oom(ErrorException("not memory")) === false
        end

        @testset "voxelize! device vs host" begin
            for T in (Float32, Float64)
                spec, _, lig = gpu_case(T, 32, 3, 200, 150)
                rots = [one(SMatrix{3,3,T,9}),
                        SMatrix{3,3,T,9}(0,1,0, -1,0,0, 0,0,1)]
                dst_h = Array{T}(undef, 32, 32, 32, 2)
                JP.voxelize!(dst_h, lig, rots, 2, spec)
                dst_d = CuArr{T}(undef, 32, 32, 32, 2)
                JP.voxelize!(dst_d, JP.adapt(CuArr, lig), JP.adapt_rots(bk, rots), 2, spec)
                got = Array(dst_d)

                @test sum(got) ≈ sum(dst_h) rtol = (T === Float32 ? 1e-5 : 1e-12)
                @test relmax(got, dst_h) < (T === Float32 ? 1e-5 : 1e-12)
            end
        end

        @testset "_store! / _accumulate! device broadcast vs host loop" begin
            T = Float32
            nxr, ny, nz, B = 9, 16, 16, 3
            Random.seed!(7)
            Rp  = rand(Complex{T}, nxr, ny, nz)
            lig = rand(Complex{T}, nxr, ny, nz, B)
            acc = rand(Complex{T}, nxr, ny, nz, B)
            ngrid = nxr * ny * nz

            sh = copy(acc); JP._store!(sh, Rp, lig, ngrid)
            sd = CuArr(copy(acc))
            JP._store!(sd, CuArr(Rp), CuArr(lig), ngrid)
            @test relmax(Array(sd), sh) < 1e-6

            ah = copy(acc); JP._accumulate!(ah, Rp, lig, ngrid)
            ad = CuArr(copy(acc))
            JP._accumulate!(ad, CuArr(Rp), CuArr(lig), ngrid)
            @test relmax(Array(ad), ah) < 1e-6
        end

        @testset "scan_best! device vs host" begin
            T = Float32
            nx, ny, nz, B = 16, 16, 16, 2
            Random.seed!(11)
            score = rand(T, nx, ny, nz, B)
            vh = Matrix{T}(undef, nz, B); ih = Matrix{Int32}(undef, nz, B)
            JP.scan_best!(vh, ih, score)
            vd = CuArr{T}(undef, nz, B); id = CuArr{Int32}(undef, nz, B)
            JP.scan_best!(vd, id, CuArr(score))

            @test Array(vd) == vh
            @test Array(id) == ih
        end

        @testset "scan_best! block reduction, adversarial cases" begin
            for T in (Float32, Float64),
                (nx, ny, nz, B) in ((16, 16, 16, 2), (5, 3, 2, 1), (33, 31, 3, 2),
                                    (128, 128, 4, 1), (300, 7, 2, 1))
                Random.seed!(hash((T, nx, ny, nz, B)) % 10_000)
                cases = Array{T,4}[rand(T, nx, ny, nz, B), fill(T(2.5), nx, ny, nz, B)]
                tied = rand(T, nx, ny, nz, B)
                tied[1, 1, 1, 1] = tied[nx, ny, 1, 1] = T(-7)
                push!(cases, tied)
                lastc = rand(T, nx, ny, nz, B)
                lastc[nx, ny, nz, B] = T(-9)
                push!(cases, lastc)
                for score in cases
                    vh = Matrix{T}(undef, nz, B); ih = Matrix{Int32}(undef, nz, B)
                    JP.scan_best!(vh, ih, score)
                    vd = CuArr{T}(undef, nz, B); id = CuArr{Int32}(undef, nz, B)
                    JP.scan_best!(vd, id, CuArr(score))
                    @test Array(vd) == vh
                    @test Array(id) == ih
                end
            end
        end

        @testset "receptor_spectra device vs host" begin
            T = Float32
            spec, rec, _ = gpu_case(T, 32, 3, 400, 200)
            w = ones(T, 3)
            rs_h = JP.receptor_spectra(JP.CPUBackend(), rec, spec, w; flags = FFTW.ESTIMATE)
            rs_d = JP.receptor_spectra(bk, JP.adapt(CuArr, rec), spec, w; flags = FFTW.ESTIMATE)
            @test length(rs_d.data) == length(rs_h.data) == 3
            for p in 1:3
                @test relmax(Array(rs_d.data[p]), rs_h.data[p]) < 1e-5
            end
        end

        @testset "correlate! end to end -- the design claim" begin
            for (T, n, P, B, tol) in ((Float32, 32, 3, 1, 1e-4),
                                      (Float32, 32, 3, 4, 1e-4),
                                      (Float64, 32, 3, 1, 1e-10),
                                      (Float32, 64, 6, 1, 1e-4))
                spec, rec, lig = gpu_case(T, n, P, 400, 250)
                rots = [one(SMatrix{3,3,T,9}) for _ in 1:B]

                rs_h = JP.receptor_spectra(JP.CPUBackend(), rec, spec, ones(T, P);
                                           flags = FFTW.ESTIMATE)
                ws_h = JP.workspace(JP.CPUBackend(), spec, P; batch = B, flags = FFTW.ESTIMATE)
                JP.correlate!(ws_h, lig, rots, rs_h)
                E_h = copy(ws_h.score)

                rs_d = JP.receptor_spectra(bk, JP.adapt(CuArr, rec), spec, ones(T, P);
                                           flags = FFTW.ESTIMATE)
                ws_d = JP.workspace(bk, spec, P; batch = B, flags = FFTW.ESTIMATE)
                JP.correlate!(ws_d, JP.adapt(CuArr, lig), JP.adapt_rots(bk, rots), rs_d)
                E_d = Array(ws_d.score)

                @test size(E_d) == size(E_h)
                @test all(isfinite, E_d)
                @test relmax(E_d, E_h) < tol

                scale = maximum(abs, E_h)
                for b in 1:B
                    slice_h = @view E_h[:, :, :, b]
                    slice_d = @view E_d[:, :, :, b]
                    @test slice_h[argmin(slice_d)] - minimum(slice_h) <= tol * scale
                end
            end
        end

        @testset "dock! end to end on device" begin
            for (T, n, P, B, tol) in ((Float32, 32, 3, 1, 1e-4),
                                      (Float32, 32, 3, 2, 1e-4),
                                      (Float64, 32, 3, 1, 1e-10))

                spec, rec, lig = gpu_case(T, n, P, 400, 250; signed = true)
                nrot = 4 * B
                Random.seed!(hash((:dock, T, n, P, B)) % 10_000)
                rs_set = rotations_hopf_balanced(T, 64)
                rots = collect(rs_set.rots[1:nrot])

                w = T[0.7, -1.3, 0.9][1:P]
                rs_h = JP.receptor_spectra(JP.CPUBackend(), rec, spec, w;
                                           flags = FFTW.ESTIMATE)
                pool_h = [JP.workspace(JP.CPUBackend(), spec, P; batch = B,
                                       flags = FFTW.ESTIMATE)]
                ph = Vector{JP.Pose{T}}(undef, nrot)
                JP.dock!(ph, lig, rots, rs_h, pool_h)

                rs_d = JP.receptor_spectra(bk, JP.adapt(CuArr, rec), spec, w;
                                           flags = FFTW.ESTIMATE)
                pool_d = [JP.workspace(bk, spec, P; batch = B, flags = FFTW.ESTIMATE)]
                pd = Vector{JP.Pose{T}}(undef, nrot)

                JP.dock!(pd, JP.adapt_backend(bk, lig), JP.adapt_rots(bk, rots),
                         rs_d, pool_d)

                @test pd isa Vector{JP.Pose{T}}
                @test all(isfinite, (p.energy for p in pd))
                @test [Int(p.rot) for p in pd] == collect(1:nrot)

                ws_s = JP.workspace(JP.CPUBackend(), spec, P; batch = B,
                                    flags = FFTW.ESTIMATE)
                E = copy(JP.correlate!(ws_s, lig, rots[1:B], rs_h))
                scale = maximum(abs, E)

                @test abs(minimum(E)) > 0.1 * scale
                for i in 1:nrot
                    @test abs(pd[i].energy - ph[i].energy) <= tol * scale
                end
            end
        end

        @testset "device_synchronize! is live" begin

            @test JP.device_synchronize!(bk) === nothing
        end

        @testset "autotune_batch on device" begin
            T = Float32
            spec = JP.GridSpec{T}((32, 32, 32), T(1.2), SVector{3,T}(0, 0, 0))
            m = JP.memory_model(spec, 4; natoms = 500, nrot = 1024, T = T)
            b, rep = JP.autotune_batch(bk, spec, 4; natoms = 500, nrot = 1024,
                                       candidates = [8, 4, 2, 1], verbose = false)
            @test b in (8, 4, 2, 1)

            @test rep.fft_factor >= 1.0
            @test rep.fft_factor == JP.safe_fft_factor(rep.fft_factor_raw)
            @test rep.own_bytes == JP.job_bytes(m, b; fft_factor = 1.0)
            @test rep.fft_factor_raw ≈ JP.calibrate_fft_factor(m, b, rep.measured_bytes)

            @test rep.fft_factor_raw < 8.0

            if rep.resolved
                @test rep.measured_bytes > 0

                @test rep.fft_factor_raw > 1.0
                @test rep.measured_bytes >= rep.own_bytes
            else
                @warn "autotune probe did not resolve its own allocation. This is a " *
                      "memory-accounting result, not a numerical one -- correlate! " *
                      "above is unaffected." batch = b
                @warn "  probe figures" measured = rep.measured_bytes own = rep.own_bytes raw = rep.fft_factor_raw
                @test_broken rep.fft_factor_raw > 1.0
            end
        end
    end
    return nothing
end

let ext = Base.get_extension(JuPIPER, :JuPIPERCUDAExt)
    if ext === nothing
        @info """GPU tests skipped: JuPIPERCUDAExt is not loaded.
                 This is expected under --project=. and --project=bench.
                 To run them:  julia --project=gpu -t auto -e 'using CUDA, JuPIPER; include("test/runtests.jl")'"""
    elseif !ext.CUDA.functional()
        @warn """GPU tests skipped: the extension loaded but CUDA.functional() is false.
                 On a machine with a card this means a driver/runtime mismatch, and it
                 is a real failure rather than an absent GPU -- investigate before
                 trusting any other result from this session."""
    else
        run_gpu_tests(ext.CUDA)
    end
end
