"""
CPU fast path == KernelAbstractions path.

Two implementations of the same kernel exist for performance reasons. They must not
drift. They already drifted once: the CPU path dropped atoms whose trilinear stencil
straddled the grid boundary while the KA path clipped them, a 0.44 discrepancy in total
grid mass that no other test would have caught.

When a CUDA GPU is available this file should additionally compare the CuArray
path against the CPU path.
"""

using Test, JuPIPER, StaticArrays, Random
const JP = JuPIPER

"Force the generic (KernelAbstractions) method even for an `Array` argument."
function voxelize_ka!(dst::Array{T,4}, s, rots, term, spec) where {T}
    invoke(JP.voxelize!,
           Tuple{AbstractArray{T,4},JP.Structure,AbstractVector,Integer,
                 JP.GridSpec{T},JP.Trilinear},
           dst, s, rots, term, spec, JP.Trilinear())
end

@testset "voxelize!: Array path == KernelAbstractions path" begin
    Random.seed!(99)
    T = Float64
    n = 16
    P = 3
    spec = JP.GridSpec{T}((n, n, n), T(1.2), SVector{3,T}(-9.6, -9.6, -9.6))

    for spread in (1.0, 4.0, 7.0, 12.0)
        s = JP.Structure(
            [SVector{3,T}(randn(), randn(), randn()) .* spread for _ in 1:400],
            rand(T, 400, P))
        for th in (0.0, 0.9, 2.4)
            rots = [SMatrix{3,3,T,9}(cos(th), sin(th), 0, -sin(th), cos(th), 0, 0, 0, 1)]
            a = Array{T}(undef, n, n, n, 1)
            b = Array{T}(undef, n, n, n, 1)
            JP.voxelize!(a, s, rots, 2, spec)
            voxelize_ka!(b, s, rots, 2, spec)

            @test maximum(abs, a .- b) < 1e-13
            @test isapprox(sum(a), sum(b); rtol = 1e-13)
        end
    end
end

@testset "voxelize! conserves deposited mass when fully inside" begin
    T = Float64
    n = 32
    spec = JP.GridSpec{T}((n, n, n), T(1.0), SVector{3,T}(-16, -16, -16))

    w = rand(T, 200, 1)
    s = JP.Structure([SVector{3,T}(randn(), randn(), randn()) .* 3 for _ in 1:200], w)
    dst = Array{T}(undef, n, n, n, 1)
    JP.voxelize!(dst, s, [one(SMatrix{3,3,T,9})], 1, spec)
    @test isapprox(sum(dst), sum(w); rtol = 1e-12)
end

@testset "_accumulate!: SIMD path == broadcast path" begin
    Random.seed!(4242)
    for T in (Float32, Float64), (nxr, ny, nz, B) in ((9, 16, 16, 1), (17, 32, 32, 3))
        ngrid = nxr * ny * nz
        Rp = rand(Complex{T}, nxr, ny, nz)
        L  = rand(Complex{T}, nxr, ny, nz, B)
        a0 = rand(Complex{T}, nxr, ny, nz, B)

        fast = copy(a0)
        JP._accumulate!(fast, Rp, L, ngrid)

        slow = copy(a0)
        invoke(JP._accumulate!,
               Tuple{AbstractArray{Complex{T},4},AbstractArray{Complex{T},3},
                     AbstractArray{Complex{T},4},Int},
               slow, Rp, L, ngrid)

        @test maximum(abs, fast .- slow) / maximum(abs, slow) < 8 * eps(T)

        want = copy(a0)
        for b in 1:B
            want[:, :, :, b] .+= Rp .* L[:, :, :, b]
        end
        @test maximum(abs, fast .- want) / maximum(abs, want) < 8 * eps(T)

        sfast = copy(a0)
        JP._store!(sfast, Rp, L, ngrid)
        sslow = copy(a0)
        invoke(JP._store!,
               Tuple{AbstractArray{Complex{T},4},AbstractArray{Complex{T},3},
                     AbstractArray{Complex{T},4},Int},
               sslow, Rp, L, ngrid)
        swant = similar(a0)
        for b in 1:B
            swant[:, :, :, b] .= Rp .* L[:, :, :, b]
        end
        @test maximum(abs, sfast .- swant) / maximum(abs, swant) < 8 * eps(T)
        @test maximum(abs, sslow .- swant) / maximum(abs, swant) < 8 * eps(T)

        chain = similar(a0)
        JP._store!(chain, Rp, L, ngrid)
        JP._accumulate!(chain, Rp, L, ngrid)
        @test maximum(abs, chain .- 2 .* swant) / maximum(abs, swant) < 8 * eps(T)
    end
end

@testset "scan_best!: Array path == KernelAbstractions path" begin
    Random.seed!(5)
    T = Float32
    nx, ny, nz, B = 12, 10, 8, 3
    score = rand(T, nx, ny, nz, B)
    v1 = Matrix{T}(undef, nz, B); i1 = Matrix{Int32}(undef, nz, B)
    v2 = Matrix{T}(undef, nz, B); i2 = Matrix{Int32}(undef, nz, B)
    JP.scan_best!(v1, i1, score)
    invoke(JP.scan_best!,
           Tuple{AbstractMatrix{T},AbstractMatrix{<:Integer},AbstractArray{T,4}},
           v2, i2, score)
    @test v1 == v2
    @test i1 == i2

    for b in 1:B, k in 1:nz
        @test score[i1[k, b] + nx * ny * nz * (b - 1)] == v1[k, b]
        @test v1[k, b] == minimum(@view score[:, :, k, b])
    end
end


@testset "scan_best!: block reduction == serial, adversarially" begin

    sizes = [(12, 10, 8, 3), (16, 16, 16, 2), (5, 3, 2, 1), (1, 1, 3, 1),
             (64, 64, 4, 2), (300, 7, 2, 1), (33, 31, 3, 2), (128, 128, 2, 1)]
    for T in (Float32, Float64), (nx, ny, nz, B) in sizes
        Random.seed!(hash((T, nx, ny, nz, B)) % 10_000)
        nxy = nx * ny
        landscapes = Dict{String,Array{T,4}}()
        landscapes["random"] = rand(T, nx, ny, nz, B)

        landscapes["constant"] = fill(T(2.5), nx, ny, nz, B)

        tied = rand(T, nx, ny, nz, B)
        tied[1, 1, 1, 1] = tied[nx, ny, 1, 1] = T(-7)
        landscapes["tied ends"] = tied

        lastc = rand(T, nx, ny, nz, B)
        lastc[nx, ny, nz, B] = T(-9)
        landscapes["min in last cell"] = lastc

        landscapes["all NaN"] = fill(T(NaN), nx, ny, nz, B)

        for (name, score) in landscapes
            v1 = Matrix{T}(undef, nz, B); i1 = Matrix{Int32}(undef, nz, B)
            v2 = Matrix{T}(undef, nz, B); i2 = Matrix{Int32}(undef, nz, B)
            JP.scan_best!(v1, i1, score)
            invoke(JP.scan_best!,
                   Tuple{AbstractMatrix{T},AbstractMatrix{<:Integer},AbstractArray{T,4}},
                   v2, i2, score)
            @test v1 == v2
            @test i1 == i2
            if name == "constant"
                @test all(i1[k, b] == nxy * (k - 1) + 1 for k in 1:nz, b in 1:B)
            elseif name == "tied ends"
                @test i1[1, 1] == 1
            end
        end
    end
end

@testset "scan_best!: Val(1) is the serial kernel, exactly" begin

    @test !(1 in JP.SCAN_GROUPSIZES)
    ka = JP.KernelAbstractions
    for T in (Float32, Float64), (nx, ny, nz, B) in ((12, 10, 8, 3), (33, 31, 3, 2),
                                                     (1, 1, 3, 1), (64, 64, 4, 2))
        Random.seed!(hash((:val1, T, nx, ny, nz, B)) % 10_000)
        for score in (rand(T, nx, ny, nz, B), fill(T(2.5), nx, ny, nz, B))
            vs = Matrix{T}(undef, nz, B); is = Matrix{Int32}(undef, nz, B)
            v1 = Matrix{T}(undef, nz, B); i1 = Matrix{Int32}(undef, nz, B)
            JP.scan_best!(vs, is, score)
            bk = ka.get_backend(score)
            JP._launch_slab_min!(bk, Val(1), v1, i1, score, nx, ny, nz, B)
            ka.synchronize(bk)
            @test v1 == vs
            @test i1 == is
        end
    end
end

@testset "scan_best!: workgroup sizing" begin

    @test JP._scan_groupsize(1) == first(JP.SCAN_GROUPSIZES)
    @test JP._scan_groupsize(10) == first(JP.SCAN_GROUPSIZES)
    @test JP._scan_groupsize(32) == 32
    @test JP._scan_groupsize(63) == 32
    @test JP._scan_groupsize(64) == 64
    @test JP._scan_groupsize(16_384) == last(JP.SCAN_GROUPSIZES)
    @test all(ispow2, JP.SCAN_GROUPSIZES)
    @test issorted(collect(JP.SCAN_GROUPSIZES))
end
