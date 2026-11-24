"""
Allocation regression guard.

"Zero allocations in the hot loop" is a project requirement, so it is asserted
here rather than believed. If this test fails, the fix is to remove the
allocation -- never to relax the threshold.

History: this test is why `voxelize!` and `scan_best!` have hand-written `::Array`
methods. KernelAbstractions' CPU backend allocates ~224 B per launch, which took
`correlate!` to 896 B/call.
"""

using Test, JuPIPER, StaticArrays, FFTW, Random
const JP = JuPIPER

function build(T, n, P, natoms)
    spec = JP.GridSpec{T}((n, n, n), T(1.2), SVector{3,T}(-n * T(0.6), -n * T(0.6), -n * T(0.6)))
    bk = JP.CPUBackend()
    s = JP.Structure([SVector{3,T}(randn(), randn(), randn()) .* T(n * 0.1) for _ in 1:natoms],
                     rand(T, natoms, P))
    rs = JP.receptor_spectra(bk, s, spec, ones(T, P); flags = FFTW.ESTIMATE)
    ws = JP.workspace(bk, spec, P; batch = 1, flags = FFTW.ESTIMATE)
    (spec, s, rs, ws)
end

@testset "hot loop allocates nothing" begin
    Random.seed!(7)
    for T in (Float32, Float64), n in (16, 32)
        P = 4
        spec, s, rs, ws = build(T, n, P, 300)
        rots = [one(SMatrix{3,3,T,9})]
        vals = Matrix{T}(undef, n, 1)
        idxs = Matrix{Int32}(undef, n, 1)

        JP.correlate!(ws, s, rots, rs)
        JP.voxelize!(ws.lig_real, s, rots, 1, spec)
        JP.scan_best!(vals, idxs, ws.score)

        @test (@allocated JP.correlate!(ws, s, rots, rs)) == 0
        @test (@allocated JP.voxelize!(ws.lig_real, s, rots, 1, spec)) == 0
        @test (@allocated JP.scan_best!(vals, idxs, ws.score)) == 0
    end
end

@testset "workspace_bytes matches the documented budget" begin
    T = Float32
    n = 128
    P = 6
    spec = JP.GridSpec{T}((n, n, n), T(1.2), SVector{3,T}(-76.8, -76.8, -76.8))
    ws = JP.workspace(JP.CPUBackend(), spec, P; batch = 1, flags = FFTW.ESTIMATE)
    mib = JP.workspace_bytes(ws) / 2^20

    @test 32.0 < mib < 32.5
end
