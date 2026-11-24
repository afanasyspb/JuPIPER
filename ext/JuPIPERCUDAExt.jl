"""
    JuPIPERCUDAExt

Activated by `using CUDA, JuPIPER`. Supplies device allocation and memory
queries, cuFFT plans, and batched workspace construction for CUDA arrays.
Run device tests with `julia --project=gpu -t auto -e 'using CUDA, JuPIPER; include("test/runtests.jl")'`.
"""
module JuPIPERCUDAExt

using JuPIPER
using CUDA
using FFTW: plan_rfft, plan_brfft
using StaticArrays
using Adapt

const JP = JuPIPER


JP.allocate(::JP.CUDABackend, ::Type{T}, dims::Integer...) where {T} =
    CuArray{T}(undef, dims...)


JP.adapt_rots(::JP.CUDABackend, v::AbstractVector) = adapt(CuArray, v)


JP.adapt_backend(::JP.CUDABackend, x) = adapt(CuArray, x)


JP.device_free_bytes(::JP.CUDABackend) = Int(CUDA.free_memory())
JP.device_total_bytes(::JP.CUDABackend) = Int(CUDA.total_memory())


JP.device_collect!(::JP.CUDABackend) = (CUDA.reclaim(); nothing)


JP.device_synchronize!(::JP.CUDABackend) = (CUDA.synchronize(); nothing)


JP._is_device_oom(::CUDA.OutOfGPUMemoryError) = true


JP.FFTW_plan_rfft(A::CuArray, flags) = plan_rfft(A, 1:3)
JP.FFTW_plan_brfft(A::CuArray, d::Integer, flags) = plan_brfft(A, d, 1:3)


"""
    gpu_workspace(spec, nterms; batch, T)

Convenience constructor with an explicit `batch`.

Prefer [`gpu_autotune`](@ref) unless you already know the batch size that fits.
`batch` is the tuning knob that does not exist on CPU, it is the only thing that
can cause an OOM in this pipeline, and it is also what decides whether cuFFT
reaches its throughput plateau -- guessing it is exactly the mistake the
budgeter exists to prevent.
"""
gpu_workspace(spec, nterms; batch::Integer = 32, T = Float32) =
    JP.workspace(JP.CUDABackend(), spec, nterms; batch = batch)

"""
    gpu_autotune(spec, nterms; T=Float32, natoms=0, nrot=0, verbose=true)

Pick `batch` by allocating for real, largest-first, and keep the first size that
fits with headroom intact. Returns `(batch, report)`; see
[`JuPIPER.autotune_batch`](@ref) for the report fields.

The measured `report.fft_factor` is the number worth recording: it is the ratio
of true device cost to modelled buffer cost, and the difference is cuFFT's
out-of-pool work area, which no amount of arithmetic on our side can predict.
Log it with any GPU benchmark result.
"""
function gpu_autotune(spec::JP.GridSpec{T}, nterms::Integer;
                      natoms::Integer = 0, nrot::Integer = 0,
                      verbose::Bool = true, kwargs...) where {T}
    CUDA.functional() || error("no functional CUDA device; cannot autotune")
    return JP.autotune_batch(JP.CUDABackend(), spec, nterms;
                             natoms = natoms, nrot = nrot,
                             verbose = verbose, kwargs...)
end

"""
    gpu_diagnostics([io]) -> Nothing

Print device, driver, and memory state.
"""
function gpu_diagnostics(io::IO = stdout)
    if !CUDA.functional()
        println(io, "CUDA is NOT functional -- CPU path only.")
        println(io, "Common causes on a fresh cloud instance: no NVIDIA driver,")
        println(io, "a driver older than the CUDA runtime, or no GPU attached.")
        return nothing
    end
    CUDA.versioninfo(io)
    dev = CUDA.device()
    free, total = CUDA.memory_info()
    println(io)
    println(io, "device            ", CUDA.name(dev))
    println(io, "compute capability", " ", CUDA.capability(dev))
    println(io, "memory            ", round(free / 1024^3; digits = 2), " GiB free / ",
                round(total / 1024^3; digits = 2), " GiB total")
    println(io)
    CUDA.pool_status(io)
    return nothing
end

end
