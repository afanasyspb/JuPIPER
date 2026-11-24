#!/usr/bin/env julia
"""
    GPUArm

The seam that lets one benchmark script carry both a CPU and a GPU arm.

# Why this file exists

The mechanism both times was environmental, not algorithmic: every tier ran under
`--project=bench`, which has no CUDA, so `CuArray` was never constructed and every tier
silently measured the host CPU.

The fix has to survive a session that is not paying attention, so it is built
out of three properties rather than a convention:

  1. **Detection is by environment, not by flag.** `Base.identify_package`
     consults the *active* project's `[deps]`. Under `--project=bench` it
     returns `nothing` and the GPU arm skips; under `--project=gpu` it finds
     CUDA and the arm runs. Neither environment needs to know about the other,
     and no caller can accidentally claim a GPU arm that cannot exist.
  2. **An *invocation* can demand the GPU.** Set `JUPIPER_REQUIRE_GPU=1` and
     [`require_gpu!`](@ref) turns "no CUDA" from a silent skip into an immediate, loud
     failure, so a mis-wired harness fails in the first seconds of a metered instance
     instead of after forty minutes of benchmarking the host.
  3. **Every emitted row is self-describing.** [`provenance`](@ref) puts the device
     name, compute capability, driver, CUDA runtime, CUDA.jl version and memory into the
     result file, and each row carries `arm = "cpu"` or `arm = "gpu"`.

# Rows in older result files have no `arm` field

Everything committed before this file existed is CPU-only.

# API notes (read out of the depot, CUDA.jl 6.2.1)

`CUDA.synchronize`, `CUDA.memory_info`, `CUDA.driver_version`, `CUDA.runtime_version`,
`CUDA.attribute(dev, code)` all verified present.

The event API used by [`timeit_gpu_stages`](@ref) was read the same way, and it contains
the trap in its usual form. The type is **`CuEvent`**, not `Event`:
`CUDACore/lib/cudadrv/events.jl` defines `CuEvent(flags = EVENT_DEFAULT)`, `record(e,
stream = stream)` and `elapsed(start, stop)` -- the last returning **seconds**, having
divided the driver's milliseconds by 1000 -- while `synchronize(::CuEvent)` is
documented in that file but implemented at `CUDACore/lib/cudadrv/synchronization.jl:217`
with `blocking = false, spin = true` defaults. All four reach us as `CUDA.*` because
`src/CUDA.jl` does `@reexport using CUDACore`. `CUDA.Event` does not exist and is the
name one writes from memory.

The enum in `lib/cudadrv/libcuda.jl` spells them
`CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE`, which matches the C API and is therefore
exactly what one would write from memory -- but `lib/cudadrv/devices.jl:168` applies
`@enum_without_prefix visibility=:public CUdevice_attribute CU_`, so the name reachable
from outside is `CUDA.DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE`. The `CU_` form throws
`UndefVarError`. `capability` in that same file uses the stripped form, which is the
confirmation.
"""
module GPUArm

export gpu_available, require_gpu!, provenance, gpu_sync, timeit_gpu, timeit_gpu_stages
export gpu_free_bytes, fits_on_device, CUDAMOD


const _HAVE_CUDA_DEP = Base.identify_package("CUDA") !== nothing

if _HAVE_CUDA_DEP
    @eval import CUDA
    _probe() =
        try
            CUDA.functional()
        catch err
            @warn "CUDA is a dependency of this environment but is not functional" err
            false
        end
else
    _probe() = false
end

"""Whether a GPU arm can run: CUDA is in the active environment *and* functional."""
const AVAILABLE = _probe()

"""
The CUDA module itself, or `nothing`. Exposed so a caller can reach
`CUDAMOD.CuArray` without taking a hard dependency on CUDA in its own
environment -- which is what lets `bench/synthetic.jl` run unmodified under both
`--project=bench` and `--project=gpu`.
"""
const CUDAMOD = _HAVE_CUDA_DEP ? CUDA : nothing

gpu_available() = AVAILABLE

"""
    active_env_name -> String

The *name* of the active environment ("bench", "gpu", "JuPIPER"), never its path.

`Base.active_project` returns an absolute path, and this string is written into
committed result files. The environment name carries the entire diagnostic value --
"bench" versus "gpu" is exactly what tells you the harness was mis-wired -- so the path
is not withheld at any cost.
"""
function active_env_name()
    p = Base.active_project()
    p === nothing && return "none"
    return basename(dirname(p))
end

"""
    require_gpu!(what) -> Nothing

Fail immediately if `JUPIPER_REQUIRE_GPU=1` and no functional GPU arm exists.

This is the pre-flight guard in executable form. It exists because the failure mode it
prevents is not "the GPU arm crashed" -- that would have been noticed -- but "the GPU
arm was never reached and the run looked successful", which is what happened twice.

It is scoped to that launcher and explicitly cleared on the CPU one.

A host-wide demand makes this function refuse the first of those, which is a correct
configuration.
"""
function require_gpu!(what::AbstractString = "this benchmark")
    get(ENV, "JUPIPER_REQUIRE_GPU", "0") == "1" || return nothing
    AVAILABLE && return nothing
    error("""
        JUPIPER_REQUIRE_GPU=1 but no functional CUDA device is reachable, so the
        GPU arm of $what cannot run.

          CUDA in the active environment : $(_HAVE_CUDA_DEP) CUDA.functional : $(_HAVE_CUDA_DEP
          ? "false" : "n/a -- CUDA not loadable here") active environment : $(active_env_name)

        The usual cause is running a tier under `--project=bench`, which has no
        CUDA on purpose (adding it would rewrite the committed bench manifest and
        invalidate every A/B in this repository -- see gpu/Project.toml). Run the
        GPU arm under `--project=gpu` instead.""")
end

"""
    gpu_sync() -> Nothing

Block until the device is idle. Every GPU timing in this repository brackets its
work with this, because CUDA launches are asynchronous and an unsynchronised
`@elapsed` around a kernel launch measures the launch, not the kernel -- a
mistake that reports impossible speedups rather than obviously wrong ones.
"""
gpu_sync() = (AVAILABLE && CUDA.synchronize(); nothing)

"""
    timeit_gpu(f!; warm=3, reps=25) -> min_seconds

Minimum **host wall time** over `reps` synchronised runs.

Deliberately the *same* statistic and the same repetition count as the CPU
`timeit` in each tier: min-of-25 wall clock. The headline quantity for a tier is
a CPU-vs-GPU ratio, and a ratio between two differently-defined clocks is not a
ratio. Both arms measure host wall time around synchronised work, so the
comparison is licensed. Launch overhead is inside the measurement on purpose: it
is a real cost of a small batch, and revealing it is the point of the batch
sweep.

**Correct for a whole pipeline, wrong for a stage of one**, and using it for stages cost
this project a result. Because it synchronises around *every* measurement, a
decomposition built from repeated calls charges one full launch-plus-sync latency per
stage and then multiplies the per-term stages by `P`, so the sum counts roughly `3P+1`
host waits where the composed pipeline -- one stream, one wait at the end -- pays
exactly one. At the 0.15-0.40 ms per rotation an RTX 4090 actually takes, that latency
is not a correction, it is most of the reading: the stage sum came out at 114-466% of
the measured total, up to 4.7x the whole it was meant to decompose. Use
[`timeit_gpu_stages`](@ref) for anything that has to add up.
"""
function timeit_gpu(f!; warm::Int = 3, reps::Int = 25)
    for _ in 1:warm; f!(); end
    gpu_sync()
    best = Inf
    for _ in 1:reps
        t = @elapsed begin
            f!()
            gpu_sync()
        end
        best = min(best, t)
    end
    return best
end

"""
    timeit_gpu_stages(stages::NamedTuple; warm=3, reps=25) -> NamedTuple

Per-stage **device** time, one CUDA event per stage boundary, **one** host wait
per repetition. Same keys as `stages`, values in seconds, min over `reps`.

# What this fixes

[`timeit_gpu`](@ref) is a host clock wrapped around a device sync, so calling it once
per stage makes every stage pay a launch-plus-sync latency the composed pipeline pays
once. A "decomposition" that can total 4.7x its own whole is not a decomposition.

Here the stages are enqueued back to back on the task-local stream with an event
recorded at each boundary. `cuEventRecord` is a marker *in* the stream: it does
not stall the host and it does not serialise a stream that is already in-order,
so the sequence runs at pipeline speed and each interval is the device time
between two boundaries. The host waits once, on the final event, after all the
work is enqueued.

# Read the numbers with these three properties in mind

  * **It is a different clock from `t_total`, on purpose.** `t_total` stays on
    [`timeit_gpu`](@ref) because it is one side of a CPU/GPU ratio. So
    `sum(stages)` and `t_total` are no longer commensurable by construction, and
    their difference is now *meaningful* -- launch overhead and host-side cost
    that the device clock genuinely does not see -- rather than an artefact of
    how many times we synchronised. It should sit **below** `t_total`; if it does
    not, something else is wrong.
  * **An interval includes device idle time waiting for the next launch.** That
    is the honest reading: it is the stage's contribution to stream occupancy,
    which is what a pipeline pays.
  * **Resolution is the driver's, about half a microsecond.** Fine for stages of
    tens of microseconds and up; a stage that lands near the floor should be
    reported as such rather than divided into.

The min is taken per stage across repetitions, matching the min-of-25 used everywhere
else in this repository.
"""
function timeit_gpu_stages(stages::NamedTuple; warm::Int = 3, reps::Int = 25)
    AVAILABLE || error("timeit_gpu_stages requires a functional CUDA device")
    fs = collect(values(stages))
    n  = length(fs)

    evs = [CUDA.CuEvent() for _ in 1:(n + 1)]

    for _ in 1:warm, f! in fs
        f!()
    end
    gpu_sync()

    best = fill(Inf, n)
    for _ in 1:reps
        CUDA.record(evs[1])
        for i in 1:n
            fs[i]()
            CUDA.record(evs[i + 1])
        end

        CUDA.synchronize(evs[n + 1])
        for i in 1:n
            best[i] = min(best[i], Float64(CUDA.elapsed(evs[i], evs[i + 1])))
        end
    end
    return NamedTuple{keys(stages)}(Tuple(best))
end

"""Free device bytes, or `-1` with no device. The driver figure, not pool accounting."""
gpu_free_bytes() = AVAILABLE ? Int(CUDA.free_memory()) : -1

"""
    fits_on_device(bytes; margin=0.15) -> Bool

Whether `bytes` fits in free device memory with `margin` left over.

Used to skip an oversized sweep point rather than let it throw. A skip is
recorded in the output; a crash halfway through a metered run is not recoverable.
"""
function fits_on_device(bytes::Real; margin::Real = 0.15)
    AVAILABLE || return false
    return bytes <= (1 - margin) * gpu_free_bytes()
end

"""
    peak_bandwidth_gbs -> Union{Float64,Nothing}

Theoretical peak memory bandwidth in GB/s (10^9), from the **device's own
reported** clock and bus width:

    2 * memory_clock_kHz * (bus_width_bits / 8) / 1e6

the doubling being DDR. Taking the number from the driver rather than from a vendor page
means it describes the card actually in the machine.

Returns `nothing` if the attributes are unavailable, because provenance must
never be able to abort a benchmark.
"""
function peak_bandwidth_gbs()
    AVAILABLE || return nothing
    try
        dev = CUDA.device()
        clock_khz = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)
        bus_bits  = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)
        return 2.0 * clock_khz * (bus_bits / 8) / 1e6
    catch err
        @warn "could not read device bandwidth attributes; peak will be reported as null" err
        return nothing
    end
end

"""
    provenance() -> NamedTuple

Everything a reader needs to know that a result file describes a real device.

Goes into the JSON header of every tier that has a GPU arm. `available = false`
is itself the useful answer on a CPU-only run: it is the field that distinguishes
"this file has no GPU rows because there is no GPU" from "this file has no GPU
rows because the harness was mis-wired", and the second is what happened twice.
"""
function provenance()
    AVAILABLE || return (available = false,
                         reason = _HAVE_CUDA_DEP ? "CUDA loaded but not functional" :
                                                   "CUDA not in the active environment",
                         active_env = active_env_name())
    dev = CUDA.device()
    free, total = CUDA.memory_info()

    ver(f, default) = try f() catch; default end
    return (available     = true,
            device        = String(CUDA.name(dev)),
            capability    = string(CUDA.capability(dev)),
            driver        = ver(() -> string(CUDA.driver_version()), "unknown"),
            cuda_runtime  = ver(() -> string(CUDA.runtime_version()), "unknown"),
            cuda_jl       = ver(() -> string(pkgversion(CUDA)), "unknown"),
            sms           = ver(() -> Int(CUDA.attribute(dev,
                                CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)), -1),
            peak_gbs      = peak_bandwidth_gbs(),

            l2_bytes      = ver(() -> Int(CUDA.attribute(dev,
                                CUDA.DEVICE_ATTRIBUTE_L2_CACHE_SIZE)), -1),
            mem_free_gib  = free / 2^30,
            mem_total_gib = total / 2^30)
end

"""
    l2_bytes() -> Int

L2 cache size in bytes, or `-1` with no device. Read from the driver, never
recalled -- see the note in [`provenance`](@ref).
"""
function l2_bytes()
    AVAILABLE || return -1
    try
        return Int(CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_L2_CACHE_SIZE))
    catch
        return -1
    end
end

"""Print the device banner a benchmark log should open with."""
function banner(io::IO = stdout)
    p = provenance()
    if !p.available
        println(io, "GPU arm: SKIPPED -- ", p.reason)
        println(io, "         active environment: ", p.active_env)
        return nothing
    end
    println(io, "GPU arm: ", p.device, "  (cc ", p.capability, ", ", p.sms, " SMs)")
    println(io, "         driver ", p.driver, " | CUDA runtime ", p.cuda_runtime,
                " | CUDA.jl ", p.cuda_jl)
    print(io, "         memory ", round(p.mem_free_gib; digits = 2), " / ",
              round(p.mem_total_gib; digits = 2), " GiB free")
    p.peak_gbs === nothing ? println(io) :
        println(io, " | peak BW ", round(p.peak_gbs; digits = 1), " GB/s (device-reported spec)")
    p.l2_bytes > 0 && println(io, "         L2 cache ", p.l2_bytes, " B = ",
                              round(p.l2_bytes / 2^20; digits = 2),
                              " MiB (driver-reported)")
    return nothing
end

end
