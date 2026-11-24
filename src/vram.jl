"""
    device_free_bytes(backend) -> Int
    device_total_bytes(backend) -> Int

Free and total memory of the device `backend` allocates on.

On `CPUBackend` these are the host figures from `Sys`. The CUDA extension maps
them to `CUDA.free_memory()` / `CUDA.total_memory()`.

!!! note "cuFFT work areas are invisible to the pool, not to these"
    cuFFT allocates each plan's work area itself, outside Julia's memory pool
    (verified in CUDA.jl 6.2.1, `lib/cufft/src/wrappers.jl`: *"we currently do
    not use the worksize to allocate our own workarea, instead relying on the
    automatic allocation strategy"*). That memory therefore does **not** appear
    in `CUDA.used_memory()` / `CUDA.cached_memory()` and cannot be freed by the
    Julia GC -- but it does reduce `free_memory()`. Budget against these
    functions, never against pool accounting.
"""
device_free_bytes(::CPUBackend) = Int(Sys.free_memory())
device_total_bytes(::CPUBackend) = Int(Sys.total_memory())

device_free_bytes(b::Backend) = _backend_unavailable(b, "device_free_bytes")
device_total_bytes(b::Backend) = _backend_unavailable(b, "device_total_bytes")

"""
    device_synchronize!(backend) -> Nothing

Block until every operation the backend has queued has actually executed.

**A host-side memory query is a race without this.** CUDA.jl allocates through
`cuMemAllocFromPoolAsync` -- verified in the depot, CUDA.jl 6.2.1,
`CUDACore/src/memory.jl:_pool_alloc` -> `lib/cudadrv/memory.jl:alloc(...; async=true)`
-- which is *stream-ordered*: the host gets a pointer immediately and the driver
reserves backing memory only when the stream reaches that point. `CUDA.free_memory` is a
live `cuMemGetInfo_v2` issued from the host. So allocating and then differencing free
memory, with nothing in between, can legitimately observe none of the allocation.

It did: the first RTX 4090 device-test run measured **exactly zero bytes** for a 4.9 MiB
probe, which drove the calibrated `fft_factor` negative. No-op on CPU, where
`Sys.free_memory` has no queue in front of it.

Deliberately **not** called from [`maybe_collect!`](@ref): that is a pressure
heuristic on the hot path, and a device stall there would cost far more than the
precision it buys. Calibration synchronises; pressure checks do not.
"""
device_synchronize!(::CPUBackend) = nothing
device_synchronize!(b::Backend) = _backend_unavailable(b, "device_synchronize!")

"""
    _free_after_sync(backend) -> Int

`device_free_bytes` with the queue drained first. Every free-memory reading used
in a *calibration* goes through this; the name is the invariant.
"""
_free_after_sync(backend::Backend) =
    (device_synchronize!(backend); device_free_bytes(backend))

"""
    device_collect!(backend) -> Nothing

Return cached device memory to the driver. No-op on CPU.

On CUDA this is `CUDA.reclaim()`, which at its default `RECLAIM_DROP` level
**already runs a full `GC.gc(true)` internally** (CUDA.jl 6.2.1,
`CUDACore/src/memory.jl`). Writing `GC.gc(true); CUDA.reclaim()` therefore
collects twice and is strictly slower for no benefit.

Call this at job and sweep boundaries only -- see [`MemoryGuard`](@ref) for why
it must not appear in the rotation loop.
"""
device_collect!(::CPUBackend) = nothing
device_collect!(b::Backend) = _backend_unavailable(b, "device_collect!")

"""
    MemoryModel

Exact device-byte accounting for one docking job, split into the part that
scales with `batch` and the part that does not.

Fields are counts, not bytes, so the same model answers for any element type.

    grid      nx*ny*nz              real-grid elements
    speclen   (nx>>1+1)*ny*nz       r2c spectrum elements (complex)
    nz        grid slabs            scan scratch rows
    elsize    sizeof(T)
    nterms    P                     receptor spectra held resident
    natoms    ligand atom count
    nrot      rotations resident on device

Build one with [`memory_model`](@ref) and query with [`bytes_per_batch`](@ref),
[`bytes_fixed`](@ref), [`job_bytes`](@ref).
"""
struct MemoryModel
    grid::Int
    speclen::Int
    nz::Int
    elsize::Int
    nterms::Int
    natoms::Int
    nrot::Int
end

"""
    memory_model(spec, nterms; natoms=0, nrot=0, T=Float32) -> MemoryModel

Derive the byte model from the grid the job will actually run on.

`natoms` and `nrot` are the ligand size and the number of rotations that will be
resident on the device. Both contribute only to the fixed term and are small
(a few MiB at realistic sizes), but they are counted rather than waved away
because "small" is how budgets drift.
"""
function memory_model(spec::GridSpec, nterms::Integer;
                      natoms::Integer = 0, nrot::Integer = 0, T::Type = Float32)
    nx, ny, nz = spec.n
    return MemoryModel(nx * ny * nz, ((nx >> 1) + 1) * ny * nz, nz,
                       sizeof(T), Int(nterms), Int(natoms), Int(nrot))
end

"""
    bytes_per_batch(m::MemoryModel) -> Int

Device bytes consumed by **one** additional batch slot.

This is `Workspace`'s four buffers plus the per-slot scan scratch:

    lig_real  grid * elsize
    score     grid * elsize
    lig_spec  speclen * 2 * elsize
    accum     speclen * 2 * elsize
    scan      nz * (elsize + 4)        vals::T and idxs::Int32

`test_vram.jl` asserts the agreement against `workspace_bytes` rather than trusting the
arithmetic.
"""
bytes_per_batch(m::MemoryModel) =
    2 * m.elsize * m.grid + 4 * m.elsize * m.speclen + m.nz * (m.elsize + 4)

"""
    bytes_fixed(m::MemoryModel) -> Int

Device bytes that do **not** scale with `batch`: the P receptor spectra (the
dominant term), the resident rotation set, and the ligand's coordinates and
term weights.
"""
function bytes_fixed(m::MemoryModel)
    spectra = m.nterms * m.speclen * 2 * m.elsize
    rots    = m.nrot * 9 * m.elsize
    ligand  = m.natoms * (3 + m.nterms) * m.elsize
    return spectra + rots + ligand
end

"""
    FFT_FACTOR_MEASURED

The calibrated `fft_factor` for cuFFT: **1.50**, measured on an NVIDIA RTX 4090 (CUDA
13.2, driver 595.58.03, CUDA.jl 6.2.1) at 128^3 / P=6 / Float32, where `autotune_batch`
selected `batch = 321` and measured 15.22 GiB resident against 15.2149 GiB predicted at
this factor. Supersedes the 2.0 `[ESTIMATE]` that stood in this file until the first
device run.

The residual over our own buffers is `0.50 * bytes_per_batch` = 16.13 MiB per
batch slot, which brackets neatly between two real grids (0.4961) and two complex
spectra (0.5039): **one work area per plan, each about the size of one transform
buffer, with two plans live.** That the measurement captures *both* plans and not
just the forward one was checked against the installed source rather than assumed
-- `cufftSetAutoAllocation` is never called by CUDA.jl, so cuFFT's automatic
strategy allocates inside `cufftXtMakePlanMany` at **plan creation**;
`plan_rfft`/`plan_brfft` acquire the handle eagerly (`lib/cufft/src/fft.jl`); and
[`workspace`](@ref) builds both plans up front. So both work areas are resident
before `_probe_workspace` differences free memory.

!!! warning "One grid size, one configuration"
    This is a single measurement at 128^3 / P=6 / Float32 on one card. `autotune_batch`
    re-measures it on every run, which is the intended path; this constant is the
    default for arithmetic done without a device present.

!!! note "The calibration this was obtained under has since been corrected"
    `_probe_workspace` used to build only the `Workspace`, while
    [`calibrate_fft_factor`](@ref) subtracted [`bytes_fixed`](@ref) -- memory the probe
    had never allocated. The error is `bytes_fixed / (bytes_per_batch * batch)`, so it
    **shrinks with grid size and with batch**: 0.5% at the 128^3 / batch=321
    configuration this constant was measured at, and 14% at the 32^3 / batch=8
    configuration the device test uses, where it flipped the sign. Recomputing the
    1.5001 reported at the time gives **1.5048 from the spectra term alone and
    approximately 1.505 with the resident rotations and ligand included**; the latter is
    reconstructed rather than recorded, because that run wrote no file.

    The constant is left at **1.50**, deliberately. The shift is 0.3%, an order of
    magnitude below the 10% `headroom` it feeds, and re-deriving a committed constant
    from arithmetic over a transcribed console figure would be worse provenance than
    leaving it.
"""
const FFT_FACTOR_MEASURED = 1.50

"""
    FFT_FACTOR_FLOOR

Smallest `fft_factor` that may be used to *size* a job: **1.0**, i.e. "the FFT
library needs no scratch beyond our own buffers".

A calibration can legitimately come out below this -- a probe that fails to resolve its
own allocation returns 0 bytes and the arithmetic then returns a negative number, which
is what the first device run produced. Feeding such a value to [`max_batch`](@ref) would
admit a batch larger than the device can hold, which is the one direction this whole
file exists to prevent. So sizing uses [`safe_fft_factor`](@ref) and reporting keeps the
raw value; the two are separate fields in the autotune report and must not be collapsed
into one.
"""
const FFT_FACTOR_FLOOR = 1.0

"""
    safe_fft_factor(raw) -> Float64

`max(FFT_FACTOR_FLOOR, raw)` -- the value it is safe to size a job with.

Clamping is **not** a way to make a bad measurement usable; it is a way to stop a
bad measurement from being *dangerous*. The unclamped number is always reported
alongside as `fft_factor_raw`, because a factor below 1.0 is diagnostic: it says
the probe never saw the memory it allocated, and that is information, not noise.
"""
safe_fft_factor(raw::Real) = max(FFT_FACTOR_FLOOR, Float64(raw))

"""
    job_bytes(m::MemoryModel, batch; fft_factor=FFT_FACTOR_MEASURED) -> Int

Total device bytes for the job at a given `batch`, including an allowance for
FFT work areas.

`fft_factor` multiplies [`bytes_per_batch`](@ref) only. It is the one term this
file cannot compute:

  * FFTW plans hold negligible scratch for out-of-place transforms, so
    `fft_factor = 1.0` is right on CPU.
  * cuFFT auto-allocates a work area per plan, outside the Julia pool, and its
    size depends on the transform decomposition. Two plans are live (r2c forward,
    c2r inverse).

The default is [`FFT_FACTOR_MEASURED`](@ref) = 1.50, which is a *measurement*
from an RTX 4090 rather than the 2.0 guess it replaced -- read its docstring for
the configuration it was measured at, which is narrower than the set of
configurations this function will be called with. [`autotune_batch`](@ref)
obtains the true value for the run in hand by allocating for real and
differencing `device_free_bytes`, which remains the only way to see memory the
pool does not account for.
"""
job_bytes(m::MemoryModel, batch::Integer; fft_factor::Real = FFT_FACTOR_MEASURED) =
    bytes_fixed(m) + ceil(Int, fft_factor * bytes_per_batch(m)) * Int(batch)

"""
    max_batch(m, free_bytes; headroom=0.10, reserve=512*1024^2,
              fft_factor=FFT_FACTOR_MEASURED, limit=1024) -> Int

Largest `batch` whose job fits in `free_bytes`, keeping a safety margin.

Solves `bytes_fixed + fft_factor * bytes_per_batch * batch <= usable` with

    usable = free_bytes * (1 - headroom) - reserve

Two independent margins, because they cover different things. `reserve` is an
absolute floor for costs that do not scale with the problem: the CUDA context,
the loaded module image, cuBLAS/cuFFT handle caches, and the driver's own
bookkeeping -- collectively a few hundred MiB and roughly constant.  `headroom`
is a proportional margin for the part that *does* scale and that we model only
approximately, chiefly the cuFFT work areas.

Returns `0` when even `batch = 1` does not fit; the caller must treat that as a
hard failure (a smaller grid or a bigger card), not clamp it to 1.

This is a **capacity** figure and not a batch to run at -- see the note on
[`autotune_batch`](@ref), where the gap between the two is quantified.
"""
function max_batch(m::MemoryModel, free_bytes::Integer;
                   headroom::Real = 0.10, reserve::Integer = 512 * 1024^2,
                   fft_factor::Real = FFT_FACTOR_MEASURED, limit::Integer = 1024)
    0 <= headroom < 1 || throw(ArgumentError("headroom must be in [0, 1)"))
    usable = floor(Int, free_bytes * (1 - headroom)) - Int(reserve)
    avail = usable - bytes_fixed(m)
    avail <= 0 && return 0
    per = ceil(Int, fft_factor * bytes_per_batch(m))
    return clamp(avail ÷ per, 0, Int(limit))
end

"""
    calibrate_fft_factor(m, batch, measured_bytes) -> Float64

Back out the true `fft_factor` from a measured allocation.

`measured_bytes` is the device memory actually consumed by building the job at
`batch` -- the drop in [`device_free_bytes`](@ref) across construction. Because
that figure includes cuFFT's out-of-pool work areas, this is the only route to
a real number. It is how [`FFT_FACTOR_MEASURED`](@ref) was obtained, and it is
still the right call on any new card or grid size, because that constant is one
measurement at one configuration rather than a law.

!!! warning "`measured_bytes` must cover the fixed part too"
    This is the exact inverse of [`job_bytes`](@ref), so it subtracts
    [`bytes_fixed`](@ref) -- and is therefore only correct if whatever produced
    `measured_bytes` actually allocated the fixed part. [`_probe_workspace`](@ref) once
    did not, which made the returned factor low by `bytes_fixed / (bytes_per_batch *
    batch)`: invisible at 128^3 with batch=321 and 14% at 32^3 with batch=8. The probe
    now pads to the full job footprint. If you call this with a figure from somewhere
    else, check what that figure includes.

Returns the multiplier on [`bytes_per_batch`](@ref) that reproduces the
measurement, **uncensored**. A value at or below 1.0 is returned as measured
rather than clamped, because it is diagnostic: our own buffers alone account for
a factor of 1.0, so anything less means the measurement did not resolve them.
Use [`safe_fft_factor`](@ref) before feeding the result to [`max_batch`](@ref).
"""
function calibrate_fft_factor(m::MemoryModel, batch::Integer, measured_bytes::Integer)
    batch > 0 || throw(ArgumentError("batch must be positive"))
    scaling = Int(measured_bytes) - bytes_fixed(m)
    return scaling / (bytes_per_batch(m) * Int(batch))
end

"""
    autotune_batch(backend, spec, nterms; T, natoms, nrot, flags,
                   candidates=nothing, headroom, reserve, verbose) -> (batch, report)

Choose `batch` by construction rather than by arithmetic: build a real
`Workspace` at descending candidate sizes and keep the largest one that both
allocates and leaves the requested headroom intact.

Why this exists rather than trusting [`max_batch`](@ref): the arithmetic cannot
see cuFFT's work area (allocated outside the Julia pool) and cannot see the
plan-handle cache. Allocating for real and differencing `device_free_bytes`
measures both.

Returns `(batch, report)` where `report` is a NamedTuple carrying
`free_before`, `free_after_release`, `measured_bytes`, `predicted_bytes`,
`own_bytes`, `fft_factor`, `fft_factor_raw`, `resolved`, and `tried` -- the
descending list of candidates with their outcomes. Log the report with any GPU
benchmark result; a throughput number without the batch size it ran at is not a
result.

!!! danger "This answers *what fits*, not *what is fast*. They are far apart."
    The returned `batch` is a **capacity** answer: the largest workspace the device will
    hold with the requested margins. It is not a recommendation, and on this workload
    the two numbers differ by more than two orders of magnitude. Sizing a production run
    from this function's return value therefore buys nothing and costs a factor of a few
    -- the classic reading of a GPU tuner as if bigger batches were free.

    Use this function for the ceiling -- "will this grid run at all on this card" -- and
    the sweep for the operating point.

!!! note "`fft_factor` and `fft_factor_raw` are different numbers on purpose"
    `fft_factor_raw` is exactly what the arithmetic produced. `fft_factor` is `max(1.0,
    raw)` -- the value it is *safe* to size with, since a factor below 1.0 fed to
    [`max_batch`](@ref) would admit a batch the device cannot hold. `resolved` says
    which of the two to believe: it is `false` when the probe came back below its own
    floor, meaning the free-memory query did not resolve an allocation we know we made.
    **When `resolved` is false, `fft_factor_raw` is not a measurement** and must not be
    recorded as one -- the run that motivated this returned exactly 0 bytes and hence a
    negative factor. `own_bytes` is `job_bytes` at `fft_factor = 1.0`, i.e. the floor
    the probe is measured against.

!!! warning "Reclaims between probes, deliberately"
    [`device_collect!`](@ref) is called after every failed or discarded probe.
    This is one of the two places collection is correct (the other is a job
    boundary): on CUDA it is what purges the cuFFT plan-handle cache, which is
    keyed on the *full* input size including the batch dimension. Without it, a
    descending sweep accumulates one retained work area per candidate tried and
    can exhaust memory through the tuner itself.
"""
function autotune_batch(backend::Backend, spec::GridSpec{T}, nterms::Integer;
                        natoms::Integer = 0, nrot::Integer = 0,
                        flags = FFTW.ESTIMATE,
                        candidates::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                        headroom::Real = 0.10, reserve::Integer = 512 * 1024^2,
                        verbose::Bool = false) where {T}
    m = memory_model(spec, nterms; natoms = natoms, nrot = nrot, T = T)
    free0 = device_free_bytes(backend)
    hi = max_batch(m, free0; headroom = headroom, reserve = reserve)
    hi == 0 && error("""
        No batch size fits: grid $(spec.n) with $nterms terms needs
        $(_fmt_bytes(job_bytes(m, 1))) at batch=1 but only
        $(_fmt_bytes(free0)) is free (reserve $(_fmt_bytes(reserve)),
        headroom $(round(100*headroom))%). Use a smaller grid or a larger device.
        """)

    cands = candidates === nothing ? _descending_batches(hi) : sort(collect(Int.(candidates)); rev = true)
    tried = Tuple{Int,Symbol,Int}[]
    for b in cands
        b < 1 && continue
        device_collect!(backend)
        before = _free_after_sync(backend)
        ok, used = _probe_workspace(backend, m, spec, nterms, b, flags)
        if !ok
            push!(tried, (b, :oom, 0))
            verbose && println("  batch=$b: allocation failed")
            continue
        end
        push!(tried, (b, :ok, used))
        verbose && println("  batch=$b: $(_fmt_bytes(used)) resident while loaded")
        device_collect!(backend)

        raw = calibrate_fft_factor(m, b, used)

        res = raw >= FFT_FACTOR_FLOOR
        if !res

            @warn """autotune probe did not resolve its own allocation, so `fft_factor_raw` is NOT a
measurement for this run. Sizing falls back to the floor ($(FFT_FACTOR_FLOOR)).""" batch =
                  b maxlog = 1
        end
        return (b, (free_before = before,
                    free_after_release = _free_after_sync(backend),
                    measured_bytes = used,
                    predicted_bytes = job_bytes(m, b),
                    own_bytes = job_bytes(m, b; fft_factor = 1.0),
                    fft_factor = safe_fft_factor(raw),
                    fft_factor_raw = raw,
                    resolved = res,
                    tried = tried))
    end
    device_collect!(backend)
    error("every candidate batch size failed to allocate: $(cands)")
end


function _descending_batches(hi::Integer)
    out = Int[]
    b = 1
    while b <= hi
        push!(out, b)
        b <<= 1
    end
    isempty(out) && return [1]
    hi > last(out) && push!(out, Int(hi))
    return reverse(out)
end

"""
    _probe_workspace(backend, m, spec, nterms, batch, flags) -> (ok, used_bytes)

Build the **whole job** for real and report the device bytes it cost. Returns
`(false, 0)` on an allocation failure so the caller can step down.

Three things here are load-bearing, and two of them were learned by being wrong.

**It pads to the full job footprint.** `workspace` allocates the four hot-loop buffers
and nothing else; the receptor spectra, the resident rotation set, the ligand and the
scan scratch are [`bytes_fixed`](@ref) plus a per-slot term, and they are what
[`job_bytes`](@ref) -- and hence [`max_batch`](@ref) and [`calibrate_fft_factor`](@ref)
-- account for. Probing without them measured one thing and calibrated against another;
the discrepancy is `bytes_fixed/(bytes_per_batch*batch)`, which is 0.5% at
128^3/batch=321 and 14% at 32^3/batch=8. It also meant a batch could pass the probe and
then OOM when the spectra were allocated. The pad is one flat `UInt8` buffer of exactly
the missing bytes: the same byte count, allocated in one block rather than several,
which is the one respect in which it is not the real job.

**It executes before measuring.** A lazily-planned backend allocates its work
area on first execution rather than at plan construction, so measuring straight
after `workspace` would see a work area of zero.

**It synchronises before measuring.** Knowing *when* the device does the work is not
enough if the host never waits for it -- CUDA.jl's allocator is stream-ordered and
asynchronous, so `cuMemGetInfo` issued straight afterwards can observe none of it. That
is not hypothetical: it returned exactly 0 bytes on the first device run. See
[`device_synchronize!`](@ref).

The GC calls are not hot-loop policy -- they run once per probe, to make the
before/after difference attributable to this workspace rather than to whatever
the previous probe left unfinalised.
"""
function _probe_workspace(backend::Backend, m::MemoryModel, spec::GridSpec{T},
                          nterms::Integer, batch::Integer, flags) where {T}
    GC.gc()
    device_collect!(backend)
    before = _free_after_sync(backend)
    try
        ws = workspace(backend, spec, nterms; batch = batch, flags = flags)

        pad = max(job_bytes(m, batch; fft_factor = 1.0) - workspace_bytes(ws), 1)
        padbuf = allocate(backend, UInt8, pad)

        used = GC.@preserve ws padbuf begin
            fill!(padbuf, 0x00)
            fill!(ws.lig_real, zero(T))
            mul!(ws.lig_spec, ws.fwd, ws.lig_real)
            before - _free_after_sync(backend)
        end
        return (true, used)
    catch err

        _is_device_oom(err) || rethrow()
        return (false, 0)
    finally
        GC.gc()
    end
end

"""
    _is_device_oom(err) -> Bool

Backend hook: is this exception an out-of-device-memory condition? The CUDA
extension adds a method for `CUDA.OutOfGPUMemoryError`. Everything else is a
genuine bug and must propagate.
"""
_is_device_oom(::Any) = false
_is_device_oom(::OutOfMemoryError) = true


"""
    MemoryGuard

Pressure-triggered device collection for the rotation loop, with the check
interval **derived** rather than tuned.

# The derivation

Let `H` be the usable headroom in bytes (free memory minus the reserve we refuse
to dip into) and `a` the bytes of device garbage produced per rotation. If the
guard checks every `k` rotations, worst-case growth between two checks is `k*a`.
Never exceeding the headroom requires

    k * a <= H    =>    k* = floor(H / a)

Checking more often than `k*` cannot improve safety and costs a `free_memory()`
query -- a driver call -- every time. Checking less often can miss an overrun.
So `k*` is optimal in both directions, and [`guard_interval`](@ref) computes it.

# What that yields here

`a = 0`. The rotation loop is allocation-free, and `test/test_allocations.jl`
asserts it at 0 B rather than leaving it as an aspiration. So `k* = infinity`:
the correct number of collections inside a JuPIPER rotation loop is **zero**,
and that is a consequence of the memory model, not a tuning result. A
`MemoryGuard` built from a measured `a = 0` therefore never fires, and
[`maybe_collect!`](@ref) degenerates to one integer comparison.

The guard is kept, rather than deleted as dead code, for two reasons. It makes
the zero-allocation invariant load-bearing at runtime instead of only in the
test suite; and if a future feature does allocate per rotation (streaming the
rotation set from host, or per-pose output buffers), the same formula
immediately yields a finite, correct interval with no redesign.

# What it does not protect against

If the shortfall is cuFFT's work area, no collection helps -- that memory is
outside Julia's pool and outside the GC's reach (see [`device_free_bytes`](@ref)).
The only lever there is a smaller `batch`, which is [`autotune_batch`](@ref)'s
job. A guard that appears to protect against an OOM it cannot in fact prevent is
worse than no guard, so this is stated here rather than left implicit.
"""
mutable struct MemoryGuard{B<:Backend}
    const backend::B
    const interval::Int
    const low_water::Int
    counter::Int
    collections::Int
end

"""
    guard_interval(headroom_bytes, bytes_per_rotation) -> Int

`floor(H / a)`, the largest safe number of rotations between pressure checks.
Returns `typemax(Int)` for `a == 0`, which is the measured value for this
codebase's hot loop and means "never check".
"""
function guard_interval(headroom_bytes::Integer, bytes_per_rotation::Real)
    bytes_per_rotation <= 0 && return typemax(Int)
    return max(1, floor(Int, headroom_bytes / bytes_per_rotation))
end

"""
    memory_guard(backend; bytes_per_rotation=0, reserve=512*1024^2, low_water=nothing)

Build a [`MemoryGuard`](@ref) whose interval is derived from the measured
per-rotation allocation rate.

The default of `0` encodes the current, asserted invariant.
"""
function memory_guard(backend::Backend; bytes_per_rotation::Real = 0,
                      reserve::Integer = 512 * 1024^2,
                      low_water::Union{Nothing,Integer} = nothing)
    free = device_free_bytes(backend)
    H = max(0, free - Int(reserve))
    lw = low_water === nothing ? Int(reserve) : Int(low_water)
    return MemoryGuard(backend, guard_interval(H, bytes_per_rotation), lw, 0, 0)
end

"""
    maybe_collect!(guard) -> Bool

Advance the guard by one rotation; collect and return `true` only if the
interval has elapsed **and** free memory is below the low-water mark.

Allocation-free, and on the common path (interval not elapsed) it is a single
increment and compare. With `interval == typemax(Int)` -- the value
[`memory_guard`](@ref) derives for this codebase -- the free-memory query is
never issued at all.

`maybe_collect!(nothing)` returns `false`. That method is why passing
`guard = nothing` to [`dock!`](@ref) costs literally nothing: Julia specialises
on `Nothing`, constant-folds the call to `false`, and deletes the branch. The
default docking path is therefore bit-identical to one with no guard in it, and
`test/test_allocations.jl` still measures 0 B.
"""
function maybe_collect!(g::MemoryGuard)
    g.counter += 1
    g.counter < g.interval && return false
    g.counter = 0
    if device_free_bytes(g.backend) < g.low_water
        device_collect!(g.backend)
        g.collections += 1
        return true
    end
    return false
end

maybe_collect!(::Nothing) = false

function _fmt_bytes(b::Integer)
    b < 1024 && return "$(b) B"
    b < 1024^2 && return @sprintf("%.1f KiB", b / 1024)
    b < 1024^3 && return @sprintf("%.1f MiB", b / 1024^2)
    return @sprintf("%.2f GiB", b / 1024^3)
end

"""
    memory_report(backend, spec, nterms, batch; T, natoms, nrot) -> String

Human-readable breakdown of where the device memory goes.
"""
function memory_report(backend::Backend, spec::GridSpec{T}, nterms::Integer,
                       batch::Integer; natoms::Integer = 0, nrot::Integer = 0) where {T}
    m = memory_model(spec, nterms; natoms = natoms, nrot = nrot, T = T)
    free = device_free_bytes(backend)
    total = device_total_bytes(backend)
    io = IOBuffer()
    println(io, "Device memory budget -- grid $(spec.n), P=$(nterms), $(T), batch=$(batch)")
    println(io, "  total on device      ", _fmt_bytes(total))
    println(io, "  free now             ", _fmt_bytes(free))
    println(io, "  fixed (spectra+rots) ", _fmt_bytes(bytes_fixed(m)))
    println(io, "  per batch slot       ", _fmt_bytes(bytes_per_batch(m)))
    println(io, "  job at batch=$(batch)      ", _fmt_bytes(job_bytes(m, batch)),
                "  [fft_factor=$(FFT_FACTOR_MEASURED) MEASURED @ 128^3/P=6/f32, RTX 4090]")
    println(io, "  max safe batch       ", max_batch(m, free))
    return String(take!(io))
end
