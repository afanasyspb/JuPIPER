using Test, JuPIPER

@testset "JuPIPER" begin
    include("test_correctness.jl")
    include("test_backends.jl")
    include("test_allocations.jl")
    include("test_rmsd.jl")
    include("test_driver.jl")
    include("test_rotations.jl")
    include("test_shape.jl")
    include("test_pdb.jl")
    include("test_physics.jl")
    include("test_validate.jl")
    include("test_scoring.jl")
    include("test_fibre.jl")
    include("test_masking.jl")
    include("test_multimer.jl")
    include("test_constraints.jl")
    include("test_robustness.jl")
    include("test_vram.jl")

    include("test_gpu.jl")
end
