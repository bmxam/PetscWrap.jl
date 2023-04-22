using PetscWrap
using MPI
using Test
using LinearAlgebra

# from :
# https://discourse.julialang.org/t/what-general-purpose-commands-do-you-usually-end-up-adding-to-your-projects/4889
@generated function compare_struct(x, y)
    if !isempty(fieldnames(x)) && x == y
        mapreduce(n -> :(x.$n == y.$n), (a, b) -> :($a && $b), fieldnames(x))
    else
        :(x == y)
    end
end

"""
Custom way to "include" a file to print infos.
"""
function custom_include(path)
    filename = split(path, "/")[end]
    print("Running test file " * filename * "...")
    include(path)
    println("done.")
end

MPI.Init()
PetscInitialize()

error("Bug : tests are not running anymore when using `test`")
@testset "PetscWrap.jl" begin
    custom_include("./linear_system.jl")
    custom_include("./linear_system_fancy.jl")
    custom_include("./misc.jl")
end

PetscFinalize()
MPI.Finalize()
