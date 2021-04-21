using PetscWrap
using Test


# from :
# https://discourse.julialang.org/t/what-general-purpose-commands-do-you-usually-end-up-adding-to-your-projects/4889
@generated function compare_struct(x, y)
    if !isempty(fieldnames(x)) && x == y
        mapreduce(n -> :(x.$n == y.$n), (a,b)->:($a && $b), fieldnames(x))
    else
        :(x == y)
    end
end


@testset "PetscWrap.jl" begin
    include("./linear_system.jl")
    include("./linear_system_fancy.jl")
end
