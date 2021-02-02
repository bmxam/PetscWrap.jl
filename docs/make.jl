push!(LOAD_PATH,"../src/")

using JuliaPetsc
using Documenter
using Literate

# Generate examples
example_src = joinpath(@__DIR__,"..","example")
example_dir = joinpath(@__DIR__,"src","example")
Sys.rm(example_dir; recursive=true, force=true)
Literate.markdown(joinpath(example_src, "example1.jl"), example_dir; documenter = false, execute = false) # documenter = false to avoid Documenter to execute cells
Literate.markdown(joinpath(example_src, "example2.jl"), example_dir; documenter = false, execute = false) # documenter = false to avoid Documenter to execute cells

makedocs(;
    modules=[JuliaPetsc],
    authors="bmxam",
    sitename="JuliaPetsc.jl",
    clean=true,doctest=false,
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://github.com/bmxam/JuliaPetsc.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Examples" => Any[
            "example/example1.md",
            "example/example2.md",
        ],
        "API Reference" => Any[
            "api/init.md",
            "api/vec.md",
            "api/mat.md",
            "api/ksp.md",
        ]
    ],
)

deploydocs(;
    repo="github.com:bmxam/JuliaPetsc.jl.git",
    push_preview = true
)
