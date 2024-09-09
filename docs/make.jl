push!(LOAD_PATH, "../src/")

using PetscWrap
using Documenter
using Literate

# Generate examples
example_src = joinpath(@__DIR__, "..", "example")
example_dir = joinpath(@__DIR__, "src", "example")
Sys.rm(example_dir; recursive = true, force = true)
Literate.markdown(
    joinpath(example_src, "linear_system.jl"),
    example_dir;
    documenter = false,
    execute = false,
) # documenter = false to avoid Documenter to execute cells
Literate.markdown(
    joinpath(example_src, "linear_system_fancy.jl"),
    example_dir;
    documenter = false,
    execute = false,
) # documenter = false to avoid Documenter to execute cells

makedocs(;
    modules = [PetscWrap],
    authors = "bmxam",
    sitename = "PetscWrap.jl",
    clean = true,
    doctest = false,
    checkdocs = :none,
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://github.com/bmxam/PetscWrap.jl",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Examples" => Any["example/linear_system.md",],
        "Fancy examples" => Any["example/linear_system_fancy.md",],
        "API Reference" =>
            Any["api/init.md", "api/viewer.md", "api/vec.md", "api/mat.md", "api/ksp.md"],
        "API fancy" => "api/fancy/fancy.md",
    ],
)

deploydocs(; repo = "github.com/bmxam/PetscWrap.jl.git", push_preview = true)
