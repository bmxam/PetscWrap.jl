
function Base.:*(x::Vec, alpha::Number)
    y = VecCopy(x)
    scale!(y, alpha)
    return y
end
Base.:*(alpha::Number, vec::Vec) = vec * alpha

function assemble!(vec::Vec)
    assemblyBegin(vec)
    assemblyEnd(vec)
end

"""
    create_vector(
        comm::MPI.Comm = MPI.COMM_WORLD;
        nrows_loc = PETSC_DECIDE,
        nrows_glo = PETSC_DECIDE,
        auto_setup = false,
    )

Create a `Vec` vector of global size `(nrows_glo)`.
"""
function create_vector(
    comm::MPI.Comm = MPI.COMM_WORLD;
    nrows_loc = PETSC_DECIDE,
    nrows_glo = PETSC_DECIDE,
    auto_setup = false,
)
    vec = create(Vec, comm)
    setSizes(vec, nrows_loc, nrows_glo)

    if auto_setup
        set_from_options!(vec)
        set_up!(vec)
    end

    return vec
end

duplicate(vec::Vec, n::Int) = ntuple(i -> duplicate(vec), n)

"""
    get_range(vec::Vec)

Wrapper to `VecGetOwnershipRange`

However, the result `(rstart, rend)` is such that `vec[rstart:rend]` are the rows handled by the local processor.
This is different from the default `PETSc.VecGetOwnershipRange` result where the indexing starts at zero and where
`rend-1` is last row handled by the local processor.
"""
function get_range(vec::Vec)
    rstart, rend = getOwnershipRange(vec)
    return (rstart + 1, rend)
end

"""
    get_urange(vec::Vec)

Provide a `UnitRange` from the method `get_range`.
"""
function get_urange(vec::Vec)
    rstart, rend = getOwnershipRange(vec)
    return (rstart + 1):rend
end

Base.ndims(::Type{Vec}) = 1

const scale! = scale

"""
    Base.setindex!(vec::Vec, value::Number, row::Integer)

`row` must be in [1,size(vec)], i.e indexing starts at 1 (Julia).

# Implementation

For some unkwnown reason, calling `VecSetValue` fails.
"""
function Base.setindex!(vec::Vec, value::Number, row::Integer)
    setValues(vec, PetscInt[row .- 1], PetscScalar[value], INSERT_VALUES)
end

# This is stupid but I don't know how to do better yet
function Base.setindex!(vec::Vec, values, rows)
    setValues(vec, collect(rows .- 1), values, INSERT_VALUES)
end

set_global_size!(vec::Vec, nrows) = setSizes(vec, PETSC_DECIDE, nrows)

function set_local_to_global!(vec::Vec, lid2gid::Vector{Integer})
    mapping = create(ISLocalToGlobalMapping, lid2gid)
    setLocalToGlobalMapping(vec, mapping)
    destroy(mapping)
end

set_local_size!(vec::Vec, nrows) = setSizes(vec, nrows, PETSC_DECIDE)

"""
    set_values!(vec::Vec, rows, values, mode::InsertMode = INSERT_VALUES)
    set_values!(vec::Vec, values)

Wrapper to `VecSetValues`, using julia 1-based indexing.
"""
function set_values!(vec::Vec, rows, values, mode::InsertMode = INSERT_VALUES)
    setValues(vec, rows .- 1, values, mode)
end

function set_values!(vec::Vec, values)
    setValues(vec, collect(get_urange(vec) .- 1), values, INSERT_VALUES)
end

Base.show(::IO, vec::Vec) = view(vec)

"""
    vec2array(vec::Vec)

Convert a `Vec` into a Julia `Array`. Allocation is involved in the process since the `Vec`
allocated by PETSC is copied into a freshly allocated array. If you prefer not to allocate memory,
use `VectGetArray` and `VecRestoreArray`
"""
function vec2array(vec::Vec)
    arrayFromC, array_ref = getArray(vec)
    array = copy(arrayFromC)
    restoreArray(vec, array_ref)
    return array
end

"""
    vec2file(vec::Vec, filename::String, format::PetscViewerFormat = PETSC_VIEWER_ASCII_CSV, type::String = "ascii")

Write a Vec to a file.
"""
function vec2file(
    vec::Vec,
    filename::String,
    format::PetscViewerFormat = PETSC_VIEWER_ASCII_CSV,
    type::String = "ascii",
)
    viewer = PetscViewer(vec.comm, filename, format, type)
    view(vec, viewer)
    destroy!(viewer)
end

# Discutable choice(s)
Base.length(vec::Vec) = getLocalSize(vec)
Base.size(vec::Vec) = (length(vec),)
