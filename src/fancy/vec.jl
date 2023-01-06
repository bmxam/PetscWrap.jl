
function Base.:*(x::PetscVec, alpha::Number)
    y = VecCopy(x)
    scale!(y, alpha)
    return y
end
Base.:*(alpha::Number, vec::PetscVec) = vec * alpha

function assemble!(vec::PetscVec)
    VecAssemblyBegin(vec)
    VecAssemblyEnd(vec)
end

"""
    create_vector(nrows, nrows_loc = PETSC_DECIDE)

Create a `PetscVec` vector of global size `(nrows)`.
"""
function create_vector(nrows, nrows_loc=PETSC_DECIDE; auto_setup=false, comm::MPI.Comm=MPI.COMM_WORLD)
    vec = VecCreate(comm)
    VecSetSizes(vec::PetscVec, nrows_loc, nrows)

    if (auto_setup)
        set_from_options!(vec)
        set_up!(vec)
    end

    return vec
end

destroy!(vec::PetscVec) = VecDestroy(vec)

duplicate(vec::PetscVec) = VecDuplicate(vec)
duplicate(vec::PetscVec, n::Int) = ntuple(i -> VecDuplicate(vec), n)

"""
    get_range(vec::PetscVec)

Wrapper to `VecGetOwnershipRange`

However, the result `(rstart, rend)` is such that `vec[rstart:rend]` are the rows handled by the local processor.
This is different from the default `PETSc.VecGetOwnershipRange` result where the indexing starts at zero and where
`rend-1` is last row handled by the local processor.
"""
function get_range(vec::PetscVec)
    rstart, rend = VecGetOwnershipRange(vec)
    return (rstart + 1, rend)
end

"""
    get_urange(vec::PetscVec)

Provide a `UnitRange` from the method `get_range`.
"""
function get_urange(vec::PetscVec)
    rstart, rend = VecGetOwnershipRange(vec)
    return rstart+1:rend
end

Base.ndims(::Type{PetscVec}) = 1

scale!(vec::PetscVec, alpha::Number) = VecScale(vec, alpha)

"""
    Base.setindex!(vec::PetscVec, value::Number, row::Integer)

`row` must be in [1,size(vec)], i.e indexing starts at 1 (Julia).

# Implementation
For some unkwnown reason, calling `VecSetValue` fails.
"""
function Base.setindex!(vec::PetscVec, value::Number, row::Integer)
    VecSetValues(vec, PetscInt[row.-1], PetscScalar[value], INSERT_VALUES)
end

# This is stupid but I don't know how to do better yet
Base.setindex!(vec::PetscVec, values, rows) = VecSetValues(vec, collect(rows .- 1), values, INSERT_VALUES)

set_from_options!(vec::PetscVec) = VecSetFromOptions(vec)

set_global_size!(vec::PetscVec, nrows) = VecSetSizes(vec, PETSC_DECIDE, nrows)
set_local_size!(vec::PetscVec, nrows) = VecSetSizes(vec, nrows, PETSC_DECIDE)

set_up!(vec::PetscVec) = VecSetUp(vec)

set_values!(vec::PetscVec, values) = VecSetValues(vec, collect(get_urange(vec) .- 1), values, INSERT_VALUES)

Base.show(::IO, vec::PetscVec) = VecView(vec)

"""
Wrapper to `VecSetValues`, using julia 1-based indexing.
"""
function set_values!(vec::PetscVec, rows::Vector{PetscInt}, values::Vector{PetscScalar}, mode::InsertMode=INSERT_VALUES)
    VecSetValues(vec, rows .- PetscIntOne, values, mode)
end

"""
    vec2array(vec::PetscVec)

Convert a `PetscVec` into a Julia `Array`. Allocation is involved in the process since the `PetscVec`
allocated by PETSC is copied into a freshly allocated array. If you prefer not to allocate memory,
use `VectGetArray` and `VecRestoreArray`
"""
function vec2array(vec::PetscVec)
    arrayFromC, array_ref = VecGetArray(vec)
    array = copy(arrayFromC)
    VecRestoreArray(vec, array_ref)
    return array
end

"""
    vec2file(vec::PetscVec, filename::String, format::PetscViewerFormat = PETSC_VIEWER_ASCII_CSV, type::String = "ascii")

Write a PetscVec to a file.
"""
function vec2file(vec::PetscVec, filename::String, format::PetscViewerFormat=PETSC_VIEWER_ASCII_CSV, type::String="ascii")
    viewer = PetscViewer(vec.comm, filename, format, type)
    VecView(vec, viewer)
    destroy!(viewer)
end

# Discutable choice(s)
Base.length(vec::PetscVec) = VecGetLocalSize(vec)
Base.size(vec::PetscVec) = (length(vec),)