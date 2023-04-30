
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
        autosetup = false,
    )

Create a `Vec` vector of global size `(nrows_glo)`.
"""
function create_vector(
    comm::MPI.Comm = MPI.COMM_WORLD;
    nrows_loc = PETSC_DECIDE,
    nrows_glo = PETSC_DECIDE,
    autosetup = false,
)
    vec = create(Vec, comm)
    setSizes(vec, nrows_loc, nrows_glo)

    if autosetup
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

"""
Wrapper to `ISSetLocalToGlobalMapping`

1-based indexing
"""
function set_local_to_global!(vec::Vec, lid2gid::Vector{I}) where {I<:Integer}
    mapping = create(ISLocalToGlobalMapping, vec.comm, lid2gid .- 1)
    setLocalToGlobalMapping(vec, mapping)
    destroy(mapping)
end

set_local_size!(vec::Vec, nrows) = setSizes(vec, nrows, PETSC_DECIDE)

"""
    set_value!(vec::Vec, row, value, mode::InsertMode = INSERT_VALUES)

Wrapper to `VecSetValue`, using julia 1-based indexing.
"""
function set_value!(vec::Vec, row, value, mode::InsertMode = INSERT_VALUES)
    setValue(vec, row - 1, value, mode)
end

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

"""
    set_value_local!(vec::Vec, row, value, mode::InsertMode = INSERT_VALUES)

Wrapper to `VecSetValueLocal`, using julia 1-based indexing.
"""
function set_value_local!(vec::Vec, row, value, mode::InsertMode = INSERT_VALUES)
    setValueLocal(vec, row - 1, value, mode)
end

"""
    set_values_local!(vec::Vec, rows, values, mode::InsertMode = INSERT_VALUES)
    set_values_local!(vec::Vec, values)

Wrapper to `VecSetValuesLocal`, using julia 1-based indexing.
"""
function set_values_local!(vec::Vec, rows, values, mode::InsertMode = INSERT_VALUES)
    setValuesLocal(vec, rows .- 1, values, mode)
end

function set_values_local!(vec::Vec, values)
    # Since we are using the "local" numbering, we set element "1" to "nloc"
    setValuesLocal(vec, collect(1:getLocalSize(vec)) .- 1, values, INSERT_VALUES)
end

Base.show(::IO, vec::Vec) = vecView(vec)

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
In construction
"""
function get_values!(y::AbstractVector{PetscScalar}, x::Vec)
    ni = length(y)
    rstart, rend = getOwnershipRange(x)
    ix = collect(rstart:(rend - 1))

    error = ccall(
        (:VecGetValues, libpetsc),
        PetscErrorCode,
        (CVec, PetscInt, Ptr{PetscInt}, Ptr{PetscScalar}),
        x,
        ni,
        ix,
        y,
    )
    @assert iszero(error)
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
    vecView(vec, viewer)
    destroy!(viewer)
end

# Discutable choice(s)
Base.length(vec::Vec) = getLocalSize(vec)
Base.size(vec::Vec) = (length(vec),)
