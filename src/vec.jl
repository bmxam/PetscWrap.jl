const CVec = Ptr{Cvoid}

struct PetscVec
    ptr::Ref{CVec}

    PetscVec() = new(Ref{Ptr{Cvoid}}())
end

# allows us to pass PetscVec objects directly into CVec ccall signatures
Base.cconvert(::Type{CVec}, vec::PetscVec) = vec.ptr[]

"""
`row` must be in [1,size(vec)], i.e indexing starts at 1 (Julia).

# Implementation
For some unkwnown reason, calling `VecSetValue` fails.
"""
function Base.setindex!(vec::PetscVec, value::Number, row::Integer)
    VecSetValues(vec, PetscInt[row], PetscScalar[value], INSERT_VALUES)
end


# This is stupid but I don't know how to do better yet
Base.setindex!(vec::PetscVec, values, rows) = VecSetValues(vec, collect(rows), values, INSERT_VALUES)

Base.ndims(::Type{PetscVec}) = 1

"""
    Wrapper to VecSetValues.

    Indexing starts at 1 (Julia)
"""
function VecSetValues(vec::PetscVec, I::Vector{PetscInt}, V::Array{PetscScalar}, mode::InsertMode = INSERT_VALUES)
    nI = PetscInt(length(I))
    error = ccall((:VecSetValues, libpetsc), PetscErrorCode,
        (CVec, PetscInt, Ptr{PetscInt}, Ptr{PetscScalar}, InsertMode),
        vec, nI, I .- PetscIntOne, V, mode)
    @assert iszero(error)
end

function VecSetValues(vec::PetscVec, I, V, mode::InsertMode = INSERT_VALUES)
    VecSetValues(vec, PetscInt.(I), PetscScalar.(V), mode)
end


"""
    Wrapper to VecCreate
"""
function VecCreate(comm, vec::PetscVec)
    error = ccall((:VecCreate, libpetsc), PetscErrorCode, (MPI.MPI_Comm, Ptr{CVec}), comm, vec.ptr)
    @assert iszero(error)
end

function VecCreate(comm)
    vec = PetscVec()
    VecCreate(comm, vec)
    return vec
end

function VecCreate()
    vec = PetscVec()
    VecCreate(MPI.COMM_WORLD, vec)
    return vec
end

"""
    create_vector(nrows, nrows_loc = PETSC_DECIDE)

Create a `PetscVec` matrix of global size `(nrows)`.
"""
function create_vector(nrows, nrows_loc = PETSC_DECIDE)
    vec = VecCreate()
    VecSetSizes(vec::PetscVec, nrows_loc, nrows)
    return vec
end

"""
    VecSetSizes(vec::PetscVec, nrows_loc, nrows_glo)

Wrapper to VecSetSizes
"""
function VecSetSizes(vec::PetscVec, nrows_loc, nrows_glo)
    nr_loc = PetscInt(nrows_loc)
    nr_glo = PetscInt(nrows_glo)
    error = ccall((:VecSetSizes, libpetsc),
                PetscErrorCode,
                (CVec, PetscInt, PetscInt),
                vec, nr_loc, nr_glo
            )
    @assert iszero(error)
end

set_global_size!(vec::PetscVec, nrows) = VecSetSizes(vec, PETSC_DECIDE, nrows)
set_local_size!(vec::PetscVec, nrows) = VecSetSizes(vec, nrows, PETSC_DECIDE)

"""
    Wrapper to VecSetFromOptions
"""
function VecSetFromOptions(vec::PetscVec)
    error = ccall((:VecSetFromOptions, libpetsc), PetscErrorCode, (CVec,), vec)
    @assert iszero(error)
end

"""
    Wrapper to VecSetUp
"""
function VecSetUp(vec::PetscVec)
    error = ccall((:VecSetUp, libpetsc), PetscErrorCode, (CVec,), vec)
    @assert iszero(error)
end

"""
    Wrapper to VecGetOwnershipRange

However, the result `(rstart, rend)` is such that `mat[rstart:rend]` are the rows handled by the local processor.
This is different from the default `PETSc` result where the indexing starts at one and where `rend-1` is last row
handled by the local processor.
"""
function VecGetOwnershipRange(vec::PetscVec)
    rstart = Ref{PetscInt}(0)
    rend = Ref{PetscInt}(0)

    error = ccall((:VecGetOwnershipRange, libpetsc), PetscErrorCode, (CVec, Ref{PetscInt}, Ref{PetscInt}), vec, rstart, rend)
    @assert iszero(error)

    return rstart[] + 1, rend[]
end

"""
    VecGetSize(vec::PetscVec)

Wrapper for VecGetSize
"""
function VecGetSize(vec::PetscVec)
    n = Ref{PetscInt}()

    error = ccall((:VecGetSize, libpetsc), PetscErrorCode, (CVec, Ref{PetscInt}), vec, n)
    @assert iszero(error)

    return n[]
end

"""
    VecGetLocalSize(vec::PetscVec)

Wrapper for VecGetLocalSize
"""
function VecGetLocalSize(vec::PetscVec)
    n = Ref{PetscInt}()

    error = ccall((:VecGetLocalSize, libpetsc), PetscErrorCode, (CVec, Ref{PetscInt}), vec, n)
    @assert iszero(error)

    return n[]
end

# Discutable choice
Base.length(vec::PetscVec) = VecGetLocalSize(vec)
Base.size(vec::PetscVec) = (length(vec),)

"""
    Wrapper to VecAssemblyBegin
"""
function VecAssemblyBegin(vec::PetscVec)
    error = ccall((:VecAssemblyBegin, libpetsc), PetscErrorCode, (CVec,), vec)
    @assert iszero(error)
end

"""
    Wrapper to VecAssemblyEnd
"""
function VecAssemblyEnd(vec::PetscVec)
    error = ccall((:VecAssemblyEnd, libpetsc), PetscErrorCode, (CVec,), vec)
    @assert iszero(error)
end

function assemble!(vec::PetscVec)
    VecAssemblyBegin(vec)
    VecAssemblyEnd(vec)
end

"""
    VecDuplicate(vec::PetscVec)

Wrapper for VecDuplicate, except that it returns the new vector instead of taking it as an input.
"""
function VecDuplicate(vec::PetscVec)
    x = PetscVec()
    error = ccall((:VecDuplicate, libpetsc), PetscErrorCode, (CVec, Ptr{CVec}), vec, x.ptr)
    @assert iszero(error)

    return x
end

"""
    Wrapper for VecGetArray.

# Warning
I am not confortable at all with memory management, both on the C side and on the Julia side. Use
this at you own risk.

According to Julia documentation, "`own` optionally specifies whether Julia should take ownership
of the memory, calling free on the pointer when the array is no longer referenced."

"""
function VecGetArray(vec::PetscVec, own = false)
    # Get array pointer
    array_ref = Ref{Ptr{PetscScalar}}()
    error = ccall((:VecGetArray, libpetsc), PetscErrorCode, (CVec, Ref{Ptr{PetscScalar}}), vec, array_ref)
    @assert iszero(error)

    # Get array size
    rstart, rend = VecGetOwnershipRange(vec)
    n = rend - rstart + 1

    array = unsafe_wrap(Array, array_ref[], n; own)

    return array, array_ref
end

"""
    Wrapper for VecRestoreArray. `array_ref` is obtained from `VecGetArray`
"""
function VecRestoreArray(vec::PetscVec, array_ref)
    error = ccall((:VecRestoreArray, libpetsc), PetscErrorCode, (CVec, Ref{Ptr{PetscScalar}}), vec, array_ref)
    @assert iszero(error)
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
    Wrapper to VecView
"""
function VecView(vec::PetscVec, viewer::PetscViewer = C_NULL)
    error = ccall( (:VecView, libpetsc), PetscErrorCode, (CVec, PetscViewer), vec, viewer);
    @assert iszero(error)
end

"""
    Wrapper to VecDestroy
"""
function VecDestroy(vec::PetscVec)
    error = ccall( (:VecDestroy, libpetsc),
                    PetscErrorCode,
                    (Ptr{CVec},),
                    vec.ptr)
    @assert iszero(error)
end