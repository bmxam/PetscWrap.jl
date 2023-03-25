const CVec = Ptr{Cvoid}

struct PetscVec <: AbstractVector{PetscScalar}
    ptr::Ref{CVec}
    comm::MPI.Comm

    PetscVec(comm::MPI.Comm) = new(Ref{Ptr{Cvoid}}(), comm)
end

# allows us to pass PetscVec objects directly into CVec ccall signatures
Base.cconvert(::Type{CVec}, vec::PetscVec) = vec.ptr[]

"""
    VecAssemblyBegin(vec::PetscVec)

Wrapper to VecAssemblyBegin
"""
function VecAssemblyBegin(vec::PetscVec)
    error = ccall((:VecAssemblyBegin, libpetsc), PetscErrorCode, (CVec,), vec)
    @assert iszero(error)
end

"""
    VecAssemblyEnd(vec::PetscVec)

Wrapper to VecAssemblyEnd
"""
function VecAssemblyEnd(vec::PetscVec)
    error = ccall((:VecAssemblyEnd, libpetsc), PetscErrorCode, (CVec,), vec)
    @assert iszero(error)
end

"""
    VecCopy(x::PetscVec, y::PetscVec)

Wrapper to `VecCopy`.

https://petsc.org/main/docs/manualpages/Vec/VecCopy.html
"""
function VecCopy(x::PetscVec, y::PetscVec)
    error = ccall((:VecCopy, libpetsc), PetscErrorCode, (CVec, CVec), x, y)
    @assert iszero(error)
end

"""
    VecCopy(x::PetscVec)

Wrapper to `VecCopy`, expect that `y` is first obtained by `VecDuplicate`
"""
function VecCopy(x::PetscVec)
    y = VecDuplicate(x)
    VecCopy(x, y)
    return y
end

"""
    VecCreate(comm::MPI.Comm, vec::PetscVec)

Wrapper to VecCreate
"""
function VecCreate(comm::MPI.Comm=MPI.COMM_WORLD)
    vec = PetscVec(comm)
    error = ccall((:VecCreate, libpetsc), PetscErrorCode, (MPI.MPI_Comm, Ptr{CVec}), comm, vec.ptr)
    @assert iszero(error)
    return vec
end

"""
VecCreateGhost(comm::MPI.Comm, n::PetscInt, N::PetscInt, nghost::PetscInt, ghosts::Vector{PetscInt})

Wrapper to VecCreateGhost


# Arguments
* comm - the MPI communicator to use
* n - local vector length
* N - global vector length (or PETSC_DECIDE to have calculated if n is given)
* nghost - number of local ghost points
* ghosts - global indices of ghost points, these do not need to be in increasing order (sorted)

"""
function VecCreateGhost(
    comm::MPI.Comm,
    n::PetscInt, N::PetscInt,
    nghost::PetscInt,
    ghosts::Vector{PetscInt}
)
    vec = PetscVec(comm)

    error = ccall(
        (:VecCreateGhost, libpetsc),
        PetscErrorCode,
        (MPI.MPI_Comm, PetscInt, PetscInt, PetscInt, Ptr{PetscInt}, Ptr{CVec}),
        comm, n, N, nghost, ghosts, vec.ptr
    )
    @assert iszero(error)

    return vec
end

"""
    VecDestroy(vec::PetscVec)

Wrapper to VecDestroy
"""
function VecDestroy(vec::PetscVec)
    error = ccall((:VecDestroy, libpetsc),
        PetscErrorCode,
        (Ptr{CVec},),
        vec.ptr)
    @assert iszero(error)
end

"""
    VecDuplicate(vec::PetscVec)

Wrapper for VecDuplicate, except that it returns the new vector instead of taking it as an input.

https://petsc.org/main/docs/manualpages/Vec/VecDuplicate.html
"""
function VecDuplicate(vec::PetscVec)
    x = PetscVec(vec.comm)
    error = ccall((:VecDuplicate, libpetsc), PetscErrorCode, (CVec, Ptr{CVec}), vec, x.ptr)
    @assert iszero(error)

    return x
end

"""
    VecGetArray(vec::PetscVec, own = false)

Wrapper for VecGetArray.

# Warning
I am not confortable at all with memory management, both on the C side and on the Julia side. Use
this at you own risk.

According to Julia documentation, `own` optionally specifies whether Julia should take ownership
of the memory, calling free on the pointer when the array is no longer referenced."

"""
function VecGetArray(vec::PetscVec, own=false)
    # Get array pointer
    array_ref = Ref{Ptr{PetscScalar}}()
    error = ccall((:VecGetArray, libpetsc), PetscErrorCode, (CVec, Ref{Ptr{PetscScalar}}), vec, array_ref)
    @assert iszero(error)

    # Get array size
    rstart, rend = VecGetOwnershipRange(vec)
    n = rend - rstart # this is not `rend - rstart + 1` because of VecGetOwnershipRange convention

    array = unsafe_wrap(Array, array_ref[], n; own=own)

    return array, array_ref
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

"""
    VecGetOwnershipRange(vec::PetscVec)

Wrapper to `VecGetOwnershipRange`

The result `(rstart, rend)` is a Tuple indicating the rows handled by the local processor.

# Warning
`PETSc` indexing starts at zero (so `rstart` may be zero) and `rend-1` is the last row
handled by the local processor.
"""
function VecGetOwnershipRange(vec::PetscVec)
    rstart = Ref{PetscInt}(0)
    rend = Ref{PetscInt}(0)

    error = ccall((:VecGetOwnershipRange, libpetsc), PetscErrorCode, (CVec, Ref{PetscInt}, Ref{PetscInt}), vec, rstart, rend)
    @assert iszero(error)

    return rstart[], rend[]
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
    VecGetValues(vec::PetscVec, ni::PetscInt, ix::Vector{PetscInt})

Wrapper for VecGetValues

# Warning
I am not confortable at all with memory management, both on the C side and on the Julia side. Use
this at you own risk.

According to Julia documentation, `own` optionally specifies whether Julia should take ownership
of the memory, calling free on the pointer when the array is no longer referenced."
"""
function VecGetValues(vec::PetscVec, ni::PetscInt, ix::Vector{PetscInt}, own=false)
    # Get array pointer
    array_ref = Ref{Ptr{PetscScalar}}()
    error = ccall(
        (:VecGetValues, libpetsc),
        PetscErrorCode,
        (CVec, PetscInt, Ptr{PetscInt}, Ref{Ptr{PetscScalar}}),
        vec, ni, ix, array_ref
    )
    @assert iszero(error)

    array = unsafe_wrap(Array, array_ref[], ni; own=own)

    return array, array_ref
end

"""
    VecGhostGetLocalForm(vecGlo::PetscVec)

Wrapper for VecGhostGetLocalForm
"""
function VecGhostGetLocalForm(vecGlo::PetscVec)
    vecLoc = PetscVec(vecGlo.comm)

    error = ccall(
        (:VecGhostGetLocalForm, libpetsc),
        PetscErrorCode,
        (CVec, Ptr{CVec}),
        vecGlo, vecLoc.ptr
    )
    @assert iszero(error)

    return vecLoc
end

"""
    VecGhostUpdateBegin(vec::PetscVec, insertMode::InsertMode, scatterMode::ScatterMode)

Wrapper for VecGhostUpdateBegin
"""
function VecGhostUpdateBegin(vec::PetscVec, insertMode::InsertMode, scatterMode::ScatterMode)
    error = ccall(
        (:VecGhostUpdateBegin, libpetsc),
        PetscErrorCode,
        (CVec, InsertMode, ScatterMode),
        vec, insertMode, scatterMode
    )
    @assert iszero(error)
end

"""
    VecGhostUpdateEnd(vec::PetscVec, insertMode::InsertMode, scatterMode::ScatterMode)

Wrapper for VecGhostUpdateEnd
"""
function VecGhostUpdateEnd(vec::PetscVec, insertMode::InsertMode, scatterMode::ScatterMode)
    error = ccall(
        (:VecGhostUpdateEnd, libpetsc),
        PetscErrorCode,
        (CVec, InsertMode, ScatterMode),
        vec, insertMode, scatterMode
    )
    @assert iszero(error)
end

"""
    VecGhostRestoreLocalForm(vecGlo::PetscVec, vecLoc::PetscVec)

Wrapper for VecGhostRestoreLocalForm
"""
function VecGhostRestoreLocalForm(vecGlo::PetscVec, vecLoc::PetscVec)
    error = ccall(
        (:VecGhostRestoreLocalForm, libpetsc),
        PetscErrorCode,
        (CVec, Ptr{CVec}),
        vecGlo, vecLoc.ptr
    )
    @assert iszero(error)
end

"""
    VecRestoreArray(vec::PetscVec, array_ref)

Wrapper for VecRestoreArray. `array_ref` is obtained from `VecGetArray`
"""
function VecRestoreArray(vec::PetscVec, array_ref)
    error = ccall((:VecRestoreArray, libpetsc), PetscErrorCode, (CVec, Ref{Ptr{PetscScalar}}), vec, array_ref)
    @assert iszero(error)
end

"""
    VecScale(vec::PetscVec, alpha::PetscScalar)

Wrapper for  `VecScale`

https://petsc.org/main/docs/manualpages/Vec/VecScale.html
"""
function VecScale(vec::PetscVec, alpha::PetscScalar)
    error = ccall((:VecScale, libpetsc), PetscErrorCode, (CVec, PetscScalar), vec, alpha)
    @assert iszero(error)
end

VecScale(vec::PetscVec, alpha::Number) = VecScale(vec, PetscScalar(alpha))

"""
    VecSetFromOptions(vec::PetscVec)

Wrapper to VecSetFromOptions
"""
function VecSetFromOptions(vec::PetscVec)
    error = ccall((:VecSetFromOptions, libpetsc), PetscErrorCode, (CVec,), vec)
    @assert iszero(error)
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

"""
    VecSetUp(vec::PetscVec)

Wrapper to VecSetUp
"""
function VecSetUp(vec::PetscVec)
    error = ccall((:VecSetUp, libpetsc), PetscErrorCode, (CVec,), vec)
    @assert iszero(error)
end

"""
    VecSetValue(vec::PetscVec, i::PetscInt, v::PetscScalar, mode::InsertMode = INSERT_VALUES)

Wrapper to `VecSetValue`. Indexing starts at 0 (as in PETSc).

# Implementation
For an unknow reason, calling PETSc.VecSetValue leads to an "undefined symbol: VecSetValue" error.
So this wrapper directly call VecSetValues (anyway, this is what is done in PETSc...)
"""
function VecSetValue(vec::PetscVec, i::PetscInt, v::PetscScalar, mode::InsertMode=INSERT_VALUES)
    VecSetValues(vec, PetscIntOne, [i], [v], mode)
end

function VecSetValue(vec::PetscVec, i, v, mode::InsertMode=INSERT_VALUES)
    VecSetValue(vec, PetscInt(i), PetscScalar(v), mode)
end

"""
    VecSetValues(vec::PetscVec, nI::PetscInt, I::Vector{PetscInt}, V::Array{PetscScalar}, mode::InsertMode = INSERT_VALUES)

Wrapper to `VecSetValues`. Indexing starts at 0 (as in PETSc)
"""
function VecSetValues(vec::PetscVec, nI::PetscInt, I::Vector{PetscInt}, V::Array{PetscScalar}, mode::InsertMode=INSERT_VALUES)
    error = ccall((:VecSetValues, libpetsc), PetscErrorCode,
        (CVec, PetscInt, Ptr{PetscInt}, Ptr{PetscScalar}, InsertMode),
        vec, nI, I, V, mode)
    @assert iszero(error)
end

function VecSetValues(vec::PetscVec, I::Vector{PetscInt}, V::Array{PetscScalar}, mode::InsertMode=INSERT_VALUES)
    VecSetValues(vec, PetscInt(length(I)), I, V, mode)
end

function VecSetValues(vec::PetscVec, I, V, mode::InsertMode=INSERT_VALUES)
    VecSetValues(vec, PetscInt.(I), PetscScalar.(V), mode)
end

"""
    VecView(vec::PetscVec, viewer::PetscViewer = PetscViewerStdWorld())

Wrapper to `VecView`.
"""
function VecView(vec::PetscVec, viewer::PetscViewer=PetscViewerStdWorld())
    error = ccall((:VecView, libpetsc), PetscErrorCode, (CVec, CViewer), vec, viewer)
    @assert iszero(error)
end

