const CVec = Ptr{Cvoid}

struct Vec
    ptr::Ref{CVec}
    comm::MPI.Comm

    Vec(comm::MPI.Comm) = new(Ref{Ptr{Cvoid}}(), comm)
end

# allows us to pass Vec objects directly into CVec ccall signatures
Base.cconvert(::Type{CVec}, vec::Vec) = vec.ptr[]

"""
    assemblyBegin(vec::Vec)

Wrapper to `VecAssemblyBegin`
https://petsc.org/release/docs/manualpages/Vec/VecAssemblyBegin/
"""
function assemblyBegin(vec::Vec)
    error = ccall((:VecAssemblyBegin, libpetsc), PetscErrorCode, (CVec,), vec)
    @assert iszero(error)
end

"""
    assemblyEnd(vec::Vec)

Wrapper to `VecAssemblyEnd`
https://petsc.org/release/docs/manualpages/Vec/VecAssemblyEnd/
"""
function assemblyEnd(vec::Vec)
    error = ccall((:VecAssemblyEnd, libpetsc), PetscErrorCode, (CVec,), vec)
    @assert iszero(error)
end

"""
    copy(x::Vec, y::Vec)
    copy(x::Vec)

Wrapper to `VecCopy`
https://petsc.org/release/docs/manualpages/Vec/VecCopy/

Not really sure if I should extend Base here...
"""
function Base.copy(x::Vec, y::Vec)
    error = ccall((:VecCopy, libpetsc), PetscErrorCode, (CVec, CVec), x, y)
    @assert iszero(error)
end

function Base.copy(x::Vec)
    y = duplicate(x)
    copy(x, y)
    return y
end

"""
    create(::Type{Vec},comm::MPI.Comm = MPI.COMM_WORLD)

Wrapper to `VecCreate`
https://petsc.org/release/docs/manualpages/Vec/VecCreate/
"""
function create(::Type{Vec}, comm::MPI.Comm = MPI.COMM_WORLD)
    vec = Vec(comm)
    error = ccall(
        (:VecCreate, libpetsc),
        PetscErrorCode,
        (MPI.MPI_Comm, Ptr{CVec}),
        comm,
        vec.ptr,
    )
    @assert iszero(error)
    return vec
end

"""
    destroy(v::Vec)

Wrapper to `VecDestroy`
https://petsc.org/release/docs/manualpages/Vec/VecDestroy/
"""
function destroy(v::Vec)
    error = ccall((:VecDestroy, libpetsc), PetscErrorCode, (Ptr{CVec},), v.ptr)
    @assert iszero(error)
end

function LinearAlgebra.dot(x::Vec, y::Vec)
    val = Ref{PetscScalar}()
    error = ccall((:VecDot, libpetsc), PetscErrorCode, (CVec, CVec, Ref{PetscScalar}), x, y, val)
    @assert iszero(error)

    return val[]
end

"""
    duplicate(v::Vec)

Wrapper for `VecDuplicate`
https://petsc.org/release/docs/manualpages/Vec/VecDuplicate/
"""
function duplicate(v::Vec)
    newv = Vec(v.comm)
    error = ccall((:VecDuplicate, libpetsc), PetscErrorCode, (CVec, Ptr{CVec}), v, newv.ptr)
    @assert iszero(error)

    return newv
end

"""
    getArray(x::Vec, own::Bool = false)

Wrapper for `VecGetArray`
https://petsc.org/release/docs/manualpages/Vec/VecGetArray/

# Warning

I am not confortable at all with memory management, both on the C side and on the Julia side. Use
this at you own risk.

According to Julia documentation, `own` optionally specifies whether Julia should take ownership
of the memory, calling free on the pointer when the array is no longer referenced."
"""
function getArray(x::Vec, own::Bool = false)
    # Get array pointer
    array_ref = Ref{Ptr{PetscScalar}}()
    error = ccall(
        (:VecGetArray, libpetsc),
        PetscErrorCode,
        (CVec, Ref{Ptr{PetscScalar}}),
        x,
        array_ref,
    )
    @assert iszero(error)

    # Get array size
    rstart, rend = getOwnershipRange(x)
    n = rend - rstart # this is not `rend - rstart + 1` because of VecGetOwnershipRange convention

    array = unsafe_wrap(Array, array_ref[], n; own = own)

    return array, array_ref
end

"""
    getLocalSize(vec::Vec)

Wrapper for `VecGetLocalSize`
https://petsc.org/release/docs/manualpages/Vec/VecGetLocalSize/
"""
function getLocalSize(x::Vec)
    n = Ref{PetscInt}()

    error = ccall((:VecGetLocalSize, libpetsc), PetscErrorCode, (CVec, Ref{PetscInt}), x, n)
    @assert iszero(error)

    return n[]
end

"""
    getOwnershipRange(x::Vec)

Wrapper to `VecGetOwnershipRange`
https://petsc.org/release/docs/manualpages/Vec/VecGetOwnershipRange/

The result `(rstart, rend)` is a Tuple indicating the rows handled by the local processor.

# Warning

`PETSc` indexing starts at zero (so `rstart` may be zero) and `rend-1` is the last row
handled by the local processor.
"""
function getOwnershipRange(x::Vec)
    rstart = Ref{PetscInt}(0)
    rend = Ref{PetscInt}(0)

    error = ccall(
        (:VecGetOwnershipRange, libpetsc),
        PetscErrorCode,
        (CVec, Ref{PetscInt}, Ref{PetscInt}),
        x,
        rstart,
        rend,
    )
    @assert iszero(error)

    return rstart[], rend[]
end

"""
    getSize(x::Vec)

Wrapper for `VecGetSize`
https://petsc.org/release/docs/manualpages/Vec/VecGetSize/
"""
function getSize(x::Vec)
    n = Ref{PetscInt}()

    error = ccall((:VecGetSize, libpetsc), PetscErrorCode, (CVec, Ref{PetscInt}), x, n)
    @assert iszero(error)

    return n[]
end

"""
    getValues(x::Vec, ni::PetscInt, ix::Vector{PetscInt})
    getValues(x::Vec, ix::Vector{PetscInt})
    getValues(x::Vec, ix::Vector{I}) where {I<:Integer}

Wrapper for `VecGetValues`
https://petsc.org/release/docs/manualpages/Vec/VecGetValues/
"""
function getValues(x::Vec, ni::PetscInt, ix::Vector{PetscInt})
    y = zeros(PetscScalar, ni)

    error = ccall(
        (:VecGetValues, libpetsc),
        PetscErrorCode,
        (
            CVec,
            PetscInt,
            Ptr{PetscInt},
            Ptr{PetscScalar},
        ),
        x,
        ni,
        ix,
        y,
    )
    @assert iszero(error)

    return y
end

getValues(x::Vec, ix::Vector{PetscInt}) = getValues(x, PetscInt(length(ix)), ix)

function getValues(x::Vec, ix::Vector{I}) where {I<:Integer}
    return getValues(x, PetscInt.(ix))
end

"""
    restoreArray(x::Vec, array_ref)

Wrapper for `VecRestoreArray`. `array_ref` is obtained from `VecGetArray`
https://petsc.org/release/docs/manualpages/Vec/VecRestoreArray/
"""
function restoreArray(x::Vec, array_ref)
    error = ccall(
        (:VecRestoreArray, libpetsc),
        PetscErrorCode,
        (CVec, Ref{Ptr{PetscScalar}}),
        x,
        array_ref,
    )
    @assert iszero(error)
end

"""
    scale(x::Vec, alpha::PetscScalar)
    scale(x::Vec, alpha::Number)

Wrapper for  `VecScale`
https://petsc.org/release/docs/manualpages/Vec/VecScale/
"""
function scale(x::Vec, alpha::PetscScalar)
    error = ccall((:VecScale, libpetsc), PetscErrorCode, (CVec, PetscScalar), x, alpha)
    @assert iszero(error)
end

scale(x::Vec, alpha::Number) = scale(x, PetscScalar(alpha))

"""
    setFromOptions(vec::Vec)

Wrapper to `VecSetFromOptions`
https://petsc.org/release/docs/manualpages/Vec/VecSetFromOptions/
"""
function setFromOptions(vec::Vec)
    error = ccall((:VecSetFromOptions, libpetsc), PetscErrorCode, (CVec,), vec)
    @assert iszero(error)
end

"""
    setLocalToGlobalMapping(x::Vec, mapping::ISLocalToGlobalMapping)

Wrapper to `VecSetLocalToGlobalMapping`
https://petsc.org/release/docs/manualpages/Vec/VecSetLocalToGlobalMapping/

0-based indexing
"""
function setLocalToGlobalMapping(x::Vec, mapping::ISLocalToGlobalMapping)
    error = ccall(
        (:VecSetLocalToGlobalMapping, libpetsc),
        PetscErrorCode,
        (CVec, CISLocalToGlobalMapping),
        x,
        mapping,
    )
    @assert iszero(error)
end

"""
    setSizes(v::Vec, n::PetscInt, N::PetscInt)
    setSizes(v::Vec, n::Integer, N::Integer)

Wrapper to `VecSetSizes`
https://petsc.org/release/docs/manualpages/Vec/VecSetSizes/
"""
function setSizes(v::Vec, n::PetscInt, N::PetscInt)
    nr_loc = PetscInt(n)
    nr_glo = PetscInt(N)
    error = ccall(
        (:VecSetSizes, libpetsc),
        PetscErrorCode,
        (CVec, PetscInt, PetscInt),
        v,
        nr_loc,
        nr_glo,
    )
    @assert iszero(error)
end

setSizes(v::Vec, n::Integer, N::Integer) = setSizes(v, PetscInt(n), PetscInt(N))

"""
    setUp(vec::Vec)

Wrapper to `VecSetUp`
https://petsc.org/release/docs/manualpages/Vec/VecSetUp/
"""
function setUp(v::Vec)
    error = ccall((:VecSetUp, libpetsc), PetscErrorCode, (CVec,), v)
    @assert iszero(error)
end

"""
    setValue(v::Vec, row::PetscInt, value::PetscScalar, mode::InsertMode)
    setValue(v::Vec, row, value, mode::InsertMode = INSERT_VALUES)

Wrapper to `setValue`. Indexing starts at 0 (as in PETSc).
https://petsc.org/release/docs/manualpages/Vec/VecSetValue/

# Implementation

For an unknow reason, calling PETSc.VecSetValue leads to an "undefined symbol: VecSetValue" error.
So this wrapper directly call VecSetValues (anyway, this is what is done in PETSc...)
"""
function setValue(v::Vec, row::PetscInt, value::PetscScalar, mode::InsertMode)
    setValues(v, PetscIntOne, [row], [value], mode)
end

function setValue(v::Vec, row, value, mode::InsertMode = INSERT_VALUES)
    setValue(v, PetscInt(row), PetscScalar(value), mode)
end

"""
    setValues(
        x::Vec,
        ni::PetscInt,
        ix::Vector{PetscInt},
        y::Vector{PetscScalar},
        iora::InsertMode,
    )

    setValues(
        x::Vec,
        I::Vector{PetscInt},
        V::Vector{PetscScalar},
        mode::InsertMode = INSERT_VALUES,
    )
    setValues(x::Vec, I, V, mode::InsertMode = INSERT_VALUES)

Wrapper to `VecSetValues`. Indexing starts at 0 (as in PETSc)
https://petsc.org/release/docs/manualpages/Vec/VecSetValues/
"""
function setValues(
    x::Vec,
    ni::PetscInt,
    ix::Vector{PetscInt},
    y::Vector{PetscScalar},
    iora::InsertMode,
)
    error = ccall(
        (:VecSetValues, libpetsc),
        PetscErrorCode,
        (CVec, PetscInt, Ptr{PetscInt}, Ptr{PetscScalar}, InsertMode),
        x,
        ni,
        ix,
        y,
        iora,
    )
    @assert iszero(error)
end

function setValues(
    x::Vec,
    I::Vector{PetscInt},
    V::Vector{PetscScalar},
    mode::InsertMode = INSERT_VALUES,
)
    setValues(x, PetscInt(length(I)), I, V, mode)
end

function setValues(x::Vec, I, V, mode::InsertMode = INSERT_VALUES)
    setValues(x, PetscInt.(I), PetscScalar.(V), mode)
end

"""
    setValueLocal(v::Vec, row::PetscInt, value::PetscScalar, mode::InsertMode)
    setValueLocal(v::Vec, row, value, mode::InsertMode = INSERT_VALUES)

Wrapper to `setValueLocal`. Indexing starts at 0 (as in PETSc).
https://petsc.org/release/docs/manualpages/Vec/VecSetValueLocal/

# Implementation

For an unknow reason, calling PETSc.VecSetValue leads to an "undefined symbol: VecSetValue" error.
So this wrapper directly call VecSetValues (anyway, this is what is done in PETSc...)
"""
function setValueLocal(v::Vec, row::PetscInt, value::PetscScalar, mode::InsertMode)
    setValuesLocal(v, PetscIntOne, [row], [value], mode)
end

function setValueLocal(v::Vec, row, value, mode::InsertMode = INSERT_VALUES)
    setValueLocal(v, PetscInt(row), PetscScalar(value), mode)
end

"""
    setValuesLocal(
        x::Vec,
        ni::PetscInt,
        ix::Vector{PetscInt},
        y::Vector{PetscScalar},
        iora::InsertMode,
    )

    setValuesLocal(
        x::Vec,
        I::Vector{PetscInt},
        V::Vector{PetscScalar},
        mode::InsertMode = INSERT_VALUES,
    )
    setValues(x::Vec, I, V, mode::InsertMode = INSERT_VALUES)

Wrapper to `VecSetValues`. Indexing starts at 0 (as in PETSc)
https://petsc.org/release/docs/manualpages/Vec/VecSetValuesLocal/
"""
function setValuesLocal(
    x::Vec,
    ni::PetscInt,
    ix::Vector{PetscInt},
    y::Vector{PetscScalar},
    iora::InsertMode,
)
    error = ccall(
        (:VecSetValuesLocal, libpetsc),
        PetscErrorCode,
        (CVec, PetscInt, Ptr{PetscInt}, Ptr{PetscScalar}, InsertMode),
        x,
        ni,
        ix,
        y,
        iora,
    )
    @assert iszero(error)
end

function setValuesLocal(
    x::Vec,
    I::Vector{PetscInt},
    V::Vector{PetscScalar},
    mode::InsertMode = INSERT_VALUES,
)
    setValuesLocal(x, PetscInt(length(I)), I, V, mode)
end

function setValuesLocal(x::Vec, I, V, mode::InsertMode = INSERT_VALUES)
    setValuesLocal(x, PetscInt.(I), PetscScalar.(V), mode)
end

function Base.sum(x::Vec)
    s = Ref{PetscScalar}()
    error = ccall(
        (:VecSum, libpetsc),
        PetscErrorCode,
        (CVec, Ref{PetscScalar}),
        x,s,
    )
    @assert iszero(error)

    return s[]
end

"""
    view(vec::Vec, viewer::PetscViewer = StdWorld())

Wrapper to `VecView`
https://petsc.org/release/docs/manualpages/Vec/VecView/
"""
function view(vec::Vec, viewer::PetscViewer = StdWorld(vec.comm))
    error = ccall((:VecView, libpetsc), PetscErrorCode, (CVec, CViewer), vec, viewer)
    @assert iszero(error)
end
