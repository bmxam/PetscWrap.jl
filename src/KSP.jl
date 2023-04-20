const CKSP = Ptr{Cvoid}

mutable struct KSP
    ptr::CKSP
    comm::MPI.Comm

    KSP(comm::MPI.Comm) = new(CKSP(), comm)
end

Base.unsafe_convert(::Type{CKSP}, x::KSP) = x.ptr
Base.unsafe_convert(::Type{Ptr{CKSP}}, x::KSP) = Ptr{CKSP}(pointer_from_objref(x))

"""
    create(::Type{KSP}, comm::MPI.Comm = MPI.COMM_WORLD; add_finalizer = true)

Wrapper for `KSPCreate`
https://petsc.org/release/manualpages/KSP/KSPCreate/
"""
function create(::Type{KSP}, comm::MPI.Comm = MPI.COMM_WORLD; add_finalizer = true)
    ksp = KSP(comm)
    error =
        ccall((:KSPCreate, libpetsc), PetscErrorCode, (MPI.MPI_Comm, Ptr{CKSP}), comm, ksp)
    @assert iszero(error)

    _NREFS[] += 1
    add_finalizer && finalizer(destroy, ksp)

    return ksp
end

"""
    destroy(ksp::KSP)

Wrapper to `KSPDestroy`
https://petsc.org/release/manualpages/KSP/KSPDestroy/
"""
function destroy(ksp::KSP)
    error = ccall((:KSPDestroy, libpetsc), PetscErrorCode, (Ptr{CKSP},), ksp)
    @assert iszero(error)

    _NREFS[] -= 1
end

"""
    setFromOptions(ksp::KSP)

Wrapper to `KSPSetFromOptions`
https://petsc.org/release/manualpages/KSP/KSPSetFromOptions/
"""
function setFromOptions(ksp::KSP)
    error = ccall((:KSPSetFromOptions, libpetsc), PetscErrorCode, (CKSP,), ksp)
    @assert iszero(error)
end

"""
    setOperators(ksp::KSP, Amat::Mat, Pmat::Mat)

Wrapper for `KSPSetOperators``
https://petsc.org/release/manualpages/KSP/KSPSetOperators/
"""
function setOperators(ksp::KSP, Amat::Mat, Pmat::Mat)
    error = ccall(
        (:KSPSetOperators, libpetsc),
        PetscErrorCode,
        (CKSP, CMat, CMat),
        ksp,
        Amat,
        Pmat,
    )
    @assert iszero(error)
end

"""
    KSPSetUp(ksp::KSP)

Wrapper to `KSPSetUp`
https://petsc.org/release/manualpages/KSP/KSPSetUp/
"""
function setUp(ksp::KSP)
    error = ccall((:KSPSetUp, libpetsc), PetscErrorCode, (CKSP,), ksp)
    @assert iszero(error)
end

"""
    solve(ksp::KSP, b::Vec, x::Vec)

Wrapper for `KSPSolve`
https://petsc.org/release/manualpages/KSP/KSPSolve/
"""
function solve(ksp::KSP, b::Vec, x::Vec)
    error = ccall((:KSPSolve, libpetsc), PetscErrorCode, (CKSP, CVec, CVec), ksp, b, x)
    @assert iszero(error)
end
