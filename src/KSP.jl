const CKSP = Ptr{Cvoid}

struct KSP
    ptr::Ref{CKSP}
    comm::MPI.Comm

    KSP(comm::MPI.Comm) = new(Ref{CKSP}(), comm)
end

# allows us to pass KSP objects directly into CKSP ccall signatures
Base.cconvert(::Type{CKSP}, ksp::KSP) = ksp.ptr[]

"""
    create(::Type{KSP}, comm::MPI.Comm=MPI.COMM_WORLD)

Wrapper for `KSPCreate`
https://petsc.org/release/docs/manualpages/KSP/KSPCreate/
"""
function create(::Type{KSP}, comm::MPI.Comm = MPI.COMM_WORLD)
    ksp = KSP(comm)
    error = ccall(
        (:KSPCreate, libpetsc),
        PetscErrorCode,
        (MPI.MPI_Comm, Ptr{CKSP}),
        comm,
        ksp.ptr,
    )
    @assert iszero(error)
    return ksp
end

"""
    destroy(ksp::KSP)

Wrapper to `KSPDestroy`
https://petsc.org/release/docs/manualpages/KSP/KSPDestroy/
"""
function destroy(ksp::KSP)
    error = ccall((:KSPDestroy, libpetsc), PetscErrorCode, (Ptr{CKSP},), ksp.ptr)
    @assert iszero(error)
end

"""
    setFromOptions(ksp::KSP)

Wrapper to `KSPSetFromOptions`
https://petsc.org/release/docs/manualpages/KSP/KSPSetFromOptions/
"""
function setFromOptions(ksp::KSP)
    error = ccall((:KSPSetFromOptions, libpetsc), PetscErrorCode, (CKSP,), ksp)
    @assert iszero(error)
end

"""
    setOperators(ksp::KSP, Amat::PetscMat, Pmat::PetscMat)

Wrapper for `KSPSetOperators``
https://petsc.org/release/docs/manualpages/KSP/KSPSetOperators/
"""
function setOperators(ksp::KSP, Amat::PetscMat, Pmat::PetscMat)
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
https://petsc.org/release/docs/manualpages/KSP/KSPSetUp/
"""
function setUp(ksp::KSP)
    error = ccall((:KSPSetUp, libpetsc), PetscErrorCode, (CKSP,), ksp)
    @assert iszero(error)
end

"""
    solve(ksp::KSP, b::PetscVec, x::PetscVec)

Wrapper for `KSPSolve`
https://petsc.org/release/docs/manualpages/KSP/KSPSolve/
"""
function solve(ksp::KSP, b::PetscVec, x::PetscVec)
    error = ccall((:KSPSolve, libpetsc), PetscErrorCode, (CKSP, CVec, CVec), ksp, b, x)
    @assert iszero(error)
end