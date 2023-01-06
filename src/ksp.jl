const CKSP = Ptr{Cvoid}

struct PetscKSP
    ptr::Ref{CKSP}
    comm::MPI.Comm

    PetscKSP(comm::MPI.Comm) = new(Ref{CKSP}(), comm)
end

# allows us to pass PetscKSP objects directly into CKSP ccall signatures
Base.cconvert(::Type{CKSP}, ksp::PetscKSP) = ksp.ptr[]

"""
    KSPCreate(comm::MPI.Comm = MPI.COMM_WORLD)

Wrapper for `KSPCreate`
"""
function KSPCreate(comm::MPI.Comm=MPI.COMM_WORLD)
    ksp = PetscKSP(comm)
    error = ccall((:KSPCreate, libpetsc), PetscErrorCode, (MPI.MPI_Comm, Ptr{CKSP}), comm, ksp.ptr)
    @assert iszero(error)
    return ksp
end

"""
    KSPDestroy(ksp::PetscKSP)

Wrapper to `KSPDestroy`
"""
function KSPDestroy(ksp::PetscKSP)
    error = ccall((:KSPDestroy, libpetsc), PetscErrorCode, (Ptr{CKSP},), ksp.ptr)
    @assert iszero(error)
end

"""
    KSPSetFromOptions(ksp::PetscKSP)

Wrapper to KSPSetFromOptions
"""
function KSPSetFromOptions(ksp::PetscKSP)
    error = ccall((:KSPSetFromOptions, libpetsc), PetscErrorCode, (CKSP,), ksp)
    @assert iszero(error)
end

"""
    KSPSetOperators(ksp::PetscKSP, Amat::PetscMat, Pmat::PetscMat)

Wrapper for KSPSetOperators
"""
function KSPSetOperators(ksp::PetscKSP, Amat::PetscMat, Pmat::PetscMat)
    error = ccall((:KSPSetOperators, libpetsc), PetscErrorCode, (CKSP, CMat, CMat), ksp, Amat, Pmat)
    @assert iszero(error)
end

"""
    KSPSetUp(ksp::PetscKSP)

Wrapper to KSPSetUp
"""
function KSPSetUp(ksp::PetscKSP)
    error = ccall((:KSPSetUp, libpetsc), PetscErrorCode, (CKSP,), ksp)
    @assert iszero(error)
end

"""
    KSPSolve(ksp::PetscKSP, b::PetscVec, x::PetscVec)

Wrapper for KSPSolve
"""
function KSPSolve(ksp::PetscKSP, b::PetscVec, x::PetscVec)
    error = ccall((:KSPSolve, libpetsc), PetscErrorCode, (CKSP, CVec, CVec), ksp, b, x)
    @assert iszero(error)
end