const CKSP = Ptr{Cvoid}

struct PetscKSP
    ptr::Ref{CKSP}

    PetscKSP() = new(Ref{CKSP}())
end

# allows us to pass PetscKSP objects directly into CKSP ccall signatures
Base.cconvert(::Type{CKSP}, ksp::PetscKSP) = ksp.ptr[]

"""
    KSPCreate(comm::MPI.Comm, ksp::PetscKSP)

Wrapper for KSPCreate
"""
function KSPCreate(comm, ksp::PetscKSP)
    error = ccall((:KSPCreate, libpetsc), PetscErrorCode, (MPI.MPI_Comm, Ptr{CKSP}), comm, ksp.ptr)
    @assert iszero(error)
end

function KSPCreate(comm)
    ksp = PetscKSP()
    KSPCreate(comm, ksp)
    return ksp
end

function KSPCreate()
    ksp = PetscKSP()
    KSPCreate(MPI.COMM_WORLD, ksp)
    return ksp
end

create_ksp() = KSPCreate()

function create_ksp(A::PetscMat)
    ksp = KSPCreate()
    KSPSetOperators(ksp, A, A)
    return ksp
end

function create_ksp(Amat::PetscMat, Pmat::PetscMat)
    ksp = KSPCreate()
    KSPSetOperators(ksp, Amat, Pmat)
    return ksp
end

"""
    KSPSetOperators(ksp::PetscKSP, Amat::PetscMat, Pmat::PetscMat)

Wrapper for KSPSetOperators
"""
function KSPSetOperators(ksp::PetscKSP, Amat::PetscMat, Pmat::PetscMat)
    error = ccall((:KSPSetOperators, libpetsc), PetscErrorCode, (CKSP, CMat, CMat), ksp, Amat, Pmat)
    @assert iszero(error)
end

set_operators!(ksp::PetscKSP, A::PetscMat) = KSPSetOperators(ksp, Amat, Amat)
set_operators!(ksp::PetscKSP, Amat::PetscMat, Pmat::PetscMat) = KSPSetOperators(ksp, Amat, Pmat)


"""
    KSPSolve(ksp::PetscKSP, b::PetscVec, x::PetscVec)

Wrapper for KSPSolve
"""
function KSPSolve(ksp::PetscKSP, b::PetscVec, x::PetscVec)
    error = ccall((:KSPSolve, libpetsc), PetscErrorCode, (CKSP, CVec, CVec), ksp, b, x)
    @assert iszero(error)
end
solve!(ksp::PetscKSP, b::PetscVec, x::PetscVec) = KSPSolve(ksp, b, x)

function solve(ksp::PetscKSP, b::PetscVec)
    x = VecDuplicate(b)
    KSPSolve(ksp, b, x)
    return x
end

"""
    KSPSetUp(ksp::PetscKSP)

Wrapper to KSPSetUp
"""
function KSPSetUp(ksp::PetscKSP)
    error = ccall((:KSPSetUp, libpetsc), PetscErrorCode, (CKSP,), ksp)
    @assert iszero(error)
end
set_up!(ksp::PetscKSP) = KSPSetUp(ksp)

"""
    KSPSetFromOptions(ksp::PetscKSP)

Wrapper to KSPSetFromOptions
"""
function KSPSetFromOptions(ksp::PetscKSP)
    error = ccall((:KSPSetFromOptions, libpetsc), PetscErrorCode, (CKSP,), ksp)
    @assert iszero(error)
end
set_from_options!(ksp::PetscKSP) = KSPSetFromOptions(ksp)

"""
    KSPDestroy(ksp::PetscKSP)

Wrapper to KSPDestroy
"""
function KSPDestroy(ksp::PetscKSP)
    error = ccall((:KSPDestroy, libpetsc), PetscErrorCode, (Ptr{CKSP},), ksp.ptr)
    @assert iszero(error)
end

destroy!(ksp::PetscKSP) = KSPDestroy(ksp)