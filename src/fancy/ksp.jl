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

set_operators!(ksp::PetscKSP, Amat::PetscMat) = KSPSetOperators(ksp, Amat, Amat)
set_operators!(ksp::PetscKSP, Amat::PetscMat, Pmat::PetscMat) = KSPSetOperators(ksp, Amat, Pmat)


solve!(ksp::PetscKSP, b::PetscVec, x::PetscVec) = KSPSolve(ksp, b, x)

function solve(ksp::PetscKSP, b::PetscVec)
    x = VecDuplicate(b)
    KSPSolve(ksp, b, x)
    return x
end

set_up!(ksp::PetscKSP) = KSPSetUp(ksp)

set_from_options!(ksp::PetscKSP) = KSPSetFromOptions(ksp)

destroy!(ksp::PetscKSP) = KSPDestroy(ksp)