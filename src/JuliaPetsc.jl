"""
    I have decided to remove all exclamation marks `!` at the end of routines "modifying" their arguments
    when the name is the same as in PETSc API. It was too confusing.
"""
module JuliaPetsc

using Libdl
using MPI

include("const_arch_ind.jl")
export  PetscErrorCode, PETSC_DECIDE
# export all items of some enums
for item in Iterators.flatten((instances(InsertMode), instances(MatAssemblyType)))
    @eval export $(Symbol(item))
end

include("load.jl")
export petsc_call

include("const_arch_dep.jl")
export PetscReal, PetscScalar, PetscInt, PetscIntOne

include("init.jl")
export PetscInitialize, PetscFinalize

include("vec.jl")
export  PetscVec, CVec,
        assemble!,
        set_local_size!, set_global_size!,
        vec2array,
        VecAssemble,
        VecAssemblyBegin,
        VecAssemblyEnd,
        VecCreate,
        VecDestroy,
        VecDuplicate,
        VecGetArray,
        VecGetLocalSize,
        VecGetOwnershipRange,
        VecGetSize,
        VecRestoreArray,
        VecSetFromOptions,
        VecSetSizes,
        VecSetUp,
        VecSetValues,
        VecView

include("mat.jl")
export  PetscMat, CMat,
        MatAssemble,
        MatAssemblyBegin,
        MatAssemblyEnd,
        MatCreate,
        MatCreate,
        MatCreateVecs,
        MatDestroy,
        MatGetOwnershipRange,
        MatSetFromOptions,
        MatSetSizes,
        MatSetUp,
        MatSetValues,
        MatView

include("ksp.jl")
export  PetscKSP, CKSP,
        KSPCreate,
        KSPDestroy,
        KSPSetFromOptions,
        KSPSetOperators,
        KSPSetUp,
        KSPSolve

end