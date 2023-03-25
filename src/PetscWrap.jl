"""
    This package has been inspired and even sometimes directly copied from two projects:
    - https://github.com/gridap/GridapPETSc.jl
    - https://github.com/JuliaParallel/PETSc.jl
"""
module PetscWrap

using Libdl
using MPI

include("utils.jl")

include("const_arch_ind.jl")
export PetscErrorCode
# export all items of some enums
for item in Iterators.flatten((
    instances(InsertMode),
    instances(MatAssemblyType),
    instances(PetscViewerFormat),
    instances(PetscFileMode),
    instances(ScatterMode),
))
    @eval export $(Symbol(item))
end

include("load.jl")
export show_petsc_path

include("const_arch_dep.jl")
export PetscReal, PetscScalar, PetscInt, PetscIntOne, PETSC_DECIDE

include("init.jl")
export PetscInitialize, PetscFinalize

include("viewer.jl")
export PetscViewer, CViewer,
    PetscViewerASCIIOpen,
    PetscViewerCreate,
    PetscViewerDestroy,
    PetscViewerFileSetMode,
    PetscViewerFileSetName,
    PetscViewerHDF5Open,
    PetscViewerPopFormat,
    PetscViewerPushFormat,
    PetscViewerSetType,
    PetscViewerStdWorld,
    PetscViewerView

include("vec.jl")
export PetscVec, CVec,
    VecAssemble,
    VecAssemblyBegin,
    VecAssemblyEnd,
    VecCopy,
    VecCreate,
    VecCreateGhost,
    VecDestroy,
    VecDuplicate,
    VecGetArray,
    VecGetLocalSize,
    VecGetOwnershipRange,
    VecGetSize,
    VecGhostGetLocalForm,
    VecGhostRestoreLocalForm,
    VecGhostUpdateBegin,
    VecGhostUpdateEnd,
    VecRestoreArray,
    VecScale,
    VecSetFromOptions,
    VecSetSizes,
    VecSetUp,
    VecSetValue,
    VecSetValues,
    VecView

include("mat.jl")
export PetscMat, CMat,
    MatAssemble,
    MatAssemblyBegin,
    MatAssemblyEnd,
    MatCreate,
    MatCreateComposite,
    MatCreateDense,
    MatCreateVecs,
    MatDestroy,
    MatGetLocalSize,
    MatGetOwnershipRange,
    MatGetOwnershipRangeColumn,
    MatGetSize,
    MatGetType,
    MatMPIAIJSetPreallocation,
    MatMult,
    MatMultAdd,
    MatSeqAIJSetPreallocation,
    MatSetFromOptions,
    MatSetSizes,
    MatSetUp,
    MatSetValue,
    MatSetValues,
    MatView

include("ksp.jl")
export PetscKSP, CKSP,
    KSPCreate,
    KSPDestroy,
    KSPSetFromOptions,
    KSPSetOperators,
    KSPSetUp,
    KSPSolve

# fancy
include("fancy/viewer.jl")
export destroy!,
    push_format!,
    set_mode!,
    set_name!,
    set_type!

include("fancy/abstract_vector_interface.jl")

include("fancy/vec.jl")
export assemble!,
    create_vector,
    destroy!,
    duplicate,
    get_range,
    get_urange,
    set_local_size!, set_global_size!,
    set_from_options!,
    set_up!,
    vec2array,
    vec2file

include("fancy/mat.jl")
export create_composite_add,
    create_matrix,
    mat2file,
    preallocate!,
    set_value!,
    set_values!

include("fancy/ksp.jl")
export create_ksp,
    solve, solve!,
    set_operators!

end