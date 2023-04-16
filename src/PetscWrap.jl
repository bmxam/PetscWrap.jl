"""
    This package has been inspired and even sometimes directly copied from two projects:
    - https://github.com/gridap/GridapPETSc.jl
    - https://github.com/JuliaParallel/PETSc.jl
"""
module PetscWrap

using Libdl
using MPI
using LinearAlgebra

include("const_arch_ind.jl")
export PetscErrorCode, PETSC_DECIDE
# export all items of some enums
for item in Iterators.flatten((
    instances(InsertMode),
    instances(MatAssemblyType),
    instances(PetscViewerFormat),
    instances(PetscFileMode),
))
    @eval export $(Symbol(item))
end

include("load.jl")
export show_petsc_path

include("const_arch_dep.jl")
export PetscReal, PetscScalar, PetscInt, PetscIntOne

include("init.jl")
export PetscInitialize, PetscInitialized, PetscFinalize

include("PetscViewer.jl")
export PetscViewer,
    CViewer,
    ASCIIOpen,
    create,
    destroy,
    fileSetMode,
    fileSetName,
    HDF5Open,
    popFormat,
    pushFormat,
    setType,
    stdWorld,
    viewerView

include("ISLocalToGlobalMapping.jl")
include("Vec.jl")
export Vec,
    CVec,
    assemblyBegin,
    assemblyEnd,
    copy,
    duplicate,
    getArray,
    getLocalSize,
    getOwnershipRange,
    getSize,
    getValues,
    restoreArray,
    scale,
    setFromOptions,
    setSizes,
    setUp,
    setValue,
    setValueLocal,
    setValues,
    setValuesLocal,
    vecView

include("Mat.jl")
export Mat,
    CMat,
    createComposite,
    createDense,
    createVecs,
    getOwnershipRangeColumn,
    getType,
    MPIAIJSetPreallocation,
    mult,
    multAdd,
    SeqAIJSetPreallocation,
    matView

include("KSP.jl")
export KSP, CKSP, setOperators, solve

# fancy
include("fancy/common.jl")
include("fancy/viewer.jl")
export destroy!, push_format!, set_mode!, set_name!, set_type!

include("fancy/vec.jl")
export assemble!,
    create_vector,
    destroy!,
    duplicate,
    get_range,
    get_urange,
    set_local_size!,
    set_local_to_global!,
    set_global_size!,
    set_from_options!,
    set_up!,
    set_value!,
    set_value_local!,
    set_values!,
    set_values_local!,
    vec2array,
    vec2file

include("fancy/mat.jl")
export create_composite_add, create_matrix, mat2file, preallocate!

include("fancy/ksp.jl")
export create_ksp, solve, solve!, set_operators!

end
