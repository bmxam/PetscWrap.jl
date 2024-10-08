"""
    This package has been inspired and even sometimes directly copied from two projects:
    - https://github.com/gridap/GridapPETSc.jl
    - https://github.com/JuliaParallel/PETSc.jl
"""
module PetscWrap

using Libdl
using MPI
using LinearAlgebra

# GridapPETSc trick to track allocs/deallocs
const _NREFS = Ref(0)

include("const_arch_ind.jl")
export PetscErrorCode, PETSC_DECIDE
# export all items of some enums
for item in Iterators.flatten((
    instances(InsertMode),
    instances(MatAssemblyType),
    instances(MatOption),
    instances(MatDuplicateOption),
    instances(PetscViewerFormat),
    instances(PetscFileMode),
))
    @eval export $(Symbol(item))
end

# Find PETSc lib path and set number types
const _deps_jl = joinpath(@__DIR__, "..", "deps", "deps.jl")
if (haskey(ENV, "JULIA_REGISTRYCI_AUTOMERGE") || haskey(ENV, "DOC_DEPLOYMENT"))
    include(joinpath(@__DIR__, "..", "deps", "fake_deps.jl"))
elseif !isfile(_deps_jl)
    msg = """
    PetscWrap needs to be configured before use. Type

    pkg> build

    and try again.
    """
    error(msg)
else
    include(_deps_jl)
    if libpetsc_provider == "PETSc_jll"
        using PETSc_jll
        const libpetsc = PETSc_jll.libpetsc_path
    else
        const libpetsc = _libpetsc_path
    end
end
export PetscReal, PetscScalar, PetscInt, PetscIntOne

const PETSC_DEFAULT = PetscInt(-2)
const PETSC_DECIDE = PetscInt(-1)
const PETSC_DETERMINE = PETSC_DECIDE

include("common.jl")

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
    createShell,
    createVecs,
    getOwnershipRangeColumn,
    getType,
    MPIAIJSetPreallocation,
    mult,
    multAdd,
    SeqAIJSetPreallocation,
    setOption,
    setPreallocationCOO,
    setValuesCOO,
    shellSetOperation,
    matView,
    zeroEntries

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
export create_composite_add, create_matrix, mat2file, preallocate!, set_shell_mul!

include("fancy/ksp.jl")
export create_ksp, solve, solve!, set_operators!

end
