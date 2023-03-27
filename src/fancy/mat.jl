"""
`row` and `col` must be in [1,size(mat)], i.e indexing starts at 1 (Julia).

# Implementation

For some unkwnown reason, calling `MatSetValue` fails.
"""
function Base.setindex!(mat::Mat, value::Number, row::Integer, col::Integer)
    setValues(mat, PetscInt[row - 1], PetscInt[col - 1], PetscScalar[value], INSERT_VALUES)
end

# This is stupid but I don't know how to do better yet
function Base.setindex!(mat::Mat, values, row::Integer, cols)
    setValues(mat, [row - 1], collect(cols) .- 1, values, INSERT_VALUES)
end
function Base.setindex!(mat::Mat, values, rows, col::Integer)
    setValues(mat, collect(rows) .- 1, [col - 1], values, INSERT_VALUES)
end

Base.ndims(::Type{Mat}) = 2

"""
    create_matrix(
        comm::MPI.Comm = MPI.COMM_WORLD;
        nrows_loc = PETSC_DECIDE,
        ncols_loc = PETSC_DECIDE,
        nrows_glo = PETSC_DECIDE,
        ncols_glo = PETSC_DECIDE,
        auto_setup = false,
    )

Use `auto_setup = true` to immediatly call `set_from_options!` and `set_up!`.
"""
function create_matrix(
    comm::MPI.Comm = MPI.COMM_WORLD;
    nrows_loc = PETSC_DECIDE,
    ncols_loc = PETSC_DECIDE,
    nrows_glo = PETSC_DECIDE,
    ncols_glo = PETSC_DECIDE,
    auto_setup = false,
)
    mat = create(Mat, comm)
    setSizes(mat, nrows_loc, ncols_loc, nrows_glo, ncols_glo)

    if (auto_setup)
        set_from_options!(mat)
        set_up!(mat)
    end
    return mat
end

"""
Wrapper to `MatCreateComposite` using the "alternative construction" from the PETSc documentation.
"""
function create_composite_add(matrices)
    M, N = getSize(matrices[1])
    m, n = getLocalSize(matrices[1])
    mat = create_matrix(
        matrices[1].comm;
        nrows_loc = m,
        ncols_loc = n,
        nrows_glo = M,
        ncols_glo = N,
        auto_setup = false,
    )
    setType(mat, "composite")
    for _mat in matrices
        compositeAddMat(mat, _mat)
    end
    assemble!(mat)
    return mat
end

function set_global_size!(mat::Mat, nrows, ncols)
    setSizes(mat, PETSC_DECIDE, PETSC_DECIDE, nrows, ncols)
end

function set_local_to_global!(
    mat::Mat,
    rlid2gid::Vector{I},
    clid2gid::Vector{I},
) where {I<:Integer}
    rmapping = create(ISLocalToGlobalMapping, rlid2gid)
    cmapping = create(ISLocalToGlobalMapping, clid2gid)
    setLocalToGlobalMapping(mat, rmapping, cmapping)
    destroy.(rmapping, cmapping)
end

function set_local_size!(mat::Mat, nrows, ncols)
    setSizes(mat, nrows, ncols, PETSC_DECIDE, PETSC_DECIDE)
end

"""
    get_range(mat::Mat)

Wrapper to `MatGetOwnershipRange`

However, the result `(rstart, rend)` is such that `mat[rstart:rend]` are the rows handled by the local processor.
This is different from the default `PETSc.MatGetOwnershipRange` result where the indexing starts at zero and where
`rend-1` is last row handled by the local processor.
"""
function get_range(mat::Mat)
    rstart, rend = getOwnershipRange(mat)
    return (rstart + 1, rend)
end

"""
    get_urange(mat::Mat)

Provide a `UnitRange` from the method `get_range`.
"""
function get_urange(mat::Mat)
    rstart, rend = getOwnershipRange(mat)
    return (rstart + 1):rend
end

"""
    Wrapper to `MatAssemblyBegin` and `MatAssemblyEnd` successively.
"""
function assemble!(mat::Mat, type::MatAssemblyType = MAT_FINAL_ASSEMBLY)
    assemblyBegin(mat, type)
    assemblyEnd(mat, type)
end

"""
    set_value!(mat::Mat, I, J, V, mode = ADD_VALUES)

Set value of `mat`
`mat[i, j] = v`.

1-based indexing
"""
function set_value!(mat::Mat, i::PetscInt, j::PetscInt, v::PetscScalar, mode = ADD_VALUES)
    setValue(mat, i - 1, j - 1, v, mode)
end
function set_value!(mat, i, j, v, mode = ADD_VALUES)
    set_value!(mat, PetscInt(i), PetscInt(j), PetscScalar(v), mode)
end

"""
    set_values!(mat::Mat, I, J, V, mode = ADD_VALUES)

Set values of `mat` in `SparseArrays` fashion : using COO format:
`mat[I[k], J[k]] = V[k]`.
"""
function set_values!(
    mat::Mat,
    I::Vector{PetscInt},
    J::Vector{PetscInt},
    V::Vector{PetscScalar},
    mode = ADD_VALUES,
)
    for (i, j, v) in zip(I, J, V)
        setValue(mat, i - PetscIntOne, j - PetscIntOne, v, mode)
    end
end

function set_values!(mat, I, J, V, mode = ADD_VALUES)
    set_values!(mat, PetscInt.(I), PetscInt.(J), PetscScalar.(V), mode)
end

# Warning : cannot use Vector{Integer} because `[1, 2] isa Vector{Integer}` is `false`
function _preallocate!(mat::Mat, dnz::Integer, onz::Integer, ::Val{:mpiaij})
    MPIAIJSetPreallocation(mat, PetscInt(dnz), PetscInt(onz))
end
function _preallocate!(
    mat::Mat,
    d_nnz::Vector{I},
    o_nnz::Vector{I},
    ::Val{:mpiaij},
) where {I}
    MPIAIJSetPreallocation(
        mat,
        PetscInt(0),
        PetscInt.(d_nnz),
        PetscInt(0),
        PetscInt.(o_nnz),
    )
end
function _preallocate!(mat::Mat, nz::Integer, ::Integer, ::Val{:seqaij})
    SeqAIJSetPreallocation(mat, PetscInt(nz))
end
function _preallocate!(mat::Mat, nnz::Vector{I}, ::Vector{I}, ::Val{:seqaij}) where {I}
    SeqAIJSetPreallocation(mat, PetscInt(0), PetscInt.(nnz))
end

"""
    preallocate!(mat::Mat, dnz, onz, warn::Bool = true)

Dispatch preallocation according matrix type (seq or mpiaij for instance). TODO: should use kwargs.
"""
function preallocate!(mat::Mat, dnz, onz, warn::Bool = true)
    _preallocate!(mat, dnz, onz, Val(Symbol(MatGetType(mat))))
    setOption(mat, MAT_NEW_NONZERO_ALLOCATION_ERR, warn)
end

"""
    mat2file(mat::Mat, filename::String, format::PetscViewerFormat = PETSC_VIEWER_ASCII_CSV, type::String = "ascii")

Write a Mat to a file.
"""
function mat2file(
    mat::Mat,
    filename::String,
    format::PetscViewerFormat = PETSC_VIEWER_ASCII_CSV,
    type::String = "ascii",
)
    viewer = PetscViewer(mat.comm, filename, format, type)
    view(mat, viewer)
    destroy!(viewer)
end

Base.show(::IO, mat::Mat) = view(mat)
