"""
`row` and `col` must be in [1,size(mat)], i.e indexing starts at 1 (Julia).

# Implementation

For some unkwnown reason, calling `MatSetValue` fails.
"""
function Base.setindex!(mat::PetscMat, value::Number, row::Integer, col::Integer)
    MatSetValues(
        mat,
        PetscInt[row - 1],
        PetscInt[col - 1],
        PetscScalar[value],
        INSERT_VALUES,
    )
end

# This is stupid but I don't know how to do better yet
function Base.setindex!(mat::PetscMat, values, row::Integer, cols)
    MatSetValues(mat, [row - 1], collect(cols) .- 1, values, INSERT_VALUES)
end
function Base.setindex!(mat::PetscMat, values, rows, col::Integer)
    MatSetValues(mat, collect(rows) .- 1, [col - 1], values, INSERT_VALUES)
end

Base.ndims(::Type{PetscMat}) = 2

"""
    create_matrix(nrows, ncols, nrows_loc = PETSC_DECIDE, ncols_loc = PETSC_DECIDE; auto_setup = false)

Create a `PetscMat` matrix of global size `(nrows, ncols)`.

Use `auto_setup = true` to immediatly call `set_from_options!` and `set_up!`.
"""
function create_matrix(
    nrows,
    ncols,
    nrows_loc = PETSC_DECIDE,
    ncols_loc = PETSC_DECIDE;
    auto_setup = false,
    comm::MPI.Comm = MPI.COMM_WORLD,
)
    mat = MatCreate()
    MatSetSizes(mat::PetscMat, nrows_loc, ncols_loc, nrows, ncols)

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
    N, M = MatGetSize(matrices[1])
    n, m = MatGetLocalSize(matrices[1])
    mat = create_matrix(N, M, n, m; auto_setup = false, comm = matrices[1].comm)
    MatSetType(mat, "composite")
    for m in matrices
        MatCompositeAddMat(mat, m)
    end
    assemble!(mat)
    return mat
end

function set_global_size!(mat::PetscMat, nrows, ncols)
    MatSetSizes(mat, PETSC_DECIDE, PETSC_DECIDE, nrows, ncols)
end
function set_local_size!(mat::PetscMat, nrows, ncols)
    MatSetSizes(mat, nrows, ncols, PETSC_DECIDE, PETSC_DECIDE)
end

set_from_options!(mat::PetscMat) = MatSetFromOptions(mat)

set_up!(mat::PetscMat) = MatSetUp(mat)

"""
    get_range(mat::PetscMat)

Wrapper to `MatGetOwnershipRange`

However, the result `(rstart, rend)` is such that `mat[rstart:rend]` are the rows handled by the local processor.
This is different from the default `PETSc.MatGetOwnershipRange` result where the indexing starts at zero and where
`rend-1` is last row handled by the local processor.
"""
function get_range(mat::PetscMat)
    rstart, rend = MatGetOwnershipRange(mat)
    return (rstart + 1, rend)
end

"""
    get_urange(mat::PetscMat)

Provide a `UnitRange` from the method `get_range`.
"""
function get_urange(mat::PetscMat)
    rstart, rend = MatGetOwnershipRange(mat)
    return (rstart + 1):rend
end

"""
    Wrapper to `MatAssemblyBegin` and `MatAssemblyEnd` successively.
"""
function assemble!(mat::PetscMat, type::MatAssemblyType = MAT_FINAL_ASSEMBLY)
    MatAssemblyBegin(mat, type)
    MatAssemblyEnd(mat, type)
end

"""
    set_value!(mat::PetscMat, I, J, V, mode = ADD_VALUES)

Set value of `mat`
`mat[i, j] = v`.
"""
function set_value!(
    mat::PetscMat,
    i::PetscInt,
    j::PetscInt,
    v::PetscScalar,
    mode = ADD_VALUES,
)
    MatSetValue(mat, i - 1, j - 1, v, mode)
end
function set_value!(mat, i, j, v, mode = ADD_VALUES)
    set_value!(mat, PetscInt(i), PetscInt(j), PetscScalar(v), mode)
end

"""
    set_values!(mat::PetscMat, I, J, V, mode = ADD_VALUES)

Set values of `mat` in `SparseArrays` fashion : using COO format:
`mat[I[k], J[k]] = V[k]`.
"""
function set_values!(
    mat::PetscMat,
    I::Vector{PetscInt},
    J::Vector{PetscInt},
    V::Vector{PetscScalar},
    mode = ADD_VALUES,
)
    for (i, j, v) in zip(I, J, V)
        MatSetValue(mat, i - PetscIntOne, j - PetscIntOne, v, mode)
    end
end

function set_values!(mat, I, J, V, mode = ADD_VALUES)
    set_values!(mat, PetscInt.(I), PetscInt.(J), PetscScalar.(V), mode)
end

# Warning : cannot use Vector{Integer} because `[1, 2] isa Vector{Integer}` is `false`
function _preallocate!(mat::PetscMat, dnz::Integer, onz::Integer, ::Val{:mpiaij})
    MatMPIAIJSetPreallocation(mat, PetscInt(dnz), PetscInt(onz))
end
function _preallocate!(
    mat::PetscMat,
    d_nnz::Vector{I},
    o_nnz::Vector{I},
    ::Val{:mpiaij},
) where {I}
    MatMPIAIJSetPreallocation(
        mat,
        PetscInt(0),
        PetscInt.(d_nnz),
        PetscInt(0),
        PetscInt.(o_nnz),
    )
end
function _preallocate!(mat::PetscMat, nz::Integer, ::Integer, ::Val{:seqaij})
    MatSeqAIJSetPreallocation(mat, PetscInt(nz))
end
function _preallocate!(mat::PetscMat, nnz::Vector{I}, ::Vector{I}, ::Val{:seqaij}) where {I}
    MatSeqAIJSetPreallocation(mat, PetscInt(0), PetscInt.(nnz))
end

"""
    preallocate!(mat::PetscMat, dnz, onz, warn::Bool = true)

Dispatch preallocation according matrix type (seq or mpiaij for instance). TODO: should use kwargs.
"""
function preallocate!(mat::PetscMat, dnz, onz, warn::Bool = true)
    _preallocate!(mat, dnz, onz, Val(Symbol(MatGetType(mat))))
    MatSetOption(mat, MAT_NEW_NONZERO_ALLOCATION_ERR, warn)
end

"""
    mat2file(mat::PetscMat, filename::String, format::PetscViewerFormat = PETSC_VIEWER_ASCII_CSV, type::String = "ascii")

Write a PetscMat to a file.
"""
function mat2file(
    mat::PetscMat,
    filename::String,
    format::PetscViewerFormat = PETSC_VIEWER_ASCII_CSV,
    type::String = "ascii",
)
    viewer = PetscViewer(mat.comm, filename, format, type)
    MatView(mat, viewer)
    destroy!(viewer)
end

Base.show(::IO, mat::PetscMat) = MatView(mat)

destroy!(mat::PetscMat) = MatDestroy(mat)
