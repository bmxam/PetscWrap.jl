"""
`row` and `col` must be in [1,size(mat)], i.e indexing starts at 1 (Julia).

# Implementation
For some unkwnown reason, calling `MatSetValue` fails.
"""
function Base.setindex!(mat::PetscMat, value::Number, row::Integer, col::Integer)
    MatSetValues(mat, PetscInt[row - 1], PetscInt[col - 1], PetscScalar[value], INSERT_VALUES)
end

# This is stupid but I don't know how to do better yet
Base.setindex!(mat::PetscMat, values, row::Integer, cols) = MatSetValues(mat, [row-1], collect(cols) .- 1, values, INSERT_VALUES)
Base.setindex!(mat::PetscMat, values, rows, col::Integer) = MatSetValues(mat, collect(rows) .- 1, [col-1], values, INSERT_VALUES)

Base.ndims(::Type{PetscMat}) = 2

"""
    create_matrix(nrows, ncols, nrows_loc = PETSC_DECIDE, ncols_loc = PETSC_DECIDE)

Create a `PetscMat` matrix of global size `(nrows, ncols)`.
"""
function create_matrix(nrows, ncols, nrows_loc = PETSC_DECIDE, ncols_loc = PETSC_DECIDE)
    mat = MatCreate()
    MatSetSizes(mat::PetscMat, nrows_loc, ncols_loc, nrows, ncols)
    return mat
end


set_global_size!(mat::PetscMat, nrows, ncols) = MatSetSizes(mat, PETSC_DECIDE, PETSC_DECIDE, nrows, ncols)
set_local_size!(mat::PetscMat, nrows, ncols) = MatSetSizes(mat, nrows, ncols, PETSC_DECIDE, PETSC_DECIDE)

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
    Wrapper to `MatAssemblyBegin` and `MatAssemblyEnd` successively.
"""
function assemble!(mat::PetscMat, type::MatAssemblyType = MAT_FINAL_ASSEMBLY)
    MatAssemblyBegin(mat, type)
    MatAssemblyEnd(mat, type)
end


Base.show(::IO, mat::PetscMat) = MatView(mat)

destroy!(mat::PetscMat) = MatDestroy(mat)