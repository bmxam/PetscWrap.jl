const CMat = Ptr{Cvoid}

"""
    A Petsc matrix, actually just a pointer to the actual C matrix
"""
struct PetscMat
    ptr::Ref{CMat}

    PetscMat() = new(Ref{CMat}())
end


# allows us to pass PetscMat objects directly into CMat ccall signatures
Base.cconvert(::Type{CMat}, mat::PetscMat) = mat.ptr[]

"""
`row` and `col` must be in [1,size(mat)], i.e indexing starts at 1 (Julia).

# Implementation
For some unkwnown reason, calling `MatSetValue` fails.
"""
function Base.setindex!(mat::PetscMat, value::Number, row::Integer, col::Integer)
    MatSetValues(mat, PetscInt[row], PetscInt[col], PetscScalar[value], INSERT_VALUES)
end

# This is stupid but I don't know how to do better yet
Base.setindex!(mat::PetscMat, values, row::Integer, cols) = MatSetValues(mat, [row], collect(cols), values, INSERT_VALUES)
Base.setindex!(mat::PetscMat, values, rows, col::Integer) = MatSetValues(mat, collect(rows), [col], values, INSERT_VALUES)

Base.ndims(::Type{PetscMat}) = 2
"""
    MatSetValues(mat::PetscMat, I::Vector{PetscInt}, J::Vector{PetscInt}, V::Array{PetscScalar}, mode::InsertMode)

Wrapper to MatSetValues. Indexing starts at 1 (Julia)
"""
function MatSetValues(mat::PetscMat, I::Vector{PetscInt}, J::Vector{PetscInt}, V::Array{PetscScalar}, mode::InsertMode)
    nI = PetscInt(length(I))
    nJ = PetscInt(length(J))
    error = ccall((:MatSetValues, libpetsc), PetscErrorCode,
        (CMat, PetscInt, Ptr{PetscInt}, PetscInt, Ptr{PetscInt}, Ptr{PetscScalar}, InsertMode),
        mat, nI, I .- PetscIntOne, nJ, J .- PetscIntOne, V, mode)
    @assert iszero(error)
end

function MatSetValues(mat::PetscMat, I, J, V, mode::InsertMode)
    MatSetValues(mat, PetscInt.(I), PetscInt.(J), PetscScalar.(V), mode)
end

"""
    MatCreate(comm, mat::PetscMat)

Wrapper to MatCreate
"""
function MatCreate(comm, mat::PetscMat)
    error = ccall((:MatCreate, libpetsc), PetscErrorCode, (MPI.MPI_Comm, Ptr{CMat}), comm, mat.ptr)
    @assert iszero(error)
end

function MatCreate(comm)
    mat = PetscMat()
    MatCreate(comm, mat)
    return mat
end

function MatCreate()
    mat = PetscMat()
    MatCreate(MPI.COMM_WORLD, mat)
    return mat
end

"""
    create_matrix(nrows, ncols, nrows_loc = PETSC_DECIDE, ncols_loc = PETSC_DECIDE)

Create a `PetscMat` matrix of global size `(nrows, ncols)`.
"""
function create_matrix(nrows, ncols, nrows_loc = PETSC_DECIDE, ncols_loc = PETSC_DECIDE)
    mat = MatCreate()
    MatSetSizes(mat::PetscMat, nrows_loc, ncols_loc, nrows, ncols)
    return mat
end

"""
    MatSetSizes(mat::PetscMat, nrows_loc, ncols_loc, nrows_glo, ncols_glo)

Wrapper to MatSetSizes
"""
function MatSetSizes(mat::PetscMat, nrows_loc, ncols_loc, nrows_glo, ncols_glo)
    nr_loc = PetscInt(nrows_loc)
    nc_loc = PetscInt(ncols_loc)
    nr_glo = PetscInt(nrows_glo)
    nc_glo = PetscInt(ncols_glo)
    error = ccall((:MatSetSizes, libpetsc),
                PetscErrorCode,
                (CMat, PetscInt, PetscInt, PetscInt, PetscInt),
                mat, nr_loc, nc_loc, nr_glo, nc_glo
            )
    @assert iszero(error)
end

set_global_size!(mat::PetscMat, nrows, ncols) = MatSetSizes(mat, PETSC_DECIDE, PETSC_DECIDE, nrows, ncols)
set_local_size!(mat::PetscMat, nrows, ncols) = MatSetSizes(mat, nrows, ncols, PETSC_DECIDE, PETSC_DECIDE)

"""
    MatSetUp(mat::PetscMat)

Wrapper to MatSetUp
"""
function MatSetUp(mat::PetscMat)
    error = ccall((:MatSetUp, libpetsc), PetscErrorCode, (CMat,), mat)
    @assert iszero(error)
end

set_up!(mat::PetscMat) = MatSetUp(mat)

"""
    MatSetFromOptions(mat::PetscMat)

Wrapper to MatSetFromOptions
"""
function MatSetFromOptions(mat::PetscMat)
    error = ccall((:MatSetFromOptions, libpetsc), PetscErrorCode, (CMat,), mat)
    @assert iszero(error)
end
set_from_options!(mat::PetscMat) = MatSetFromOptions(mat)

"""
    MatGetOwnershipRange(mat::PetscMat)

Wrapper to MatGetOwnershipRange

However, the result `(rstart, rend)` is such that `mat[rstart:rend]` are the rows handled by the local processor.
This is different from the default `PETSc` result where the indexing starts at one and where `rend-1` is last row
handled by the local processor.
"""
function MatGetOwnershipRange(mat::PetscMat)
    rstart = Ref{PetscInt}(0)
    rend = Ref{PetscInt}(0)

    error = ccall((:MatGetOwnershipRange, libpetsc), PetscErrorCode, (CMat, Ref{PetscInt}, Ref{PetscInt}), mat, rstart, rend)
    @assert iszero(error)

    return rstart[] + 1, rend[]
end
get_range(mat::PetscMat) = MatGetOwnershipRange(mat)

"""
    Wrapper to `MatAssemblyBegin`
"""
function MatAssemblyBegin(mat::PetscMat, type::MatAssemblyType)
    error = ccall((:MatAssemblyBegin, libpetsc), PetscErrorCode, (CMat, MatAssemblyType), mat, type)
    @assert iszero(error)
end

"""
    Wrapper to `MatAssemblyEnd`
"""
function MatAssemblyEnd(mat::PetscMat, type::MatAssemblyType)
    error = ccall((:MatAssemblyEnd, libpetsc), PetscErrorCode, (CMat, MatAssemblyType), mat, type)
    @assert iszero(error)
end

"""
    Wrapper to `MatAssemblyBegin` and `MatAssemblyEnd` successively.
"""
function assemble!(mat::PetscMat, type::MatAssemblyType = MAT_FINAL_ASSEMBLY)
    MatAssemblyBegin(mat, type)
    MatAssemblyEnd(mat, type)
end

"""
    Wrapper to MatCreateVecs
"""
function MatCreateVecs(mat::PetscMat, vecr::PetscVec, veci::PetscVec)
    error = ccall((:MatCreateVecs, libpetsc), PetscErrorCode, (CMat, Ptr{CVec}, Ptr{CVec}), mat, vecr.ptr, veci.ptr)
    @assert iszero(error)
end

function MatCreateVecs(mat::PetscMat)
    vecr = PetscVec(); veci = PetscVec()
    MatCreateVecs(mat, vecr, veci)
    return vecr, veci
end

"""
    MatView(mat::PetscMat, viewer::PetscViewer = C_NULL)

Wrapper to `MatView`
"""
function MatView(mat::PetscMat, viewer::PetscViewer = C_NULL)
    error = ccall((:MatView, libpetsc), PetscErrorCode, (CMat, PetscViewer), mat, viewer)
    @assert iszero(error)
end

"""
    MatDestroy(mat::PetscMat)

Wrapper to MatDestroy
"""
function MatDestroy(mat::PetscMat)
    error = ccall((:MatDestroy, libpetsc), PetscErrorCode, (Ptr{CMat},), mat.ptr)
    @assert iszero(error)
end
destroy!(mat::PetscMat) = MatDestroy(mat)
