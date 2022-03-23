const CMat = Ptr{Cvoid}

"""
    A Petsc matrix, actually just a pointer to the actual C matrix
"""
struct PetscMat
    ptr::Ref{CMat}
    comm::MPI.Comm

    PetscMat(comm::MPI.Comm) = new(Ref{CMat}(), comm)
end


# allows us to pass PetscMat objects directly into CMat ccall signatures
Base.cconvert(::Type{CMat}, mat::PetscMat) = mat.ptr[]

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
    MatCompositeAddMat(mat::PetscMat, smat::PetscMat)

Wrapper to `MatCompositeAddMat`.
"""
function MatCompositeAddMat(mat::PetscMat, smat::PetscMat)
    error = ccall((:MatCompositeAddMat, libpetsc), PetscErrorCode, (CMat, CMat), mat, smat)
    @assert iszero(error)
end

"""
    MatCreate(comm::MPI.Comm, mat::PetscMat)

Wrapper to `MatCreate`
"""
function MatCreate(comm::MPI.Comm = MPI.COMM_WORLD)
    mat = PetscMat(comm)
    error = ccall((:MatCreate, libpetsc), PetscErrorCode, (MPI.MPI_Comm, Ptr{CMat}), comm, mat.ptr)
    @assert iszero(error)
    return mat
end

"""
    MatCreateDense(comm::MPI.Comm, m::PetscInt, n::PetscInt, M::PetscInt, N::PetscInt)

Wrapper to `MatCreateDense`. Last argument `data` is not supported yet (NULL is passed).
"""
function MatCreateDense(comm::MPI.Comm, m::PetscInt, n::PetscInt, M::PetscInt, N::PetscInt)
    mat = PetscMat(comm)
    error = ccall((:MatCreateDense, libpetsc), PetscErrorCode,
        (MPI.MPI_Comm, PetscInt, PetscInt, PetscInt, PetscInt, Ptr{PetscScalar}, Ptr{CMat}),
        comm, m, n, M, N, C_NULL, mat.ptr)
    @assert iszero(error)
    return mat
end

function MatCreateDense(comm::MPI.Comm,
    m::Integer = PETSC_DECIDE,
    n::Integer = PETSC_DECIDE,
    M::Integer = PETSC_DECIDE,
    N::Integer = PETSC_DECIDE)

    return MatCreateDense(comm, PetscInt(m), PetscInt(n), PetscInt(M), PetscInt(N))
end

"""
    Wrapper to MatCreateVecs
"""
function MatCreateVecs(mat::PetscMat, vecr::PetscVec, veci::PetscVec)
    error = ccall((:MatCreateVecs, libpetsc), PetscErrorCode, (CMat, Ptr{CVec}, Ptr{CVec}), mat, vecr.ptr, veci.ptr)
    @assert iszero(error)
end

function MatCreateVecs(mat::PetscMat)
    vecr = PetscVec(mat.comm); veci = PetscVec(mat.comm)
    MatCreateVecs(mat, vecr, veci)
    return vecr, veci
end

"""
    MatDestroy(mat::PetscMat)

Wrapper to MatDestroy
"""
function MatDestroy(mat::PetscMat)
    error = ccall((:MatDestroy, libpetsc), PetscErrorCode, (Ptr{CMat},), mat.ptr)
    @assert iszero(error)
end

"""
    MatGetLocalSize(mat::PetscMat)

Wrapper to `MatGetLocalSize`. Return the number of local rows and cols of the matrix (i.e on the processor).
"""
function MatGetLocalSize(mat::PetscMat)
    nrows_loc = Ref{PetscInt}(0)
    ncols_loc = Ref{PetscInt}(0)

    error = ccall((:MatGetLocalSize, libpetsc), PetscErrorCode, (CMat, Ref{PetscInt}, Ref{PetscInt}), mat, nrows_loc, ncols_loc)
    @assert iszero(error)

    return nrows_loc[], ncols_loc[]
end

"""
    MatGetOwnershipRange(mat::PetscMat)

Wrapper to `MatGetOwnershipRange`

The result `(rstart, rend)` is a Tuple indicating the rows handled by the local processor.

# Warning
`PETSc` indexing starts at zero (so `rstart` may be zero) and `rend-1` is the last row
handled by the local processor.
"""
function MatGetOwnershipRange(mat::PetscMat)
    rstart = Ref{PetscInt}(0)
    rend = Ref{PetscInt}(0)

    error = ccall((:MatGetOwnershipRange, libpetsc), PetscErrorCode, (CMat, Ref{PetscInt}, Ref{PetscInt}), mat, rstart, rend)
    @assert iszero(error)

    return rstart[], rend[]
end

"""
    MatGetOwnershipRangeColumn(mat::PetscMat)

Wrapper to `MatGetOwnershipRangeColumn`

The result `(cstart, cend)` is a Tuple indicating the columns handled by the local processor.

# Warning
`PETSc` indexing starts at zero (so `cstart` may be zero) and `cend-1` is the last column
handled by the local processor.
"""
function MatGetOwnershipRangeColumn(mat::PetscMat)
    cstart = Ref{PetscInt}(0)
    cend = Ref{PetscInt}(0)

    error = ccall((:MatGetOwnershipRangeColumn, libpetsc), PetscErrorCode, (CMat, Ref{PetscInt}, Ref{PetscInt}), mat, cstart, cend)
    @assert iszero(error)

    return cstart[], cend[]
end

"""
    MatGetSize(mat::PetscMat)

Wrapper to `MatGetSize`. Return the number of rows and cols of the matrix (global number).
"""
function MatGetSize(mat::PetscMat)
    nrows_glo = Ref{PetscInt}(0)
    ncols_glo = Ref{PetscInt}(0)

    error = ccall((:MatGetSize, libpetsc), PetscErrorCode, (CMat, Ref{PetscInt}, Ref{PetscInt}), mat, nrows_glo, ncols_glo)
    @assert iszero(error)

    return nrows_glo[], ncols_glo[]
end

"""
    MatGetType(mat::PetscMat)

Wrapper to `MatGetType`. Return the matrix type as a string. See matrix types here:
https://petsc.org/release/docs/manualpages/Mat/MatType.html
"""
function MatGetType(mat::PetscMat)
    type = Ref{Cstring}()

    error = ccall((:MatGetType, libpetsc), PetscErrorCode, (CMat, Ptr{Cstring}), mat, type)
    @assert iszero(error)

    return unsafe_string(type[])
end


"""
    MatMult(mat::PetscMat, x::PetscVec, y::PetscVec)

Wrapper to `MatMult`. Computes `y = Ax`

https://petsc.org/main/docs/manualpages/Mat/MatMult.html
"""
function MatMult(mat::PetscMat, x::PetscVec, y::PetscVec)
    error = ccall((:MatMult, libpetsc), PetscErrorCode, (CMat, CVec, CVec), mat, x, y)
    @assert iszero(error)
end

"""
    MatMultAdd(A::PetscMat, v1::PetscVec, v2::PetscVec, v3::PetscVec)

Wrapper to `MatMultAdd`. Computes `v3 = v2 + A * v1`.

https://petsc.org/main/docs/manualpages/Mat/MatMultAdd.html
"""
function MatMultAdd(A::PetscMat, v1::PetscVec, v2::PetscVec, v3::PetscVec)
    error = ccall((:MatMultAdd, libpetsc), PetscErrorCode, (CMat, CVec, CVec, CVec), A, v1, v2, v3)
    @assert iszero(error)
end

"""
    MatSetFromOptions(mat::PetscMat)

Wrapper to MatSetFromOptions
"""
function MatSetFromOptions(mat::PetscMat)
    error = ccall((:MatSetFromOptions, libpetsc), PetscErrorCode, (CMat,), mat)
    @assert iszero(error)
end

"""
    MatSetSizes(mat::PetscMat, nrows_loc::PetscInt, ncols_loc::PetscInt, nrows_glo::PetscInt, ncols_glo::PetscInt)

Wrapper to MatSetSizes
"""
function MatSetSizes(mat::PetscMat, nrows_loc::PetscInt, ncols_loc::PetscInt, nrows_glo::PetscInt, ncols_glo::PetscInt)
    error = ccall((:MatSetSizes, libpetsc),
                PetscErrorCode,
                (CMat, PetscInt, PetscInt, PetscInt, PetscInt),
                mat, nrows_loc, ncols_loc, nrows_glo, ncols_glo
            )
    @assert iszero(error)
end

function MatSetSizes(mat::PetscMat, nrows_loc::Integer, ncols_loc::Integer, nrows_glo::Integer, ncols_glo::Integer)
    MatSetSizes(mat, PetscInt(nrows_loc), PetscInt(ncols_loc), PetscInt(nrows_glo), PetscInt(ncols_glo))
end

"""
    MatSetType(mat::PetscMat, type::String)

Wrapper for `MatSetType`. Values for `type` alors available here:
https://petsc.org/release/docs/manualpages/Mat/MatType.html#MatType
"""
function MatSetType(mat::PetscMat, type::String)
    error = ccall((:MatSetType, libpetsc), PetscErrorCode, (CMat, Cstring), mat, type)
    @assert iszero(error)
end


"""
    MatSetUp(mat::PetscMat)

Wrapper to MatSetUp
"""
function MatSetUp(mat::PetscMat)
    error = ccall((:MatSetUp, libpetsc), PetscErrorCode, (CMat,), mat)
    @assert iszero(error)
end

"""
    MatMPIAIJSetPreallocation(mat::PetscMat, dnz::PetscInt, dnnz::Vector{PetscInt}, onz::PetscInt, onnz::Vector{PetscInt})

Wrapper to `MatMPIAIJSetPreallocation`. `dnnz` and `onnz` not tested yet.

# Arguments
- `dnz::PetscInt`: number of nonzeros per row in DIAGONAL portion of local submatrix (same value is used for all local rows)
- `dnnz::Vector{PetscInt}`: array containing the number of nonzeros in the various rows of the DIAGONAL portion of the local
   submatrix (possibly different for each row) or NULL (PETSC_NULL_INTEGER in Fortran), if d_nz is used to specify the nonzero
   structure. The size of this array is equal to the number of local rows, i.e 'm'. For matrices that will be factored, you
   must leave room for (and set) the diagonal entry even if it is zero.
- `onz::PetscInt`: number of nonzeros per row in the OFF-DIAGONAL portion of local submatrix (same value is used for all local rows).
- `onnz::Vector{PetscInt}`: array containing the number of nonzeros in the various rows of the OFF-DIAGONAL portion of the local
   submatrix (possibly different for each row) or NULL (PETSC_NULL_INTEGER in Fortran), if o_nz is used to specify the nonzero structure.
   The size of this array is equal to the number of local rows, i.e 'm'.
"""
function MatMPIAIJSetPreallocation(mat::PetscMat, dnz::PetscInt, dnnz::Vector{PetscInt}, onz::PetscInt, onnz::Vector{PetscInt})
    error = ccall((:MatMPIAIJSetPreallocation, libpetsc), PetscErrorCode,
        (CMat, PetscInt, Ptr{PetscInt}, PetscInt, Ptr{PetscInt}),
        mat, dnz, dnnz, onz, onnz)
    @assert iszero(error)
end

function MatMPIAIJSetPreallocation(mat::PetscMat, dnz::PetscInt, onz::PetscInt)
    error = ccall((:MatMPIAIJSetPreallocation, libpetsc), PetscErrorCode,
        (CMat, PetscInt, Ptr{PetscInt}, PetscInt, Ptr{PetscInt}),
        mat, dnz, C_NULL, onz, C_NULL)
    @assert iszero(error)
end

"""
    MatSeqAIJSetPreallocation(mat::PetscMat, nz::PetscInt, nnz::Vector{PetscInt})

Wrapper to `MatSeqAIJSetPreallocation`.
"""
function MatSeqAIJSetPreallocation(mat::PetscMat, nz::PetscInt, nnz::Vector{PetscInt})
    error = ccall((:MatSeqAIJSetPreallocation, libpetsc), PetscErrorCode,
        (CMat, PetscInt, Ptr{PetscInt}),
        mat, nz, nnz)
    @assert iszero(error)
end

function MatSeqAIJSetPreallocation(mat::PetscMat, nz::PetscInt)
    error = ccall((:MatSeqAIJSetPreallocation, libpetsc), PetscErrorCode,
        (CMat, PetscInt, Ptr{PetscInt}),
        mat, nz, C_NULL)
    @assert iszero(error)
end

"""
Wrapper for `MatSetOption`
"""
function MatSetOption(mat::PetscMat, option::MatOption, value::Bool)
    error = ccall((:MatSetOption, libpetsc), PetscErrorCode, (CMat, MatOption, Cuchar), mat, option, value)
    @assert iszero(error)
end

"""
MatSetValue(mat::PetscMat, i::PetscInt, j::PetscInt, v::PetscScalar, mode::InsertMode)

Wrapper to `MatSetValue`. Indexing starts at 0 (as in PETSc).

# Implementation
For an unknow reason, calling PETSc.MatSetValue leads to an "undefined symbol: MatSetValue" error.
So this wrapper directly call MatSetValues (anyway, this is what is done in PETSc...)
"""
function MatSetValue(mat::PetscMat, i::PetscInt, j::PetscInt, v::PetscScalar, mode::InsertMode)
    MatSetValues(mat, PetscIntOne, [i], PetscIntOne, [j], [v], mode)
end

function MatSetValue(mat::PetscMat, i::Integer, j::Integer, v::Number, mode::InsertMode)
    MatSetValue(mat, PetscInt(i), PetscInt(j), PetscScalar(v), mode)
end

"""
MatSetValues(mat::PetscMat, nI::PetscInt, I::Vector{PetscInt}, nJ::PetscInt, J::Vector{PetscInt}, V::Array{PetscScalar}, mode::InsertMode)

Wrapper to `MatSetValues`. Indexing starts at 0 (as in PETSc)
"""
function MatSetValues(mat::PetscMat, nI::PetscInt, I::Vector{PetscInt}, nJ::PetscInt, J::Vector{PetscInt}, V::Array{PetscScalar}, mode::InsertMode)
    error = ccall((:MatSetValues, libpetsc), PetscErrorCode,
        (CMat, PetscInt, Ptr{PetscInt}, PetscInt, Ptr{PetscInt}, Ptr{PetscScalar}, InsertMode),
        mat, nI, I, nJ, J, V, mode)
    @assert iszero(error)
end

function MatSetValues(mat::PetscMat, I::Vector{PetscInt}, J::Vector{PetscInt}, V::Array{PetscScalar}, mode::InsertMode)
    MatSetValues(mat, PetscInt(length(I)), I, PetscInt(length(J)), J, V, mode)
end

function MatSetValues(mat::PetscMat, I, J, V, mode::InsertMode)
    MatSetValues(mat, PetscInt.(I), PetscInt.(collect(J)), PetscScalar.(V), mode)
end

"""
    MatView(mat::PetscMat, viewer::PetscViewer = PetscViewerStdWorld())

Wrapper to `MatView`
"""
function MatView(mat::PetscMat, viewer::PetscViewer = PetscViewerStdWorld())
    error = ccall((:MatView, libpetsc), PetscErrorCode, (CMat, CViewer), mat, viewer)
    @assert iszero(error)
end