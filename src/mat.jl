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
    MatSetUp(mat::PetscMat)

Wrapper to MatSetUp
"""
function MatSetUp(mat::PetscMat)
    error = ccall((:MatSetUp, libpetsc), PetscErrorCode, (CMat,), mat)
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
    MatView(mat::PetscMat, viewer::PetscViewer = PetscViewerStdWorld())

Wrapper to `MatView`
"""
function MatView(mat::PetscMat, viewer::PetscViewer = PetscViewerStdWorld())
    error = ccall((:MatView, libpetsc), PetscErrorCode, (CMat, CViewer), mat, viewer)
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
