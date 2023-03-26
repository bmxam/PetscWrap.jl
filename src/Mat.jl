const CMat = Ptr{Cvoid}

"""
    A Petsc matrix, actually just a pointer to the actual C matrix
"""
struct Mat
    ptr::Ref{CMat}
    comm::MPI.Comm

    Mat(comm::MPI.Comm) = new(Ref{CMat}(), comm)
end

# allows us to pass Mat objects directly into CMat ccall signatures
Base.cconvert(::Type{CMat}, mat::Mat) = mat.ptr[]

"""
    assemblyBegin(mat::Mat, type::MatAssemblyType)

Wrapper to `MatAssemblyBegin`
https://petsc.org/release/docs/manualpages/Mat/MatAssemblyBegin/
"""
function assemblyBegin(mat::Mat, type::MatAssemblyType)
    error = ccall(
        (:MatAssemblyBegin, libpetsc),
        PetscErrorCode,
        (CMat, MatAssemblyType),
        mat,
        type,
    )
    @assert iszero(error)
end

"""
    assemblyEnd(mat::Mat, type::MatAssemblyType)

Wrapper to `MatAssemblyEnd`
https://petsc.org/release/docs/manualpages/Mat/MatAssemblyEnd/
"""
function assemblyEnd(mat::Mat, type::MatAssemblyType)
    error = ccall(
        (:MatAssemblyEnd, libpetsc),
        PetscErrorCode,
        (CMat, MatAssemblyType),
        mat,
        type,
    )
    @assert iszero(error)
end

"""
    compositeAddMat(mat::Mat, smat::Mat)

Wrapper to `MatCompositeAddMat`
https://petsc.org/release/docs/manualpages/Mat/MatCompositeAddMat/
"""
function compositeAddMat(mat::Mat, smat::Mat)
    error = ccall((:MatCompositeAddMat, libpetsc), PetscErrorCode, (CMat, CMat), mat, smat)
    @assert iszero(error)
end

"""
    create(::Type{Mat}, comm::MPI.Comm)

Wrapper to `MatCreate`
https://petsc.org/release/docs/manualpages/Mat/MatCreate/
"""
function create(::Type{Mat}, comm::MPI.Comm)
    mat = Mat(comm)
    error = ccall(
        (:MatCreate, libpetsc),
        PetscErrorCode,
        (MPI.MPI_Comm, Ptr{CMat}),
        comm,
        mat.ptr,
    )
    @assert iszero(error)
    return mat
end

"""
    createDense(comm::MPI.Comm, m::PetscInt, n::PetscInt, M::PetscInt, N::PetscInt)
    createDense(
        comm::MPI.Comm,
        m::Integer = PETSC_DECIDE,
        n::Integer = PETSC_DECIDE,
        M::Integer = PETSC_DECIDE,
        N::Integer = PETSC_DECIDE,
    )

Wrapper to `MatCreateDense`
https://petsc.org/release/docs/manualpages/Mat/MatCreateDense/

Last argument `data` is not supported yet (NULL is passed).
"""
function createDense(comm::MPI.Comm, m::PetscInt, n::PetscInt, M::PetscInt, N::PetscInt)
    mat = Mat(comm)
    error = ccall(
        (:MatCreateDense, libpetsc),
        PetscErrorCode,
        (MPI.MPI_Comm, PetscInt, PetscInt, PetscInt, PetscInt, Ptr{PetscScalar}, Ptr{CMat}),
        comm,
        m,
        n,
        M,
        N,
        C_NULL,
        mat.ptr,
    )
    @assert iszero(error)
    return mat
end

function createDense(
    comm::MPI.Comm,
    m::Integer = PETSC_DECIDE,
    n::Integer = PETSC_DECIDE,
    M::Integer = PETSC_DECIDE,
    N::Integer = PETSC_DECIDE,
)
    return createDense(comm, PetscInt(m), PetscInt(n), PetscInt(M), PetscInt(N))
end

"""
    createVecs(mat::Mat, vecr::PetscVec, veci::PetscVec)
    createVecs(mat::Mat)

Wrapper to `MatCreateVecs`
https://petsc.org/release/docs/manualpages/Mat/MatCreateVecs/
"""
function createVecs(mat::Mat, right::PetscVec, left::PetscVec)
    error = ccall(
        (:MatCreateVecs, libpetsc),
        PetscErrorCode,
        (CMat, Ptr{CVec}, Ptr{CVec}),
        mat,
        right.ptr,
        left.ptr,
    )
    @assert iszero(error)
end

function createVecs(mat::Mat)
    right = PetscVec(mat.comm)
    left = PetscVec(mat.comm)
    createVecs(mat, right, left)
    return right, left
end

"""
    destroy(A::Mat)

Wrapper to `MatDestroy`
https://petsc.org/release/docs/manualpages/Mat/MatDestroy/
"""
function destroy(A::Mat)
    error = ccall((:MatDestroy, libpetsc), PetscErrorCode, (Ptr{CMat},), A.ptr)
    @assert iszero(error)
end

"""
    getLocalSize(mat::Mat)

Wrapper to `MatGetLocalSize`
https://petsc.org/release/docs/manualpages/Mat/MatGetLocalSize/

Return the number of local rows and cols of the matrix (i.e on the processor).
"""
function getLocalSize(mat::Mat)
    m = Ref{PetscInt}(0)
    n = Ref{PetscInt}(0)

    error = ccall(
        (:MatGetLocalSize, libpetsc),
        PetscErrorCode,
        (CMat, Ref{PetscInt}, Ref{PetscInt}),
        mat,
        m,
        n,
    )
    @assert iszero(error)

    return m[], n[]
end

"""
    getOwnershipRange(mat::Mat)

Wrapper to `MatGetOwnershipRange`
https://petsc.org/release/docs/manualpages/Mat/MatGetOwnershipRange/

The result `(rstart, rend)` is a Tuple indicating the rows handled by the local processor.

# Warning

`PETSc` indexing starts at zero (so `rstart` may be zero) and `rend-1` is the last row
handled by the local processor.
"""
function getOwnershipRange(mat::Mat)
    rstart = Ref{PetscInt}(0)
    rend = Ref{PetscInt}(0)

    error = ccall(
        (:MatGetOwnershipRange, libpetsc),
        PetscErrorCode,
        (CMat, Ref{PetscInt}, Ref{PetscInt}),
        mat,
        rstart,
        rend,
    )
    @assert iszero(error)

    return rstart[], rend[]
end

"""
    getOwnershipRangeColumn(mat::Mat)

Wrapper to `MatGetOwnershipRangeColumn`
https://petsc.org/release/docs/manualpages/Mat/MatGetOwnershipRangeColumn/

The result `(cstart, cend)` is a Tuple indicating the columns handled by the local processor.

# Warning

`PETSc` indexing starts at zero (so `cstart` may be zero) and `cend-1` is the last column
handled by the local processor.
"""
function getOwnershipRangeColumn(mat::Mat)
    cstart = Ref{PetscInt}(0)
    cend = Ref{PetscInt}(0)

    error = ccall(
        (:MatGetOwnershipRangeColumn, libpetsc),
        PetscErrorCode,
        (CMat, Ref{PetscInt}, Ref{PetscInt}),
        mat,
        cstart,
        cend,
    )
    @assert iszero(error)

    return cstart[], cend[]
end

"""
    getSize(mat::Mat)

Wrapper to `MatGetSize`
https://petsc.org/release/docs/manualpages/Mat/MatGetSize/

Return the number of rows and cols of the matrix (global number).
"""
function getSize(mat::Mat)
    nrows_glo = Ref{PetscInt}(0)
    ncols_glo = Ref{PetscInt}(0)

    error = ccall(
        (:MatGetSize, libpetsc),
        PetscErrorCode,
        (CMat, Ref{PetscInt}, Ref{PetscInt}),
        mat,
        nrows_glo,
        ncols_glo,
    )
    @assert iszero(error)

    return nrows_glo[], ncols_glo[]
end

"""
    getType(mat::Mat)

Wrapper to `MatGetType`
https://petsc.org/release/docs/manualpages/Mat/MatType/

Return the matrix type as a string. See matrix types here:
https://petsc.org/release/docs/manualpages/Mat/MatType.html
"""
function getType(mat::Mat)
    type = Ref{Cstring}()

    error = ccall((:MatGetType, libpetsc), PetscErrorCode, (CMat, Ptr{Cstring}), mat, type)
    @assert iszero(error)

    return unsafe_string(type[])
end

"""
    MPIAIJSetPreallocation(
        B::Mat,
        d_nz::PetscInt,
        d_nnz::Vector{PetscInt},
        o_nz::PetscInt,
        o_nnz::Vector{PetscInt},
    )
    MPIAIJSetPreallocation(mat::Mat, dnz::PetscInt, onz::PetscInt)

Wrapper to `MatMPIAIJSetPreallocation`
https://petsc.org/release/docs/manualpages/Mat/MatMPIAIJSetPreallocation/

# Warning

`dnnz` and `onnz` not tested yet.

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
function MPIAIJSetPreallocation(
    B::Mat,
    d_nz::PetscInt,
    d_nnz::Vector{PetscInt},
    o_nz::PetscInt,
    o_nnz::Vector{PetscInt},
)
    error = ccall(
        (:MatMPIAIJSetPreallocation, libpetsc),
        PetscErrorCode,
        (CMat, PetscInt, Ptr{PetscInt}, PetscInt, Ptr{PetscInt}),
        B,
        d_nz,
        d_nnz,
        o_nz,
        o_nnz,
    )
    @assert iszero(error)
end

function MPIAIJSetPreallocation(mat::Mat, dnz::PetscInt, onz::PetscInt)
    error = ccall(
        (:MatMPIAIJSetPreallocation, libpetsc),
        PetscErrorCode,
        (CMat, PetscInt, Ptr{PetscInt}, PetscInt, Ptr{PetscInt}),
        mat,
        dnz,
        C_NULL,
        onz,
        C_NULL,
    )
    @assert iszero(error)
end

"""
    mult(mat::Mat, x::PetscVec, y::PetscVec)

Wrapper to `MatMult`
https://petsc.org/release/docs/manualpages/Mat/MatMult/

Compute `y = Ax`
"""
function mult(mat::Mat, x::PetscVec, y::PetscVec)
    error = ccall((:MatMult, libpetsc), PetscErrorCode, (CMat, CVec, CVec), mat, x, y)
    @assert iszero(error)
end

"""
    multAdd(A::Mat, v1::PetscVec, v2::PetscVec, v3::PetscVec)

Wrapper to `MatMultAdd`
https://petsc.org/release/docs/manualpages/Mat/MatMultAdd/

Compute `v3 = v2 + A * v1`.
"""
function multAdd(A::Mat, v1::PetscVec, v2::PetscVec, v3::PetscVec)
    error = ccall(
        (:MatMultAdd, libpetsc),
        PetscErrorCode,
        (CMat, CVec, CVec, CVec),
        A,
        v1,
        v2,
        v3,
    )
    @assert iszero(error)
end

"""
    SeqAIJSetPreallocation(mat::Mat, nz::PetscInt, nnz::Vector{PetscInt})
    SeqAIJSetPreallocation(mat::Mat, nz::PetscInt)

Wrapper to `MatSeqAIJSetPreallocation`
https://petsc.org/release/docs/manualpages/Mat/MatSeqAIJSetPreallocation/
"""
function SeqAIJSetPreallocation(mat::Mat, nz::PetscInt, nnz::Vector{PetscInt})
    error = ccall(
        (:MatSeqAIJSetPreallocation, libpetsc),
        PetscErrorCode,
        (CMat, PetscInt, Ptr{PetscInt}),
        mat,
        nz,
        nnz,
    )
    @assert iszero(error)
end

function SeqAIJSetPreallocation(mat::Mat, nz::PetscInt)
    error = ccall(
        (:MatSeqAIJSetPreallocation, libpetsc),
        PetscErrorCode,
        (CMat, PetscInt, Ptr{PetscInt}),
        mat,
        nz,
        C_NULL,
    )
    @assert iszero(error)
end

"""
    setFromOptions(mat::Mat)

Wrapper to `MatSetFromOptions`
https://petsc.org/release/docs/manualpages/Mat/MatSetFromOptions/
"""
function setFromOptions(mat::Mat)
    error = ccall((:MatSetFromOptions, libpetsc), PetscErrorCode, (CMat,), mat)
    @assert iszero(error)
end

"""
    setOption(mat::Mat, op::MatOption, flg::PetscBool)

Wrapper for `MatSetOption`
https://petsc.org/release/docs/manualpages/Mat/MatSetOption/
"""
function setOption(mat::Mat, op::MatOption, flg::PetscBool)
    error = ccall(
        (:MatSetOption, libpetsc),
        PetscErrorCode,
        (CMat, MatOption, Cuchar),
        mat,
        op,
        flg,
    )
    @assert iszero(error)
end

setOption(mat::Mat, op::MatOption, flg::Bool) = setOption(mat, op, bool2petsc(flg))

"""
    setSizes(mat::Mat, m::PetscInt, n::PetscInt, M::PetscInt, N::PetscInt)
    setSizes(
        mat::Mat,
        nrows_loc::Integer,
        ncols_loc::Integer,
        nrows_glo::Integer,
        ncols_glo::Integer,
    )

Wrapper to `MatSetSizes`
https://petsc.org/release/docs/manualpages/Mat/MatSetSizes/
"""
function setSizes(mat::Mat, m::PetscInt, n::PetscInt, M::PetscInt, N::PetscInt)
    error = ccall(
        (:MatSetSizes, libpetsc),
        PetscErrorCode,
        (CMat, PetscInt, PetscInt, PetscInt, PetscInt),
        mat,
        m,
        n,
        M,
        N,
    )
    @assert iszero(error)
end

function setSizes(
    mat::Mat,
    nrows_loc::Integer,
    ncols_loc::Integer,
    nrows_glo::Integer,
    ncols_glo::Integer,
)
    setSizes(
        mat,
        PetscInt(nrows_loc),
        PetscInt(ncols_loc),
        PetscInt(nrows_glo),
        PetscInt(ncols_glo),
    )
end

"""
    setType(mat::Mat, type::String)

Wrapper for `MatSetType`
https://petsc.org/release/docs/manualpages/Mat/MatSetType/

Values for `type` alors available here:
https://petsc.org/release/docs/manualpages/Mat/MatType.html#MatType
"""
function setType(mat::Mat, type::String)
    error = ccall((:MatSetType, libpetsc), PetscErrorCode, (CMat, Cstring), mat, type)
    @assert iszero(error)
end

"""
    setUp(mat::Mat)

Wrapper to `MatSetUp`
https://petsc.org/release/docs/manualpages/Mat/MatSetUp/
"""
function setUp(mat::Mat)
    error = ccall((:MatSetUp, libpetsc), PetscErrorCode, (CMat,), mat)
    @assert iszero(error)
end

# Avoid allocating an array of size 1 for each call to MatSetValue
const _ivec = zeros(PetscInt, 1)
const _jvec = zeros(PetscInt, 1)
const _vvec = zeros(PetscScalar, 1)

"""
    setValue(
        m::Mat,
        row::PetscInt,
        col::PetscInt,
        value::PetscScalar,
        mode::InsertMode,
    )
    setValue(m::Mat, row::Integer, col::Integer, value::Number, mode::InsertMode)

Wrapper to `MatSetValue`
https://petsc.org/release/docs/manualpages/Mat/MatSetValue/

Indexing starts at 0 (as in PETSc).

# Implementation

For an unknow reason, calling PETSc.MatSetValue leads to an "undefined symbol: MatSetValue" error.
So this wrapper directly call MatSetValues (anyway, this is what is done in PETSc...)
"""
function setValue(
    m::Mat,
    row::PetscInt,
    col::PetscInt,
    value::PetscScalar,
    mode::InsertMode,
)
    # Convert to arrays
    _ivec[1] = row
    _jvec[1] = col
    _vvec[1] = value

    setValues(m, PetscIntOne, _ivec, PetscIntOne, _jvec, _vvec, mode)
    #MatSetValues(mat, PetscIntOne, [i], PetscIntOne, [j], [v], mode)
end

function setValue(m::Mat, row::Integer, col::Integer, value::Number, mode::InsertMode)
    setValue(m, PetscInt(row), PetscInt(col), PetscScalar(value), mode)
end

"""
    setValues(
        mat::Mat,
        m::PetscInt,
        idxm::Vector{PetscInt},
        n::PetscInt,
        idxn::Vector{PetscInt},
        V::Array{PetscScalar},
        addv::InsertMode,
    )
    setValues(
        mat::Mat,
        I::Vector{PetscInt},
        J::Vector{PetscInt},
        V::Array{PetscScalar},
        mode::InsertMode,
    )
    setValues(mat::Mat, I, J, V, mode::InsertMode)

Wrapper to `MatSetValues`
https://petsc.org/release/docs/manualpages/Mat/MatSetValues/

Indexing starts at 0 (as in PETSc)
"""
function setValues(
    mat::Mat,
    m::PetscInt,
    idxm::Vector{PetscInt},
    n::PetscInt,
    idxn::Vector{PetscInt},
    V::Array{PetscScalar},
    addv::InsertMode,
)
    error = ccall(
        (:MatSetValues, libpetsc),
        PetscErrorCode,
        (
            CMat,
            PetscInt,
            Ptr{PetscInt},
            PetscInt,
            Ptr{PetscInt},
            Ptr{PetscScalar},
            InsertMode,
        ),
        mat,
        m,
        idxm,
        n,
        idxn,
        V,
        addv,
    )
    @assert iszero(error)
end

function setValues(
    mat::Mat,
    I::Vector{PetscInt},
    J::Vector{PetscInt},
    V::Array{PetscScalar},
    mode::InsertMode,
)
    setValues(mat, PetscInt(length(I)), I, PetscInt(length(J)), J, V, mode)
end

function setValues(mat::Mat, I, J, V, mode::InsertMode)
    setValues(mat, PetscInt.(I), PetscInt.(collect(J)), PetscScalar.(V), mode)
end

"""
    view(mat::Mat, viewer::PetscViewer = PetscViewerStdWorld())

Wrapper to `MatView`
https://petsc.org/release/docs/manualpages/Mat/MatView/
"""
function view(mat::Mat, viewer::PetscViewer = PetscViewerStdWorld())
    error = ccall((:MatView, libpetsc), PetscErrorCode, (CMat, CViewer), mat, viewer)
    @assert iszero(error)
end
