
comm = MPI.COMM_WORLD

@testset "linear system fancy" begin
    # Only on one processor...

    # Number of mesh points and mesh step
    n = 11
    Δx = 1.0 / (n - 1)

    # Create a matrix of size `(n,n)` and a vector of size `(n)`
    A = create_matrix(comm; nrows_glo = n, ncols_glo = n, autosetup = true)
    b = create_vector(comm; nrows_glo = n, autosetup = true)

    # Let's build the right hand side vector. We first get the range of rows of `b` handled by the local processor.
    # The `rstart, rend = get_range(*)` methods differ a little bit from PETSc API since the two integers it
    # returns are the effective Julia range of rows handled by the local processor. If `n` is the total
    # number of rows, then `rstart ∈ [1,n]` and `rend` is the last row handled by the local processor.
    b_start, b_end = get_range(b)

    # Now let's build the right hand side vector. Their are various ways to do this, this is just one.
    n_loc = length(b) ## Note that n_loc = b_end - b_start + 1...
    b[b_start:b_end] = 2 * ones(n_loc)

    # And here is the differentiation matrix. Rembember that PETSc.MatSetValues simply ignores negatives rows indices.
    A_start, A_end = get_range(A)
    for i = A_start:A_end
        A[i, (i - 1):i] = [-1.0 1.0] / Δx
    end

    # Set boundary condition (only the proc handling index `1` is acting)
    (b_start == 1) && (b[1] = 0.0)

    # Assemble matrice and vector
    assemble!(A)
    assemble!(b)

    # Set up the linear solver
    ksp = create_ksp(A; autosetup = true)

    # Solve the system
    x = solve(ksp, b)

    # Print the solution
    @show x

    # Convert to Julia array
    array = vec2array(x)
    @test isapprox(array, range(0.0, 2.0; length = n))

    # Free memory
    destroy!(A, b, x)

    # Note that it's also possible to build a matrix using the COO format as in `SparseArrays`:
    M = create_matrix(comm; nrows_glo = 3, ncols_glo = 3, autosetup = true)
    M_start, M_end = get_range(M)
    I = [1, 1, 1, 2, 3]
    J = [1, 3, 1, 3, 2]
    V = [1, 2, 3, 4, 5]
    k = findall(x -> M_start <= x <= M_end, I) # just a stupid trick to allow this example to run in parallel
    set_values!(M, I[k], J[k], V[k], ADD_VALUES)
    assemble!(M)
    @show M
    destroy!(M)
    # This is very convenient in sequential since you can fill the three vectors I, J, V in your code and decide only
    # at the last moment if you'd like to use `SparseArrays` or `PetscMat`.

    # Reach this point?
    @test true
end
