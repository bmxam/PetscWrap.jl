@testset "linear system" begin
    # Number of mesh points and mesh step
    n = 11
    Δx = 1.0 / (n - 1)

    println("before create Mat")

    # Create a matrix and a vector
    A = create(Mat)
    println("before create Vec")
    b = create(Vec)

    println("before setSizes")

    # Set the size of the different objects, leaving PETSC to decide how to distribute. Note that we should
    # set the number of preallocated non-zeros to increase performance.
    setSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n)
    setSizes(b, PETSC_DECIDE, n)

    # We can then use command-line options to set our matrix/vectors.
    setFromOptions(A)
    setFromOptions(b)

    # Finish the set up
    setUp(A)
    setUp(b)

    println("before getOwnershipRange")

    # Let's build the right hand side vector. We first get the range of rows of `b` handled by the local processor.
    # As in PETSc, the `rstart, rend = *GetOwnershipRange` methods indicate the first row handled by the local processor
    # (starting at 0), and the last row (which is `rend-1`). This may be disturbing for non-regular PETSc users. Checkout
    # the fancy version of this example for a more Julia-like convention.
    b_start, b_end = getOwnershipRange(b)

    # Now let's build the right hand side vector. Their are various ways to do this, this is just one.
    n_loc = getLocalSize(b) # Note that n_loc = b_end - b_start...
    setValues(b, collect(b_start:(b_end - 1)), 2 * ones(n_loc))

    # And here is the differentiation matrix. Rembember that PETSc.MatSetValues simply ignores negatives rows indices.
    A_start, A_end = getOwnershipRange(A)
    for i = A_start:(A_end - 1)
        setValues(A, [i], [i - 1, i], [-1.0 1.0] / Δx, INSERT_VALUES) # MatSetValues(A, I, J, V, INSERT_VALUES)
    end

    # Set boundary condition (only the proc handling index `0` is acting)
    (b_start == 0) && setValue(b, 0, 0.0)

    # Assemble matrices
    assemblyBegin(A, MAT_FINAL_ASSEMBLY)
    assemblyBegin(b)
    assemblyEnd(A, MAT_FINAL_ASSEMBLY)
    assemblyEnd(b)

    println("before matview")
    # At this point, you can inspect `A` or `b` using a viewer (stdout by default), simply call
    matView(A)
    vecView(b)

    # Set up the linear solver
    ksp = create(KSP)
    setOperators(ksp, A, A)
    setFromOptions(ksp)
    setUp(ksp)

    # Solve the system. We first allocate the solution using the `VecDuplicate` method.
    x = duplicate(b)
    solve(ksp, b, x)

    println("before vecview")
    # Print the solution
    vecView(x)

    # Access the solution (this part is under development), getting a Julia array; and then restore it
    array, ref = getArray(x) # do something with array
    @test isapprox(array, range(0.0, 2.0; length = n))
    restoreArray(x, ref)

    # Free memory
    destroy(A)
    destroy(b)
    destroy(x)

    # Reach this point?
    @test true
end
