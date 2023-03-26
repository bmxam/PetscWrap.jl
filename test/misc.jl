@testset "composite" begin
    PetscInitialize()

    # Create two matrices
    A = create_matrix(2, 2; auto_setup = true)
    B = create_matrix(2, 2; auto_setup = true)

    # Fill

    # A
    # 1 -
    # 3 4
    A[1, 1] = 1
    A[2, 1] = 3
    A[2, 2] = 4

    # B
    # 4 3
    # - 1
    B[1, 1] = 4
    B[1, 2] = 3
    B[2, 2] = 1

    # Assemble matrices
    assemble!.((A, B))

    # Create composite mat from A and B
    # C
    #  5 3
    #  3 5
    C = create_composite_add([A, B])

    # Create vectors to check C (see below)
    x1 = create_vector(2; auto_setup = true)
    x2 = create_vector(2; auto_setup = true)
    y = create_vector(2; auto_setup = true)
    x1[1] = 1.0
    x1[2] = 0.0
    x2[1] = 0.0
    x2[2] = 1.0
    assemble!.((x1, x2, y))

    # Check result by multiplying with test vectors
    mult(C, x1, y)
    @test vec2array(y) == [5.0, 3.0]

    mult(C, x2, y)
    @test vec2array(y) == [3.0, 5.0]

    # Free memory
    destroy!.((A, B, C, x1, x2, y))
    PetscFinalize()
end