comm = MPI.COMM_WORLD

@testset "composite" begin

    # Create two matrices
    A = create_matrix(comm; nrows_glo = 2, ncols_glo = 2, autosetup = true)
    B = create_matrix(comm; nrows_glo = 2, ncols_glo = 2, autosetup = true)

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
    x1 = create_vector(comm; nrows_glo = 2, autosetup = true)
    x2 = create_vector(comm; nrows_glo = 2, autosetup = true)
    y = create_vector(comm; nrows_glo = 2, autosetup = true)
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
    destroy!(A, B, C, x1, x2, y)
end

@testset "mapping" begin
    n = 10
    x = create_vector(comm; nrows_glo = n, autosetup = true)
    y = create_vector(comm; nrows_glo = n, autosetup = true)
    l2g = reverse(collect(1:n))
    set_local_to_global!(x, l2g)

    set_value_local!(x, 1, 1.0)
    set_value!(y, 1, 33)
    set_value!(y, 10, 22)

    assemble!.((x, y))

    # @show dot(x,y)
    @test dot(x, x) == 1.0
    @test dot(x, y) == 22.0
    @test sum(x) == 1.0

    _x = vec2array(x)
    @test _x[end] == 1.0

    destroy!(x, y)
end