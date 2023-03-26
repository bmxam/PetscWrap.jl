create_ksp() = KSPCreate()

function create_ksp(A::Mat)
    ksp = create(KSP)
    setOperators(ksp, A, A)
    return ksp
end

function create_ksp(Amat::Mat, Pmat::Mat)
    ksp = create(KSP)
    setOperators(ksp, Amat, Pmat)
    return ksp
end

set_operators!(ksp::KSP, Amat::Mat) = setOperators(ksp, Amat, Amat)
set_operators!(ksp::KSP, Amat::Mat, Pmat::Mat) = setOperators(ksp, Amat, Pmat)

solve!(ksp::KSP, b::Vec, x::Vec) = solve(ksp, b, x)

function solve(ksp::KSP, b::Vec)
    x = duplicate(b)
    solve(ksp, b, x)
    return x
end

Base.show(::IO, ksp::KSP) = KSPView(ksp)
