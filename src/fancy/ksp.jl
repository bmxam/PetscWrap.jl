function create_ksp(Amat::Mat, Pmat::Mat; autosetup = false)
    ksp = create(KSP, _get_comm(Amat))
    setOperators(ksp, Amat, Pmat)

    if autosetup
        set_from_options!(ksp)
        set_up!(ksp)
    end
    return ksp
end

create_ksp(A::Mat; autosetup = false) = create_ksp(A, A; autosetup)

set_operators!(ksp::KSP, Amat::Mat) = setOperators(ksp, Amat, Amat)
set_operators!(ksp::KSP, Amat::Mat, Pmat::Mat) = setOperators(ksp, Amat, Pmat)

solve!(ksp::KSP, b::Vec, x::Vec) = solve(ksp, b, x)

function solve(ksp::KSP, b::Vec)
    x = duplicate(b)
    solve(ksp, b, x)
    return x
end

Base.show(::IO, ksp::KSP) = KSPView(ksp)
