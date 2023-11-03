function create_ksp(Amat::Mat, Pmat::Mat; autosetup = false, add_finalizer = true)
    ksp = create(KSP, _get_comm(Amat); add_finalizer)
    setOperators(ksp, Amat, Pmat)

    if autosetup
        set_from_options!(ksp)
        set_up!(ksp)
    end
    return ksp
end

create_ksp(A::Mat; kwargs...) = create_ksp(A, A; kwargs...)

set_operators!(ksp::KSP, Amat::Mat) = setOperators(ksp, Amat, Amat)
set_operators!(ksp::KSP, Amat::Mat, Pmat::Mat) = setOperators(ksp, Amat, Pmat)

solve!(ksp::KSP, b::Vec, x::Vec) = solve(ksp, b, x)

function solve(ksp::KSP, b::Vec)
    x = duplicate(b)
    solve(ksp, b, x)
    return x
end

Base.show(::IO, ksp::KSP) = KSPView(ksp)
