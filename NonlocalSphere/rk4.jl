function RK4(SLPDE::SemiLinearPDE{1, S}, T::Number, n::Int) where S
    U0 = SLPDE.U0[1]
    U = deepcopy(U0) # solution spherical harmonic coefficients
    V = zero(U) # solution Fourier coefficients
    F = zero(U) # solution function values
    NU = zero(U) # nonlinearity spherical harmonic coefficients
    NV = zero(U) # nonlinearity Fourier coefficients
    NF = zero(U) # nonlinearity function values

    # RK4 stages
    K1 = zero(U)
    K2 = zero(U)
    K3 = zero(U)
    K4 = zero(U)
    NK1 = zero(U)
    NK2 = zero(U)
    NK3 = zero(U)

    P = plan_sph2fourier(U; sketch = :none)
    Ps = FastTransforms.plan_synthesis(U)
    Pa = FastTransforms.plan_analysis(U)

    L = create_linear_operator(SLPDE.L, size(U, 1), size(U, 2)÷2 + 1)
    h = S(T)/n

    for k=1:n
        evaluate_nonlinear_operator!(SLPDE.N, P, Ps, Pa, U, V, F, NF, NV, NU)
        K1 .= L.*U .+ NU
        NK1 .= U .+ (0.5h).*K1
        evaluate_nonlinear_operator!(SLPDE.N, P, Ps, Pa, NK1, V, F, NF, NV, NU)
        K2 .= L.*NK1 .+ NU
        NK2 .= U .+ (0.5h).*K2
        evaluate_nonlinear_operator!(SLPDE.N, P, Ps, Pa, NK2, V, F, NF, NV, NU)
        K3 .= L.*NK2 .+ NU
        NK3 .= U .+ h.*K3
        evaluate_nonlinear_operator!(SLPDE.N, P, Ps, Pa, NK3, V, F, NF, NV, NU)
        K4 .= L.*NK3 .+ NU
        U .= U + (h/6.0).*(K1 .+ 2.0.*(K2 .+ K3) .+ K4)
    end

    U
end

function RK4(SLPDE::SemiLinearPDE{2, S}, T::Number, n::Int) where S
    U0 = SLPDE.U0
    U = deepcopy(U0) # solution spherical harmonic coefficients
    V = zero(U[1]), zero(U[2]) # solution Fourier coefficients
    F = zero(U[1]), zero(U[2]) # solution function values
    NU = zero(U[1]), zero(U[2]) # nonlinearity spherical harmonic coefficients
    NV = zero(U[1]), zero(U[2]) # nonlinearity Fourier coefficients
    NF = zero(U[1]), zero(U[2]) # nonlinearity function values

    # RK4 stages
    K1 = zero(U[1]), zero(U[2])
    K2 = zero(U[1]), zero(U[2])
    K3 = zero(U[1]), zero(U[2])
    K4 = zero(U[1]), zero(U[2])
    NK1 = zero(U[1]), zero(U[2])
    NK2 = zero(U[1]), zero(U[2])
    NK3 = zero(U[1]), zero(U[2])

    P = plan_sph2fourier(U[1]; sketch = :none), plan_sph2fourier(U[2]; sketch = :none)
    Ps = FastTransforms.plan_synthesis(U[1]), FastTransforms.plan_synthesis(U[2])
    Pa = FastTransforms.plan_analysis(U[1]), FastTransforms.plan_analysis(U[2])

    L = create_linear_operator(SLPDE.L[1], size(U[1], 1), size(U[1], 2)÷2 + 1), create_linear_operator(SLPDE.L[2], size(U[2], 1), size(U[2], 2)÷2 + 1)
    h = S(T)/n

    for k=1:n
        evaluate_nonlinear_operator!(SLPDE.N, P, Ps, Pa, U, V, F, NF, NV, NU)
        K1[1] .= L[1].*U[1] .+ NU[1]
        K1[2] .= L[2].*U[2] .+ NU[2]
        NK1[1] .= U[1] .+ (0.5h).*K1[1]
        NK1[2] .= U[2] .+ (0.5h).*K1[2]
        evaluate_nonlinear_operator!(SLPDE.N, P, Ps, Pa, NK1, V, F, NF, NV, NU)
        K2[1] .= L[1].*NK1[1] .+ NU[1]
        K2[2] .= L[2].*NK1[2] .+ NU[2]
        NK2[1] .= U[1] .+ (0.5h).*K2[1]
        NK2[2] .= U[2] .+ (0.5h).*K2[2]
        evaluate_nonlinear_operator!(SLPDE.N, P, Ps, Pa, NK2, V, F, NF, NV, NU)
        K3[1] .= L[1].*NK2[1] .+ NU[1]
        K3[2] .= L[2].*NK2[2] .+ NU[2]
        NK3[1] .= U[1] .+ h.*K3[1]
        NK3[2] .= U[2] .+ h.*K3[2]
        evaluate_nonlinear_operator!(SLPDE.N, P, Ps, Pa, NK3, V, F, NF, NV, NU)
        K4[1] .= L[1].*NK3[1] .+ NU[1]
        K4[2] .= L[2].*NK3[2] .+ NU[2]
        U[1] .= U[1] + (h/6.0).*(K1[1] .+ 2.0.*(K2[1] .+ K3[1]) .+ K4[1])
        U[2] .= U[2] + (h/6.0).*(K1[2] .+ 2.0.*(K2[2] .+ K3[2]) .+ K4[2])
    end

    U
end
