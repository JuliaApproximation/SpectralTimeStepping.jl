function ETDRK4(SLPDE::SemiLinearPDE{1, S}, T::Number, n::Int) where S
    U0 = SLPDE.U0[1]
    U = deepcopy(U0) # solution spherical harmonic coefficients
    NU = zero(U) # nonlinearity spherical harmonic coefficients

    # ETDRK4 stages
    A = zero(U)
    B = zero(U)
    C = zero(U)
    NA = zero(U)
    NB = zero(U)
    NC = zero(U)

    P = plan_sph2fourier(U)
    PS = plan_sph_synthesis(U)
    PA = plan_sph_analysis(U)

    L = create_linear_operator(SLPDE.L, size(U, 1), size(U, 2)÷2 + 1)
    h = S(T)/n

    # Create all the matrices of diagonal scalings used in ETDRK4. These operate
    # component-wise on the spherical harmonic coefficients.

    Z = L*h
    ez = sph_zero_spurious_modes!(exp.(Z))
    ez2 = sph_zero_spurious_modes!(exp.(Z./2))
    h2ez2m1 = sph_zero_spurious_modes!((h/2.0).*φ1.(Z./2))
    hezα = sph_zero_spurious_modes!(h.*expα.(Z))
    h2ezβ = sph_zero_spurious_modes!(2h.*expβ.(Z))
    hezγ = sph_zero_spurious_modes!(h.*expγ.(Z))
    ez2u = zero(Z)

    for k=1:n
        ez2u .= ez2.*U
        evaluate_nonlinear_operator!(SLPDE.N, P, PS, PA, NU .= U)
        A .= ez2u .+ h2ez2m1.*NU
        evaluate_nonlinear_operator!(SLPDE.N, P, PS, PA, NA .= A)
        B .= ez2u .+ h2ez2m1.*NA
        evaluate_nonlinear_operator!(SLPDE.N, P, PS, PA, NB .= B)
        C .= ez2.*A .+ h2ez2m1.*(2.0.*NB .- NU)
        evaluate_nonlinear_operator!(SLPDE.N, P, PS, PA, NC .= C)
        U .= ez.*U .+ hezα.*NU .+ h2ezβ.*(NA .+ NB) .+ hezγ.*NC
    end

    U
end
#=
function ETDRK4(SLPDE::SemiLinearPDE{2, S}, T::Number, n::Int) where S
    U0 = SLPDE.U0
    U = deepcopy(U0) # solution spherical harmonic coefficients
    V = zero(U[1]), zero(U[2]) # solution Fourier coefficients
    F = zero(U[1]), zero(U[2]) # solution function values
    NU = zero(U[1]), zero(U[2]) # nonlinearity spherical harmonic coefficients
    NV = zero(U[1]), zero(U[2]) # nonlinearity Fourier coefficients
    NF = zero(U[1]), zero(U[2]) # nonlinearity function values

    # ETDRK4 stages
    A = zero(U[1]), zero(U[2])
    B = zero(U[1]), zero(U[2])
    C = zero(U[1]), zero(U[2])
    NA = zero(U[1]), zero(U[2])
    NB = zero(U[1]), zero(U[2])
    NC = zero(U[1]), zero(U[2])

    P = plan_sph2fourier(U[1]; sketch = :none), plan_sph2fourier(U[2]; sketch = :none)
    Ps = FastTransforms.plan_synthesis(U[1]), FastTransforms.plan_synthesis(U[2])
    Pa = FastTransforms.plan_analysis(U[1]), FastTransforms.plan_analysis(U[2])

    L = create_linear_operator(SLPDE.L[1], size(U[1], 1), size(U[1], 2)÷2 + 1), create_linear_operator(SLPDE.L[2], size(U[2], 1), size(U[2], 2)÷2 + 1)
    h = S(T)/n

    # Create all the matrices of diagonal scalings used in ETDRK4. These operate
    # component-wise on the spherical harmonic coefficients.

    Z = L[1]*h, L[2]*h
    ez = exp.(Z[1]), exp.(Z[2])
    ez2 = exp.(Z[1]./2), exp.(Z[2]./2)
    h2ez2m1 = (h/2.0).*φ1.(Z[1]./2), (h/2.0).*φ1.(Z[2]./2)
    hezα = h.*expα.(Z[1]), h.*expα.(Z[2])
    h2ezβ = 2h.*expβ.(Z[1]), 2h.*expβ.(Z[2])
    hezγ = h.*expγ.(Z[1]), h.*expγ.(Z[2])
    ez2u = zero(Z[1]), zero(Z[2])

    for k=1:n
        ez2u[1] .= ez2[1].*U[1]
        ez2u[2] .= ez2[2].*U[2]
        evaluate_nonlinear_operator!(SLPDE.N, P, Ps, Pa, U, V, F, NF, NV, NU)
        A[1] .= ez2u[1] .+ h2ez2m1[1].*NU[1]
        A[2] .= ez2u[2] .+ h2ez2m1[2].*NU[2]
        evaluate_nonlinear_operator!(SLPDE.N, P, Ps, Pa, A, V, F, NF, NV, NA)
        B[1] .= ez2u[1] .+ h2ez2m1[1].*NA[1]
        B[2] .= ez2u[2] .+ h2ez2m1[2].*NA[2]
        evaluate_nonlinear_operator!(SLPDE.N, P, Ps, Pa, B, V, F, NF, NV, NB)
        C[1] .= ez2[1].*A[1] .+ h2ez2m1[1].*(2.0.*NB[1] .- NU[1])
        C[2] .= ez2[2].*A[2] .+ h2ez2m1[2].*(2.0.*NB[2] .- NU[2])
        evaluate_nonlinear_operator!(SLPDE.N, P, Ps, Pa, C, V, F, NF, NV, NC)
        U[1] .= ez[1].*U[1] .+ hezα[1].*NU[1] .+ h2ezβ[1].*(NA[1] .+ NB[1]) .+ hezγ[1].*NC[1]
        U[2] .= ez[2].*U[2] .+ hezα[2].*NU[2] .+ h2ezβ[2].*(NA[2] .+ NB[2]) .+ hezγ[2].*NC[2]
    end

    U
end
=#
function fillF!(Ft, F)
    m, n = size(F)
    FN = 0.0
    FS = 0.0
    @inbounds for j = 1:n
        FN += F[1, j]
        FS += F[m, j]
    end
    FN /= n
    FS /= n
    @inbounds for j = 1:n
        for i = 1:m
            Ft[i+1, j] = F[i, j]
        end
        Ft[1, j] = FN
        Ft[m+2, j] = FS
    end
    @inbounds for i = 1:m+2
        Ft[i, n+1] = Ft[i, 1]
    end
end


function ETDRK4(SLPDE::SemiLinearPDE{1, S}, T::Number, n::Int, surf, io) where S
    U0 = SLPDE.U0[1]
    U = deepcopy(U0) # solution spherical harmonic coefficients
    NU = zero(U) # nonlinearity spherical harmonic coefficients
    F = zero(U) # solution function values
    Ft = zeros(Float32, size(F, 1)+2, size(F, 2)+1)

    # ETDRK4 stages
    A = zero(U)
    B = zero(U)
    C = zero(U)
    NA = zero(U)
    NB = zero(U)
    NC = zero(U)

    P = plan_sph2fourier(U)
    PS = plan_sph_synthesis(U)
    PA = plan_sph_analysis(U)

    L = create_linear_operator(SLPDE.L, size(U, 1), size(U, 2)÷2 + 1)
    h = S(T)/n

    # Create all the matrices of diagonal scalings used in ETDRK4. These operate
    # component-wise on the spherical harmonic coefficients.

    Z = L*h
    ez = sph_zero_spurious_modes!(exp.(Z))
    ez2 = sph_zero_spurious_modes!(exp.(Z./2))
    h2ez2m1 = sph_zero_spurious_modes!((h/2.0).*φ1.(Z./2))
    hezα = sph_zero_spurious_modes!(h.*expα.(Z))
    h2ezβ = sph_zero_spurious_modes!(2h.*expβ.(Z))
    hezγ = sph_zero_spurious_modes!(h.*expγ.(Z))
    ez2u = zero(Z)

    for k=1:n
        ez2u .= ez2.*U
        evaluate_nonlinear_operator!(SLPDE.N, P, PS, PA, NU .= U)
        A .= ez2u .+ h2ez2m1.*NU
        evaluate_nonlinear_operator!(SLPDE.N, P, PS, PA, NA .= A)
        B .= ez2u .+ h2ez2m1.*NA
        evaluate_nonlinear_operator!(SLPDE.N, P, PS, PA, NB .= B)
        C .= ez2.*A .+ h2ez2m1.*(2.0.*NB .- NU)
        evaluate_nonlinear_operator!(SLPDE.N, P, PS, PA, NC .= C)
        U .= ez.*U .+ hezα.*NU .+ h2ezβ.*(NA .+ NB) .+ hezγ.*NC

        # plotting
        F .= U
        lmul!(P, F)
        lmul!(PS, F)
        fillF!(Ft, F)
        surf[:color] = Ft # animate scene
        recordframe!(io) # record a new frame
    end

    U
end

function sph_zero_spurious_modes!(A::AbstractMatrix)
    M, N = size(A)
    n = N÷2
    @inbounds for j = 1:n-1
        @simd for i = M-j+1:M
            A[i,2j] = 0
            A[i,2j+1] = 0
        end
    end
    @inbounds @simd for i = M-n+1:M
        A[i,2n] = 0
        2n < N && (A[i,2n+1] = 0)
    end
    A
end

#
# These formulæ, appearing in Eq. (2.5) of:
#
# A.-K. Kassam and L. N. Trefethen, Fourth-order time-stepping for stiff PDEs, SIAM J. Sci. Comput., 26:1214--1233, 2005,
#
# are derived to implement ETDRK4 in double precision without numerical instability from cancellation.
#

expα_asy(x) = (exp(x)*(4-3x+x^2)-4-x)/x^3
expβ_asy(x) = (exp(x)*(x-2)+x+2)/x^3
expγ_asy(x) = (exp(x)*(4-x)-4-3x-x^2)/x^3

# TODO: General types

expα_taylor(x::Union{Float64,ComplexF64}) = @evalpoly(x,1/6,1/6,3/40,1/45,5/1008,1/1120,7/51840,1/56700,1/492800,1/4790016,11/566092800,1/605404800,13/100590336000,1/106748928000,1/1580833013760,1/25009272288000,17/7155594141696000,1/7508956815360000)
expβ_taylor(x::Union{Float64,ComplexF64}) = @evalpoly(x,1/6,1/12,1/40,1/180,1/1008,1/6720,1/51840,1/453600,1/4435200,1/47900160,1/566092800,1/7264857600,1/100590336000,1/1494484992000,1/23712495206400,1/400148356608000,1/7155594141696000,1/135161222676480000)
expγ_taylor(x::Union{Float64,ComplexF64}) = @evalpoly(x,1/6,0/1,-1/120,-1/360,-1/1680,-1/10080,-1/72576,-1/604800,-1/5702400,-1/59875200,-1/691891200,-1/8717829120,-1/118879488000,-1/1743565824000,-1/27360571392000,-1/457312407552000,-1/8109673360588800)

expα(x::Float64) = abs(x) < 17/16 ? expα_taylor(x) : expα_asy(x)
expβ(x::Float64) = abs(x) < 19/16 ? expβ_taylor(x) : expβ_asy(x)
expγ(x::Float64) = abs(x) < 15/16 ? expγ_taylor(x) : expγ_asy(x)

expα(x::ComplexF64) = abs2(x) < (17/16)^2 ? expα_taylor(x) : expα_asy(x)
expβ(x::ComplexF64) = abs2(x) < (19/16)^2 ? expβ_taylor(x) : expβ_asy(x)
expγ(x::ComplexF64) = abs2(x) < (15/16)^2 ? expγ_taylor(x) : expγ_asy(x)

expα(x) = expα_asy(x)
expβ(x) = expβ_asy(x)
expγ(x) = expγ_asy(x)

φ1(x) = iszero(x) ? one(x) : expm1(x)/x
