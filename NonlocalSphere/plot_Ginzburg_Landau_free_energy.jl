include("SLPDE.jl")
include("etdrk4.jl")
include("evaluate_lambda.jl")
include("sphcesaro.jl")

using PyPlot

srand(0)

n = 128
U0 = zeros(4n, 8n-1);
U0[1:n, 1:2n-1] = sphrandn(Float64, n, n)/n
n = size(U0, 1)

θ = [0;(0.5:n-0.5)/n;1]
φ = [(0:2n-2)*2/(2n-1);2]
x = [cospi(φ)*sinpi(θ) for θ in θ, φ in φ]
y = [sinpi(φ)*sinpi(θ) for θ in θ, φ in φ]
z = [cospi(θ) for θ in θ, φ in φ]

V = sph2fourier(U0)
F = zero(V)
A_mul_B!(F, FastTransforms.plan_synthesis(V), V)
F = [mean(F[1,:])*ones(1, size(F, 2)); F; mean(F[end,:])*ones(1, size(F, 2))]
F = [F F[:,1]]



function ETDRK4_free_energy(SLPDE::SemiLinearPDE{1, S}, T::Number, n::Int) where S
    U0 = SLPDE.U0[1]
    U = deepcopy(U0) # solution spherical harmonic coefficients
    V = zero(U) # solution Fourier coefficients
    F = zero(U) # solution function values
    NU = zero(U) # nonlinearity spherical harmonic coefficients
    NV = zero(U) # nonlinearity Fourier coefficients
    NF = zero(U) # nonlinearity function values

    # ETDRK4 stages
    A = zero(U)
    B = zero(U)
    C = zero(U)
    NA = zero(U)
    NB = zero(U)
    NC = zero(U)

    P = plan_sph2fourier(U; sketch = :none)
    Ps = FastTransforms.plan_synthesis(U)
    Pa = FastTransforms.plan_analysis(U)

    L = create_linear_operator(SLPDE.L, size(U, 1), size(U, 2)÷2 + 1)
    h = S(T)/n

    NE = NonlinearOperator(u -> 0.5*(u^2-1))

    ENERGY = zeros(S, n+1)
    ENERGY[1] = -0.5*sum(U.*L.*U) + sum(evaluate_nonlinear_operator!(NE, P, Ps, Pa, U, V, F, NF, NV, NU).^2)

    # Create all the matrices of diagonal scalings used in ETDRK4. These operate
    # component-wise on the spherical harmonic coefficients.

    Z = L*h
    ez = exp.(Z)
    ez2 = exp.(Z./2)
    h2ez2m1 = (h/2.0).*φ1.(Z./2)
    hezα = h.*expα.(Z)
    h2ezβ = 2h.*expβ.(Z)
    hezγ = h.*expγ.(Z)
    ez2u = zero(Z)

    for k=1:n
        ez2u .= ez2.*U
        evaluate_nonlinear_operator!(SLPDE.N, P, Ps, Pa, U, V, F, NF, NV, NU)
        A .= ez2u .+ h2ez2m1.*NU
        evaluate_nonlinear_operator!(SLPDE.N, P, Ps, Pa, A, V, F, NF, NV, NA)
        B .= ez2u .+ h2ez2m1.*NA
        evaluate_nonlinear_operator!(SLPDE.N, P, Ps, Pa, B, V, F, NF, NV, NB)
        C .= ez2.*A .+ h2ez2m1.*(2.0.*NB .- NU)
        evaluate_nonlinear_operator!(SLPDE.N, P, Ps, Pa, C, V, F, NF, NV, NC)
        U .= ez.*U .+ hezα.*NU .+ h2ezβ.*(NA .+ NB) .+ hezγ.*NC
        ENERGY[k+1] = -0.5*sum(U.*L.*U) + sum(evaluate_nonlinear_operator!(NE, P, Ps, Pa, U, V, F, NF, NV, NU).^2)
    end

    ENERGY
end


T = 64.0
n = round(Int, 10*T)
t = linspace(0, T, n+1)

SLPDE = SemiLinearPDE(LaplaceBeltrami(0.01), NonlinearOperator(u->u-u^3), U0)
local_energy = ETDRK4_free_energy(SLPDE, T, n)

SLPDE = SemiLinearPDE(NonlocalLaplaceBeltrami(0.01, -0.5, 1.0), NonlinearOperator(u->u-u^3), U0)
nonlocal_energy = ETDRK4_free_energy(SLPDE, T, n)

clf()
plot(t, local_energy, "-k", t, nonlocal_energy, "--k")
xlabel("\$t\$"); ylabel("\${\\cal E}(u)\$"); grid()
legend(["Local","Nonlocal"])
savefig("GL_free_energy.pdf")
