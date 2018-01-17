include("SLPDE.jl")
include("etdrk4.jl")
include("evaluate_lambda.jl")

# This works assuming the reference solution is on a larger discretization.
function relerr(UN::Matrix, UR::Matrix)
    m, n = size(UN)
    M, N = size(UR)
    ERR = deepcopy(UR)
    @inbounds for j = 1:n, i = 1:m
        ERR[i,j] -= UN[i,j]
    end
    return vecnorm(ERR)/vecnorm(UR)
end

using PyPlot

IMAX = 8
Ntimes = 2

n = 1024

θ = (0.5:n-0.5)/n
φ = (0:2n-2)*2/(2n-1)
F = [cos(10*cospi(φ)*sinpi(θ)*sinpi(φ)*sinpi(θ)) for θ in θ, φ in φ]
V = zero(F)
A_mul_B!(V, FastTransforms.plan_analysis(F), F)
U0 = fourier2sph(V; sketch = :none)

SLPDE = SemiLinearPDE(NonlocalLaplaceBeltrami(0.01, -0.5, 1.0), NonlinearOperator(u->u-u^3), U0)
UN = Matrix{Matrix{Float64}}(IMAX, Ntimes)
for j = 1:Ntimes
    T = 1.0*4^(j-1)
    for i = 1:IMAX
        UN[i,j] = ETDRK4(SLPDE, T, round(Int, 2.0^i*T))
        println("Done i = ", i," j = ", j)
    end
end

n = 2048

θ = (0.5:n-0.5)/n
φ = (0:2n-2)*2/(2n-1)
F = [cos(10*cospi(φ)*sinpi(θ)*sinpi(φ)*sinpi(θ)) for θ in θ, φ in φ]
V = zero(F)
A_mul_B!(V, FastTransforms.plan_analysis(F), F)
U0 = fourier2sph(V; sketch = :none)

SLPDE = SemiLinearPDE(NonlocalLaplaceBeltrami(0.01, -0.5, 1.0), NonlinearOperator(u->u-u^3), U0)
UREF = Matrix{Matrix{Float64}}(1, Ntimes)
for j = 1:Ntimes
    T = 1.0*4^(j-1)
    UREF[1,j] = ETDRK4(SLPDE, T, round(Int, 2.0^(IMAX+1)*T))
    println("Done j = ", j)
end


errt = zeros(IMAX, Ntimes)
for j = 1:Ntimes
    for i = 1:IMAX
        errt[i,j] = relerr(UN[i,j], UREF[1,j])
        println("i = ",i,", j = ", j," and the vector 2-norm relative error: ", errt[i,j])
    end
end

h = 1./(2.0.^(1:IMAX))

clf()
loglog(h, errt[:,1], ".k", h, errt[:,2], "xk")
loglog(h, h.^4./60, "-k")
xlabel("\$h\$"); ylabel("Relative Error"); grid()
ylim((1e-13,1e-0))
legend(["\$t=1\$","\$t=4\$","\$\\mathcal{O}(h^4)\$"], loc = "lower right")
savefig("NAC_temporal_error.pdf")


IMAX = 8
Ntimes = 2

UN = Matrix{Matrix{Float64}}(IMAX, Ntimes)
for j = 1:Ntimes
    T = 1.0*4^(j-1)
    for i = 1:IMAX
        n = 64*i
        θ = (0.5:n-0.5)/n
        φ = (0:2n-2)*2/(2n-1)
        F = [cos(10*cospi(φ)*sinpi(θ)*sinpi(φ)*sinpi(θ)) for θ in θ, φ in φ]
        V = zero(F)
        A_mul_B!(V, FastTransforms.plan_analysis(F), F)
        U0 = fourier2sph(V; sketch = :none)

        SLPDE = SemiLinearPDE(NonlocalLaplaceBeltrami(0.01, -0.5, 1.0), NonlinearOperator(u->u-u^3), U0)

        UN[i,j] = ETDRK4(SLPDE, T, round(Int, 2.0^8*T))
        println("Done i = ", i," j = ", j)
    end
end

n = 64*(2*IMAX)

θ = (0.5:n-0.5)/n
φ = (0:2n-2)*2/(2n-1)
F = [cos(10*cospi(φ)*sinpi(θ)*sinpi(φ)*sinpi(θ)) for θ in θ, φ in φ]
V = zero(F)
A_mul_B!(V, FastTransforms.plan_analysis(F), F)
U0 = fourier2sph(V; sketch = :none)

SLPDE = SemiLinearPDE(NonlocalLaplaceBeltrami(0.01, -0.5, 1.0), NonlinearOperator(u->u-u^3), U0)
UREF = Matrix{Matrix{Float64}}(1, Ntimes)
for j = 1:Ntimes
    T = 1.0*4^(j-1)
    UREF[1,j] = ETDRK4(SLPDE, T, round(Int, 2.0^9*T))
    println("Done j = ", j)
end

errs = zeros(IMAX, Ntimes)
for j = 1:Ntimes
    for i = 1:IMAX
        errs[i,j] = relerr(UN[i,j], UREF[1,j])
        println("i = ",i,", j = ", j," and the vector 2-norm relative error: ", errs[i,j])
    end
end

n = 64*(1:IMAX)-1

clf()
semilogy(n, errs[:,1], "--.k", n, errs[:,2], "--xk")
xlabel("\$n\$"); ylabel("Relative Error"); grid()
ylim((1e-13,1e-0))
legend(["\$t=1\$","\$t=4\$"], loc = "lower left")
savefig("NAC_spatial_error.pdf")
