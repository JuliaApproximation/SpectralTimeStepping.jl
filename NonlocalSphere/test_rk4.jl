include("SLPDE.jl")
include("rk4.jl")
include("evaluate_lambda.jl")

srand(0)

println("Testing Local Allen--Cahn")

n = 25
U0 = zeros(4n, 8n-1);
U0[1:n, 1:2n-1] = sphrandn(Float64, n, n)/n

SLPDE = SemiLinearPDE(LaplaceBeltrami(0.1), NonlinearOperator(u->u-u^3), U0)

IMAX = 6

U = Vector{Matrix{Float64}}(IMAX)
for i = 1:IMAX
    U[i] = RK4(SLPDE, 1.0, 10*2^(i-1))
    println("Done i = ", i)
end

err = zeros(IMAX-1)
for i = 1:IMAX-1
    err[i] = vecnorm(U[i] - U[end], Inf)
    println("i = ",i,", and the ∞-norm absolute error: ", err[i])
end

for i = 1:IMAX-2
    println(err[i]/err[i+1])
end

println("Testing Nonlocal Allen--Cahn")

SLPDE = SemiLinearPDE(NonlocalLaplaceBeltrami(0.1, -0.5, 0.2), NonlinearOperator(u->u-u^3), U0)

U = Vector{Matrix{Float64}}(IMAX)
for i = 1:IMAX
    U[i] = RK4(SLPDE, 1.0, 10*2^(i-1))
    println("Done i = ", i)
end

err = zeros(IMAX-1)
for i = 1:IMAX-1
    err[i] = vecnorm(U[i] - U[end], Inf)
    println("i = ",i,", and the ∞-norm absolute error: ", err[i])
end

for i = 1:IMAX-2
    println(err[i]/err[i+1])
end

println("Testing Local Gray--Scott")

n = 25
U0 = zeros(4n, 8n-1), zeros(4n, 8n-1);
U0[1][1:n, 1:2n-1] = sphrandn(Float64, n, n)/n
U0[2][1:n, 1:2n-1] = sphrandn(Float64, n, n)/n

SLPDE = SemiLinearPDE((LaplaceBeltrami(sqrt(0.00002)), LaplaceBeltrami(sqrt(0.00001))), (NonlinearOperator((u,v)->0.04*(1-u)-u*v^2), NonlinearOperator((u,v)->-0.1*v+u*v^2)), U0)

U = Vector{Tuple{Matrix{Float64},Matrix{Float64}}}(IMAX)
for i = 1:IMAX
    U[i] = RK4(SLPDE, 1.0, 10*2^(i-1))
    println("Done i = ", i)
end

err = zeros(IMAX-1)
for i = 1:IMAX-1
    err[i] = vecnorm(U[i][1] - U[end][1], Inf) + vecnorm(U[i][2] - U[end][2], Inf)
    println("i = ",i,", and the ∞-norm absolute error: ", err[i])
end

for i = 1:IMAX-2
    println(err[i]/err[i+1])
end

println("Testing Nonlocal Gray--Scott")

SLPDE = SemiLinearPDE((NonlocalLaplaceBeltrami(sqrt(0.00002), -0.5, 0.2), NonlocalLaplaceBeltrami(sqrt(0.00001), -0.5, 0.2)), (NonlinearOperator((u,v)->0.04*(1-u)-u*v^2), NonlinearOperator((u,v)->-0.1*v+u*v^2)), U0)

U = Vector{Tuple{Matrix{Float64},Matrix{Float64}}}(IMAX)
for i = 1:IMAX
    U[i] = RK4(SLPDE, 1.0, 10*2^(i-1))
    println("Done i = ", i)
end

err = zeros(IMAX-1)
for i = 1:IMAX-1
    err[i] = vecnorm(U[i][1] - U[end][1], Inf) + vecnorm(U[i][2] - U[end][2], Inf)
    println("i = ",i,", and the ∞-norm absolute error: ", err[i])
end

for i = 1:IMAX-2
    println(err[i]/err[i+1])
end
