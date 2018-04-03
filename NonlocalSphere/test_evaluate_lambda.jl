include("evaluate_lambda.jl")

using ApproxFun, PyPlot

x = Fun(Segment{BigFloat}())

αBF = big(-0.5)
βBF = big(0.0)

δ = 1.0
α = Float64(αBF)
ℓmax = 1000

λ = zeros(Float64, ℓmax)
λfast = zeros(Float64, ℓmax)
λBF = zeros(BigFloat, ℓmax)

for ℓ in 1:ℓmax
    λ[ℓ] = evaluate_lambda_rec(ℓ, δ, α)
    λfast[ℓ] = evaluate_lambda_asy(ℓ, δ, α)
    Pℓ = legendre(BigFloat, ℓ)
    f = Pℓ(1-big(δ)^2/2*(1-x)/2)
    g = (f-1)/(1-x)
    #w = Fun(JacobiWeight(big(0.0), big(α), space(x)),[big(1.0)])
    #λBF[ℓ] = sum(g*w)
    n = ncoefficients(g)
    c = zeros(BigFloat, n)
    αG = αBF
    if space(g) == space(x)
        αG = αBF
    else
        αG = αBF-1
    end
    c[1] = big(2.0)^(βBF+αG+1)*gamma(βBF+1)*gamma(αG+1)/gamma(βBF+αG+2)
    if n > 1
        c[2] = c[1]*(βBF-αG)/(βBF+αG+2)
        for i=1:n-2
            c[i+2] = (2(βBF-αG)*c[i+1]-(βBF+αG-i+2)*c[i])/(βBF+αG+i+2)
        end
    end
    λBF[ℓ] = ApproxFun.dotu(g.coefficients, c)
    println("Finished ℓ: ",ℓ)
end

λBF64 = Float64.(λBF)

clf()
semilogy(1:ℓmax, abs.((λ-λBF64)./λBF64), ".k", 1:ℓmax, abs.((λfast-λBF64)./λBF64), "xk")
xlabel("\$\\ell\$"); ylabel("Relative Error"); grid()
legend(["REC","ASY"])
savefig("spectrum_error.pdf")

L = [collect(1:10); round.([Int], logspace(0,4,80))[22:end]]
tREC = zeros(Float64, length(L))
tASY = zeros(Float64, length(L))

j = 1
for ℓ in L
    tREC[j] = @elapsed evaluate_lambda_rec(ℓ, δ, α)
    tASY[j] = @elapsed evaluate_lambda_asy(ℓ, δ, α)
    println("Finished ℓ: ",ℓ)
    gc()
    sleep(0.1)
    j += 1
end

clf()
loglog(L, tREC, ".k", L, tASY, "xk")
loglog(L, 6.25e-8*L.*log.(L), "-k")
loglog(L, 1e-8*L.^2, "--k")
ylim(1e-5,1e0)
xlabel("\$\\ell\$"); ylabel("Execution Time (s)"); grid()
legend(["REC","ASY","\$\\mathcal{O}(\\ell\\log\\ell)\$","\$\\mathcal{O}(\\ell^2)\$"])
savefig("spectrum_time.pdf")
