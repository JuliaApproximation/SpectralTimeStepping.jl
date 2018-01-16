include("evaluate_lambda.jl")

using PyPlot

α = [-0.5;0.5]
L = round.([Int], logspace(0,4,50))
λlap = [-ℓ*(ℓ+1) for ℓ in L]

for i = 1:length(α)
    δ1 = 0.01
    λ1 = [evaluate_lambda(ℓ, δ1, α[i]) for ℓ in L]
    δ2 = 0.1
    λ2 = [evaluate_lambda(ℓ, δ2, α[i]) for ℓ in L]
    δ3 = 1.0
    λ3 = [evaluate_lambda(ℓ, δ3, α[i]) for ℓ in L]
    δ4 = 2.0
    λ4 = [evaluate_lambda(ℓ, δ4, α[i]) for ℓ in L]

    clf()
    loglog(L, -λlap, "-k", L, -λ1, ".k", L, -λ2, "xk", L, -λ3, "+k", L, -λ4, "--k")
    xlabel("\$\\ell\$"); ylabel("Numerical Evaluation of \$\\lambda_\\delta(\\ell)\$ at \$\\alpha=$(α[i])\$"); grid()
    legend(["\$-\\lambda_0(\\ell)\$","\$-\\lambda_{$(δ1)}(\\ell)\$","\$-\\lambda_{$(δ2)}(\\ell)\$","\$-\\lambda_{$(δ3 == Int(δ3) ? Int(δ3) : δ3)}(\\ell)\$","\$-\\lambda_{$(δ4 == Int(δ4) ? Int(δ4) : δ4)}(\\ell)\$"])
    savefig("plot_spectrum$(i).pdf")
end
