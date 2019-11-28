include("evaluate_lambda.jl")

using PyPlot

α = [-0.5;0.5]
L = round.(Int, exp10.(range(0, stop=4, length=50)))
λlap = [-ℓ*(ℓ+1) for ℓ in L]

for i = 1:length(α)
    δ1 = 0.01
    λ1 = evaluate_lambda(L[end]+1, α[i], δ1)[L.+1]
    δ2 = 0.1
    λ2 = evaluate_lambda(L[end]+1, α[i], δ2)[L.+1]
    δ3 = 1.0
    λ3 = evaluate_lambda(L[end]+1, α[i], δ3)[L.+1]
    δ4 = 2.0
    λ4 = evaluate_lambda(L[end]+1, α[i], δ4)[L.+1]

    clf()
    loglog(L, -λlap, "-k", L, -λ1, ".k", L, -λ2, "xk", L, -λ3, "+k", L, -λ4, "--k")
    xlabel("\$\\ell\$"); ylabel("Numerical Evaluation of \$\\lambda_\\delta(\\ell)\$ at \$\\alpha=$(α[i])\$"); grid()
    legend(["\$-\\lambda_0(\\ell)\$","\$-\\lambda_{$(δ1)}(\\ell)\$","\$-\\lambda_{$(δ2)}(\\ell)\$","\$-\\lambda_{$(δ3 == Int(δ3) ? Int(δ3) : δ3)}(\\ell)\$","\$-\\lambda_{$(δ4 == Int(δ4) ? Int(δ4) : δ4)}(\\ell)\$"])
    savefig("plot_spectrum$(i).pdf")
end
