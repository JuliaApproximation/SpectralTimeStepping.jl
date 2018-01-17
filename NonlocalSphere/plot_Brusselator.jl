include("SLPDE.jl")
include("etdrk4.jl")
include("evaluate_lambda.jl")
include("sphcesaro.jl")

using PlotlyJS

srand(0)

f = 0.8
ε = 0.075
τ = 7.8125
E = 4.0

ue = ε^2*E/(1-f)*sqrt(4π)
ve = (1-f)/(E*ε^2)*sqrt(4π)

n = 100
U0 = zeros(4n, 8n-1), zeros(4n, 8n-1)
U0[1][1:n, 1:2n-1] = sphrandn(Float64, n, n)/(π*n)*ue/100
U0[2][1:n, 1:2n-1] = sphrandn(Float64, n, n)/(π*n)*ve/100
U0[1][1] += ue
U0[2][1] += ve

n = size(U0[1], 1)

θ = [0;(0.5:n-0.5)/n;1]
φ = [(0:2n-2)*2/(2n-1);2]
x = [cospi(φ)*sinpi(θ) for θ in θ, φ in φ]
y = [sinpi(φ)*sinpi(θ) for θ in θ, φ in φ]
z = [cospi(θ) for θ in θ, φ in φ]

V = sph2fourier(U0[1]), sph2fourier(U0[2])
F = zero(V[1])
G = zero(V[2])
A_mul_B!(F, FastTransforms.plan_synthesis(V[1]), V[1])
A_mul_B!(G, FastTransforms.plan_synthesis(V[2]), V[2])
F = [mean(F[1,:])*ones(1, size(F, 2)); F; mean(F[end,:])*ones(1, size(F, 2))]
F = [F F[:,1]]
G = [mean(G[1,:])*ones(1, size(G, 2)); G; mean(G[end,:])*ones(1, size(G, 2))]
G = [G G[:,1]]

println("These are the extrema of the first component: ", extrema(F))
println(minimum(F)/(ue/sqrt(4π)),"  ",maximum(F)/(ue/sqrt(4π)))
s = surface(x=x, y=y, z=z, colorscale = "Viridis", surfacecolor = F, cmin = minimum(F), cmax = maximum(F), showscale = false)
ax = attr(visible = false)
cam = attr(up = attr(x=0,y=0,z=1), center = attr(x=0,y=0,z=0), eye = attr(x=0.75,y=0.75,z=0.75))
layout = Layout(width = 500, height = 500, autosize = false, margin = attr(l = 0, r = 0, b = 0, t = 0), scene = attr(xaxis = ax, yaxis = ax, zaxis = ax, camera = cam))
p = plot(s, layout)
savefig(p,"/Users/Mikael/LBRu0.pdf";js=:remote)
savefig(p,"/Users/Mikael/NBRu0.pdf";js=:remote)

println("These are the extrema of the second component: ", extrema(G))
println(minimum(G)/(ve/sqrt(4π)),"  ",maximum(G)/(ve/sqrt(4π)))
s = surface(x=x, y=y, z=z, colorscale = "Viridis", surfacecolor = G, cmin = minimum(G), cmax = maximum(G), showscale = false)
ax = attr(visible = false)
cam = attr(up = attr(x=0,y=0,z=1), center = attr(x=0,y=0,z=0), eye = attr(x=0.75,y=0.75,z=0.75))
layout = Layout(width = 500, height = 500, autosize = false, margin = attr(l = 0, r = 0, b = 0, t = 0), scene = attr(xaxis = ax, yaxis = ax, zaxis = ax, camera = cam))
p = plot(s, layout)
savefig(p,"/Users/Mikael/LBRv0.pdf";js=:remote)
savefig(p,"/Users/Mikael/NBRv0.pdf";js=:remote)


IMAX = 4

SLPDE = SemiLinearPDE((LaplaceBeltrami(ε),LaplaceBeltrami(inv(sqrt(τ)))), (NonlinearOperator((u,v)->ε^2*E-u+f*u^2*v),NonlinearOperator((u,v)->(u-u^2*v)/(τ*ε^2))), U0)
UL = Vector{NTuple{2,Matrix{Float64}}}(IMAX)
for i = 1:IMAX
    T = 1.0*4^(i-1)
    n = round(Int, 10*T)
    UL[i] = ETDRK4(SLPDE, T, n)
    println("Done i = ", i)

    V = sph2fourier(UL[i][1]; sketch = :none)
    F = zero(V)
    A_mul_B!(F, FastTransforms.plan_synthesis(V), V)
    F = [mean(F[1,:])*ones(1, size(F, 2)); F; mean(F[end,:])*ones(1, size(F, 2))]
    F = [F F[:,1]]

    println("These are the extrema of the first component: ", extrema(F))
    s = surface(x=x, y=y, z=z, colorscale = "Viridis", surfacecolor = F, cmin = minimum(F), cmax = maximum(F), showscale = false)
    p = plot(s, layout)
    savefig(p,"/Users/Mikael/LBRu$(i).pdf";js=:remote)

    V = sph2fourier(UL[i][2]; sketch = :none)
    F = zero(V)
    A_mul_B!(F, FastTransforms.plan_synthesis(V), V)
    F = [mean(F[1,:])*ones(1, size(F, 2)); F; mean(F[end,:])*ones(1, size(F, 2))]
    F = [F F[:,1]]

    println("These are the extrema of the second component: ", extrema(F))
    s = surface(x=x, y=y, z=z, colorscale = "Viridis", surfacecolor = F, cmin = minimum(F), cmax = maximum(F), showscale = false)
    p = plot(s, layout)
    savefig(p,"/Users/Mikael/LBRv$(i).pdf";js=:remote)
end

SLPDE = SemiLinearPDE((NonlocalLaplaceBeltrami(ε, 0.0, 1.0),NonlocalLaplaceBeltrami(inv(sqrt(τ)), 0.0, 1.0)), (NonlinearOperator((u,v)->ε^2*E-u+f*u^2*v),NonlinearOperator((u,v)->(u-u^2*v)/(τ*ε^2))), U0)
UN = Vector{NTuple{2,Matrix{Float64}}}(IMAX)
for i = 1:IMAX
    T = 1.0*4^(i-1)
    n = round(Int, 10*T)
    UN[i] = ETDRK4(SLPDE, T, n)
    println("Done i = ", i)

    V = sph2fourier(sphcesaro2(UN[i][1]).*UN[i][1]; sketch = :none)
    F = zero(V)
    A_mul_B!(F, FastTransforms.plan_synthesis(V), V)
    F = [mean(F[1,:])*ones(1, size(F, 2)); F; mean(F[end,:])*ones(1, size(F, 2))]
    F = [F F[:,1]]

    println("These are the extrema of the first component: ", extrema(F))
    s = surface(x=x, y=y, z=z, colorscale = "Viridis", surfacecolor = F, cmin = minimum(F), cmax = maximum(F), showscale = false)
    p = plot(s, layout)
    savefig(p,"/Users/Mikael/NBRu$(i).pdf";js=:remote)

    V = sph2fourier(sphcesaro2(UN[i][2]).*UN[i][2]; sketch = :none)
    F = zero(V)
    A_mul_B!(F, FastTransforms.plan_synthesis(V), V)
    F = [mean(F[1,:])*ones(1, size(F, 2)); F; mean(F[end,:])*ones(1, size(F, 2))]
    F = [F F[:,1]]

    println("These are the extrema of the second component: ", extrema(F))
    s = surface(x=x, y=y, z=z, colorscale = "Viridis", surfacecolor = F, cmin = minimum(F), cmax = maximum(F), showscale = false)
    p = plot(s, layout)
    savefig(p,"/Users/Mikael/NBRv$(i).pdf";js=:remote)
end
