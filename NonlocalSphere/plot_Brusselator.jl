using FastTransforms, LinearAlgebra, Makie, Random

include("SLPDE.jl")
include("etdrk4.jl")
include("evaluate_lambda.jl")
include("sphcesaro.jl")

Random.seed!(0)

f = 0.8
ε = 0.075
τ = 7.8125
E = 4.0

ue = ε^2*E/(1-f)*sqrt(4π)
ve = (1-f)/(E*ε^2)*sqrt(4π)

n = 64
U0 = zeros(4n, 8n-1), zeros(4n, 8n-1)
U0[1][1:n, 1:2n-1] = sphrandn(Float64, n, 2n-1)/(π*n)*ue/100
U0[2][1:n, 1:2n-1] = sphrandn(Float64, n, 2n-1)/(π*n)*ve/100
U0[1][1] += ue
U0[2][1] += ve

n = size(U0[1], 1)

θ = [0;(0.5:n-0.5)/n;1]
φ = [(0:2n-2)*2/(2n-1);2]
x = Float32[cospi(φ)*sinpi(θ) for θ in θ, φ in φ]
y = Float32[sinpi(φ)*sinpi(θ) for θ in θ, φ in φ]
z = Float32[cospi(θ) for θ in θ, φ in φ]

P = plan_sph2fourier(U0[1])
PS = plan_sph_synthesis(U0[1])

F = deepcopy(U0[1])
Ft = zeros(Float32, size(F, 1)+2, size(F, 2)+1)
lmul!(P, F)
lmul!(PS, F)
fillF!(Ft, F)
G = deepcopy(U0[2])
Gt = zeros(Float32, size(G, 1)+2, size(G, 2)+1)
lmul!(P, G)
lmul!(PS, G)
fillF!(Gt, G)

println("These are the extrema of the first component: ", extrema(F))
println(minimum(F)/(ue/sqrt(4π)),"  ",maximum(F)/(ue/sqrt(4π)))
scene = Scene(resolution = (1200, 1200));
surf = surface!(scene, x, y, z, color = Ft, colormap = :viridis, colorrange = extrema(F));
update_cam!(scene, Vec3f0(2.5), Vec3f0(0), Vec3f0(0, 0, 1))
scene.center = false
scene
Makie.save("plots/LBRu0.jpeg", scene)
Makie.save("plots/NBRu0.jpeg", scene)

println("These are the extrema of the second component: ", extrema(G))
println(minimum(G)/(ve/sqrt(4π)),"  ",maximum(G)/(ve/sqrt(4π)))
scene = Scene(resolution = (1200, 1200));
surf = surface!(scene, x, y, z, color = Gt, colormap = :viridis, colorrange = extrema(G));
update_cam!(scene, Vec3f0(2.5), Vec3f0(0), Vec3f0(0, 0, 1))
scene.center = false
scene
Makie.save("plots/LBRv0.jpeg", scene)
Makie.save("plots/NBRv0.jpeg", scene)

T = 16.0
n = round(Int, 10*T)

SLPDE = SemiLinearPDE((LaplaceBeltrami(ε),LaplaceBeltrami(inv(sqrt(τ)))), (NonlinearOperator((u,v)->ε^2*E-u+f*u^2*v),NonlinearOperator((u,v)->(u-u^2*v)/(τ*ε^2))), U0)
io = VideoStream(scene);
ETDRK4(SLPDE, T, n, surf, io; colorrange=:notfixed)
Makie.save("plots/LBR.gif", io)

SLPDE = SemiLinearPDE((NonlocalLaplaceBeltrami(ε, 0.0, 1.0),NonlocalLaplaceBeltrami(inv(sqrt(τ)), 0.0, 1.0)), (NonlinearOperator((u,v)->ε^2*E-u+f*u^2*v),NonlinearOperator((u,v)->(u-u^2*v)/(τ*ε^2))), U0)
io = VideoStream(scene);
ETDRK4(SLPDE, T, n, surf, io; colorrange=:notfixed)
Makie.save("plots/NBR.gif", io)
