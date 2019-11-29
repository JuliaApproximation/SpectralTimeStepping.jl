using FastTransforms, Makie, Random

include("SLPDE.jl")
include("etdrk4.jl")
include("evaluate_lambda.jl")
include("sphcesaro.jl")

Random.seed!(0)

n = 64
U0 = zeros(4n, 8n-1), zeros(4n, 8n-1)
U0[1][1:n, 1:2n-1] = sphrandn(Float64, n, 2n-1)/n
U0[2][1:n, 1:2n-1] = sphrandn(Float64, n, 2n-1)/3n
U0[2][1] += 0.5sqrt(4π)
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
println("The extrema of v: ", extrema(G))

scene = Scene(resolution = (1200, 1200));
surf = surface!(scene, x, y, z, color = Ft, colormap = :viridis, colorrange = extrema(Ft))[end];
update_cam!(scene, Vec3f0(2.5), Vec3f0(0), Vec3f0(0, 0, 1))
scene.center = false
scene
Makie.save("plots/LGMu0.jpeg", scene)
Makie.save("plots/NGMu0.jpeg", scene)

T = 4.0
n = round(Int, 10*T)

SLPDE = SemiLinearPDE((LaplaceBeltrami(0.01),LaplaceBeltrami(0.1)), (NonlinearOperator((u,v)->(u-v)*u/v),NonlinearOperator((u,v)->u^2-v)), U0)
io = VideoStream(scene);
ETDRK4(SLPDE, T, n, surf, io; colorrange = :notfixed)
Makie.save("plots/LGM.gif", io)

SLPDE = SemiLinearPDE((NonlocalLaplaceBeltrami(0.01, 0.0, 1.0),NonlocalLaplaceBeltrami(0.1, -0.5, 0.05)), (NonlinearOperator((u,v)->(u-v)*u/v),NonlinearOperator((u,v)->u^2-v)), U0)
io = VideoStream(scene);
ETDRK4(SLPDE, T, n, surf, io; colorrange = :notfixed)
Makie.save("plots/NGM.gif", io)
