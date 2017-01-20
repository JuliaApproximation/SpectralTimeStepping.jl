using DiffEqDocs, DiffEqBase, OrdinaryDiffEq, Sundials, ApproxFun
using Base.Test

S=Fourier()
u0=Fun(θ->cos(cos(θ-0.1)),S)
c=Fun(cos,S)

prob = SpectralTimeSteppingProblem((t,u)->u''+(c+1)*u',u0,(0.,1.))
@time u=solve(prob,Tsit5()); #   4.087185 seconds (5.85 M allocations: 226.447 MB, 2.34% gc time)
