module SpectralTimeStepping
    using ApproxFun, DifferentialEquations

    export SpectralTimeSteppingProblem

    immutable SpectralTimeSteppingProblem{S,PROB}
        problem::PROB
        space::S
    end

    function SpectralTimeSteppingProblem(f,u0,tspan)
        const sp = space(u0)
        const n = ncoefficients(u0)
        prob = ODEProblem((t,u)->pad!(f(t,Fun(sp,u)).coefficients,n),u0.coefficients,tspan)
        SpectralTimeSteppingProblem(prob,sp)
    end

    immutable SpectralTimeSteppingSolution{S,SOL}
        solution::SOL
        space::S
    end

    (sol::SpectralTimeSteppingSolution)(t) = Fun(sol.space,sol.solution(t))
    (sol::SpectralTimeSteppingSolution)(t,x) = sol(t)(x)

    DifferentialEquations.solve(prob::SpectralTimeSteppingProblem,alg,args...;kwargs...) =
        SpectralTimeSteppingSolution(solve(prob.problem,alg,args...;kwargs...),prob.space)
end    # module
