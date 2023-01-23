# Problem Set 3, Life Cycle Course, Q1
# By Marius Gruenewald

#----------------#
# Importing Pckg #
#----------------#
#import Pkg; Pkg.add("CSV")
#import Pkg; Pkg.add("DataFrames")
#import Pkg; Pkg.add("Interpolations")
#import Pkg; Pkg.add("LaTeXStrings")
#import Pkg; Pkg.add("Parameters")
#import Pkg; Pkg.add("Plots")
#import Pkg; Pkg.add("QuantEcon")
#import Pkg; Pkg.add("BenchmarkTools")
#import Pkg; Pkg.add("StatsBase")
#import Pkg; Pkg.add("HiGHS")


# using Pckg
using BenchmarkTools
using FastGaussQuadrature
using ChebyshevApprox
using CSV
using DataFrames
using DelimitedFiles
using Plots
using Interpolations
using Random
using Parameters
using LaTeXStrings
using Statistics
using StatsBase
using NLsolve
using JuMP
using Ipopt


Random.seed!(1234)
@with_kw struct Parameter_q1
    
    minval::Float64 = 1.0
    maxval::Float64 = 20.0
    p::Int64 = 3
    na::Int64 = 2
    σ_e::Float64 = 0.2
    σ::Float64 = 2.0
    β::Float64 = 0.95
    sⱼ::Float64 = 0.9
    r::Float64 = 0.05

    n::Int64 = 20
    grid::Vector{Float64} = collect(LinRange(minval, maxval, p))
    fine_grid::Vector{Float64} = collect(LinRange(minval, maxval, 1000))
    tol::Float64 = 1e-5
    maxcount::Int64 = 100

end
# Collocation with Monomials
param=Parameter_q1()

y, w = gausshermite(param.n)

last_part = exp.(sqrt(2 * param.σ^2) .* y)

guess = zeros(param.p)          # guess for the solution
function Collocation(GH_nodes, GH_weights, param::Parameter_q1, guess::Vector{Float64})
    @unpack β, sⱼ, σ, r, grid, p = param
    function f!(F, x)

        rhs = β*sⱼ*(1+r)
        denom = zeros(p)
        for i in 1:p
            denom[i]= 1/sqrt(π)*(GH_weights'*((1+r)*(x[1]+x[2]*grid[i]+x[3]*grid[i]^2).+ GH_nodes).^(-σ))
            F[i]    = ((grid[i]-x[1]-x[2]*grid[i]-x[3]*grid[i]^2)^(-σ))/denom[i]- rhs
        end
    end
    results = nlsolve(f!,guess)
    return results
end
results = Collocation(last_part, w, param, guess)
est = results.zero
policy = est[1].+est[2].*param.grid + est[3].*param.grid.^2
Approx_Mom_Col = est[1].+est[2].*param.fine_grid + est[3].*param.fine_grid.^2 


function EEE(GH_nodes, GH_weights, Approx, param)

    @unpack β, sⱼ, σ, r, fine_grid, p = param

    EEE    = zeros(1000)                                     # Euler equation erros
    for i in 1:1000

        rhs = β*sⱼ*(1+r)        
        denom = 1/sqrt(π)*(GH_weights'*((1+r)*(Approx[i]).+ GH_nodes).^(-σ))
        EEE[i]   = ((param.fine_grid[i] - Approx[i]))^(-σ)/ denom - rhs
        EEE[i]   = log(abs(EEE[i]))                              # Log of absolute value of EEE
    end
    return EEE
end

# Least squares projection momomials
function Cheb_nodes(n)
    nodes = chebyshev_nodes(1)
    for z in 2:n
        nodes = [nodes ; chebyshev_nodes(z)] 
    end
    nodes = collect(sqrt.(nodes .+ 1))
    return nodes
end

nodes_Cheb = Cheb_nodes(6)
n = length(nodes_Cheb)

function Cheb_grid(n,nodes)
    @unpack minval, maxval = param
    a_grid = zeros(n)
    for i in 1:n
        a_grid[i] = 0.5.*(nodes[i].+1)*(maxval - minval) + minval
    end
    return a_grid
end 

a_grid_Cheb = Cheb_grid(n,nodes_Cheb)
guess = zeros(param.p)

function F(x)
    @unpack β, sⱼ, σ, r = param

    denom = zeros(n)
    F     = zeros(n)
    rhs = β*sⱼ*(1+r)
    for i in 1:n
        denom[i]= 1/sqrt(pi)*(w'*((1+r)*(x[1]+x[2]*a_grid_Cheb[i]+x[3]*a_grid_Cheb[i]^2).+ last_part).^(-σ))
        F[i]    = (((a_grid_Cheb[i]-x[1]-x[2]*a_grid_Cheb[i]-x[3]*a_grid_Cheb[i]^2)^(-σ))/denom[i] - rhs)^2
    end
    return F
end
obj    = x -> dot(F(x),nodes_Cheb)
res    = optimize(obj, [-0.4; 0.47; 0.00], NelderMead())
est    = Optim.minimizer(res)
Optim.minimum(res)
Approx_Mom_LS = est[1].+est[2].*param.fine_grid + est[3].*param.fine_grid.^2 # Approximated policy function
Errors_Mom_LS = EEE(last_part, w, Approx_Mom_LS, param)


Errors_Mom_Col = EEE(last_part, w, Approx_Mom_Col, param)

pol_func = plot(param.fine_grid, [Approx_Mom_Col, Approx_Mom_LS], label=["Collocation" "Least Squares"],
 title="Policy Function", xlabel="Assets Today", ylabel="Assets Next Perdiod")
 savefig(pol_func,"pol_func_proj.png")

euler_eq_error = plot(param.fine_grid, [Errors_Mom_Col, Errors_Mom_LS], label=["Collocation" "Least Squares"],
 title="Euler Equation Errors", xlabel="Grid", ylabel="Error")
 savefig(euler_eq_error,"euler_eq_error_proj.png")


 ##### PSEUDOCODE for Question 2)

 """
    1. Import packages
    2. Define targeted moments (given in PS)
        a)
        3. Define every possible pair of moment condition
        4. Use solve to find values that bring moment conditions to approximately 0
        5. Use estimated results to compute variances of moments
        6. Compute Covariances for all possible combinations
        7. Combine both to generate Variance-Covariance Matrix

        b)
        8. Minimize sum of squared residuals for all for moments with identity matrix as weighting matrix (Γ(θ) in slides)
        9. Compute Φ, as stated in Problem set
        10. Compute Ω with formula on slide 32

        c)
        11. repeat b) with inv(Var-Cov Matrix) as weighting matrix
        
 """
