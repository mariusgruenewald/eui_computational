# This is a replication attempt at the Krusell Smith with EGM
# Problem Set 1, Computational Methods
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
#import Pkg; Pkg.add("TensorCore")
#import Pkg; Pkg.add("GLM")
#import Pkg; Pkg.add("DataFrames")

# using Pckg

using BenchmarkTools
using Plots
using Random
using Parameters
using LaTeXStrings
using Statistics
using StatsBase
using TensorCore
using LogExpFunctions
using NNlib


Random.seed!(1234)

@with_kw struct discrete_params_evs
    
    β::Float64 = 0.96
    σ::Float64 = 1.0
    z::Float64 = 1.0
    Φ::Float64 = 0.75
    r::Float64 = 0.03
    wage::Float64 = 1.0
    sigma_ϵ::Float64 = 0.01

    minval::Float64 = 0.0
    maxval::Float64 = 30.0
    na::Int64 = 200
    a_grid_lin::Vector{Float64} = collect(range(minval, maxval, na))
    a_grid_log::Vector{Float64} = exp.(LinRange(log(minval+1),log(maxval+1),na)).-1

    nn::Int64 = 2
    n_grid::Vector{Float64} = [0,1]

    tol::Float64 = 1e-5
    maxiter::Int64 = 200
end

function util_(c)
    u = prim.σ == 1 ? x -> log.(x) : x -> (x.^(1 - prim.σ) .- 1) ./ (1 - prim.σ)
    return u(c)
end

function labour_supply(prim::discrete_params_evs)

    @unpack σ, na, a_grid_log, r, wage, Φ = prim

    labour = zeros(na, na)
    u = zeros(na, na)


    for (i, a) in enumerate(a_grid_log)
        for (j, a_next) in enumerate(a_grid_log)

            if (1+r) * a - a_next >= 0

                lhs = util_(wage + (1+r) * a - a_next) - util_((1+r) * a - a_next)

                if lhs >= Φ
                    labour[i,j] = 1
                end

            end

            cons = wage * labour[i,j] + (1+r) * a - a_next

            if cons >= 0

                u[i,j] = util_(cons) - Φ * labour[i,j]

            else u[i,j] = -Inf

            end

        end

    end

    return labour, u
end

function value_function_iter(prim::discrete_params_evs, u::Matrix{Float64}, labour::Matrix{Float64})

    @unpack tol, maxiter, na, β, a_grid_log, wage, r, sigma_ϵ = prim

    V0 = zeros(na)
    V1 = zeros(na)#
    pol_idx = zeros(na)
    policy_n = zeros(na)
    err = 1
    count = 1

    while err > tol && count < maxiter 
        
        for (i, _) in enumerate(a_grid_log)

            v_temp = u[i,:] + β * sigma_ϵ .* logexp(V0./sigma_ϵ)

            V1[i] = maximum(v_temp)
            pol_idx[i] = softmax(v_temp)

        end

        err = maximum(abs.(V1 - V0))
        V0 = copy(V1)
        count = count + 1

    end

    policy_a = a_grid_log[Int.(pol_idx)]

    for (i, _) in enumerate(a_grid_log)

        policy_n[i] = labour[i, Int.(pol_idx[i])]

    end

    policy_c = wage .* policy_n + (1+r) .* a_grid_log - policy_a

    return policy_c, policy_a, policy_n, V0

end


function solve_model()

    prim = discrete_params()

    labour, util = labour_supply(prim)
    policy_c, policy_a, policy_n, V0 = value_function_iter(prim, util, labour)

    return labour, util, policy_c, policy_a, policy_n, V0

end

labour, util, policy_c, policy_a, policy_n, V0 = solve_model()

prim = discrete_params_evs()
a_grid = prim.a_grid_log
vfi_discrete_ev = plot(a_grid, V0, label = ["Value Function"], dpi=300, title = "Value Function")
xaxis!(L"a")
savefig(vfi_discrete_ev,"vfi_discrete_ev.png")

pol_discrete = plot(a_grid, policy_a, label = ["Savings"], dpi=300, title = "Capital Policy Function")
xaxis!(L"a")
savefig(pol_discrete,"pola_discrete_ev.png")


cpol_discrete = plot(a_grid, policy_c, label = ["Consumption"], dpi=300, title = "Consumption Policy Function")
xaxis!(L"a")
savefig(cpol_discrete,"polc_discrete_ev.png")

npol_discrete = plot(a_grid, policy_n, label = ["Labour"], dpi=300, title = "Labour Policy Function")
xaxis!(L"a")
savefig(npol_discrete,"poln_discrete.png")