# This is a replication attempt at Hugget 1996 JME
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

# using Pckg
using QuantEcon
using BenchmarkTools
using CSV
using DataFrames
using DelimitedFiles
using Plots
using Interpolations
using Random
using Parameters
using LaTeXStrings
using Statistics

@with_kw struct Primitives
    
    minval::Float64 = 0.0
    maxval::Float64 = 100.0
    na::Int64 = 100
    ρ::Float64 = 0.9
    σ_e::Float64 = 0.2
    nz::Int64 = 5
    q::Float64 = 2
    a_grid::Array{Float64} = LinRange(minval, maxval, na)
    N_bar::Int64 = 1
    n::Float64 = 0.01
    λ_0::Float64 = 0.195
    λ_1::Float64 = 0.107
    λ_2::Float64 = -0.00213
    J_r::Int64 = 41
    σ::Float64 = 2.0
    β::Float64 = 0.96
    a_bar::Float64 = 0.0
    α::Float64 = 0.36
    δ::Float64 = 0.08
    A::Float64 = 1.0
    J::Int64 = 71
    ω::Float64 = 0.5
    μ::Float64 = 0

end

function grids(prim::Primitives)
    
    # use Tauchen to discretize
    @unpack na, minval, maxval, nz, ρ, σ_e, μ, q = prim 

    a_grid = zeros(na)
    a_grid = collect(range(minval, maxval, length=na))

    mc = QuantEcon.tauchen(nz,ρ,σ_e,μ,2)
    z_logs = mc.state_values
    trans_mat = mc.p

    z_grid = zeros(nz)
    for i = 1:nz
        z_grid[i] = exp(z_logs[i] - 0.5*(σ_e^2/(1-ρ^2)))
    end

    return a_grid, trans_mat, z_grid
end

# Compute Invariant Distribution
function invariant_prod_dist(prim::Primitives, Phi::Matrix{Float64})
    S::Int64 = prim.nz
    Phi_sd = ones(1,S)/S
    diff = 1
    tol::Float64 = 0.0000001;
    while abs(diff) > tol
        Phi_sd1 = Phi_sd*Phi
        diff = (Phi_sd1-Phi_sd)[argmax(Phi_sd1-Phi_sd)]
        Phi_sd = Phi_sd1
    end
    return Phi_sd[1,:]
end

function population_distribution(prim::Primitives)
    # survival rates
    @unpack n, J, J_r = prim

    tmp = CSV.read("LifeTables.txt", DataFrame, header=false)
    s_j = tmp[!,1]
    cum_prod = cumprod(s_j)
    entry_pop = (1 + n).^(0:J)
    popul = ones(J)
    for j = 1:J
        popul[j] = entry_pop[J-j+1]*cum_prod[j]
    end
    tot_pop = sum(popul)
    Ψ = zeros(J)
    for p = 1:J
        Ψ[p] = popul[p]/tot_pop
    end

    return s_j, Ψ
end

function efficiency(prim::Primitives, z_grid::Vector{Float64})

    @unpack J, J_r, λ_0, λ_1, λ_2, nz = prim 

    # get the e matrix
    P = zeros(J)
    for j = 1:(J_r-1)
        P[j] = λ_0 + λ_1*j + λ_2*(j^2)  
    end
    hh_prod = zeros(nz,J)

    for j = 1:(J_r-1)
        for i = 1:nz
            hh_prod[i,j] = z_grid[i]*P[j]
        end 
    end

    return P, hh_prod
end

function agg_labour(prim::Primitives, Ψ::Vector{Float64}, Φ::Vector{Float64}, P::Vector{Float64})
    # Aggregate labour supply

    j_labor = zeros(prim.J)
    for j = 1:prim.J
        j_labor[j] = P[j]*Ψ[j]
    end

    # Take inv. stationary distribution into account

    Agg_L = sum(j_labor)
    #Agg_L = 1.038
    return Agg_L
end

function firm(prim::Primitives, agg_l::Float64, r_g::Float64)

    @unpack α, δ = param
    # Aggregate Capital from FOC of firm
    agg_k = ( (r_g + δ)/(α*A) * (Agg_L)^(1/(1-α)) )^(1/(α-1))
    # Wage from other FOC
    wage = (1-α)*A*(agg_k)^(α)* (Agg_L)^(-α)

    return agg_k, wage
end

function government(prim::Primitives, Ψ::Vector{Float64}, wage::Float64, agg_l::Float64)
    @unpack ω, J, J_r = prim

    θ = ω*sum(Ψ[J_r:J])/sum(Ψ[1:(J_r-1)]) 
    b_t = θ*wage*agg_l/sum(Ψ[J_r:J])
    b = zeros(J)
    b[J_r:J] = ones(J - J_r+1)*b_t  

    return θ, b
end


function egm(prim::Primitives, r_g::Float64, T_g::Float64, wage::Float64, b::Array{Float64,1}, e, θ, s_j, trans_mat, z_grid, a_grid)
    
    @unpack nz, na, J, J_r, β, σ  = prim

    Kg = zeros(nz, na, J)
    for j in reverse(1:J-1)
        println(j)

        for (z,_) in enumerate(z_grid)

            d1 = (1-θ)*e[z,j]*wage + b[j] + T_g
            d2 = (1-θ)*e[:,j+1]*wage .+ b[j+1] .+ T_g

            pol_a = zeros(na)
            for (q,a) in enumerate(a_grid)

                pol_a[q] =(( β*(1+r_g)*s_j[j] * trans_mat[z,:]'*((1+r_g)*a .+ d2 - Kg[:,q,j+1]).^(-σ))^(-1/σ) + a - d1)/(1+r_g)
                
            end
                    
            nodes = (pol_a,) # Define the nodes
            itp = interpolate(nodes, a_grid, Gridded(Linear())) # Perform Interpolations
            etpf = extrapolate(itp, Line()) # Set up environment for extrapolation
            kg_temp = etpf(a_grid) # Perform extrapolation

            # Make sure boundaries are kept
            kg_temp[(kg_temp .< 0)] .= 0
            kg_temp[(kg_temp .> a_grid[na])] .= a_grid[na]
            Kg[z,:,j] = kg_temp
        end


    end

    return Kg
end



function solve_model()

    prim = Primitives()
    a_grid, trans_mat, z_grid = grids(prim)
    Φ = invariant_prod_dist(prim, trans_mat)
    s_j, Ψ = population_distribution(prim)
    P, hh_prod = efficiency(prim, z_grid)
    Agg_L = sum(P.*Ψ)
    r_g = 0.02
    Agg_K, Wage = firm(prim, Agg_L, r_g)
    θ, b = government(prim, Ψ, Wage, Agg_L)
    T_g = 1.2
    Kg = egm(prim, r_g, T_g, Wage, b, hh_prod, θ, s_j, trans_mat, z_grid, a_grid)

    return a_grid, trans_mat, z_grid, Φ, s_j, Ψ, P, hh_prod, Agg_L, Agg_K, Wage, θ, b, Kg
end

a_grid, trans_mat, z_grid, Φ, s_j, Ψ, P, hh_prod, Agg_L, Agg_K, Wage, θ, b, Kg = solve_model()

## Calculate and Plot Results
gr()
savings = Kg[:,:,30] .- a_grid'
sav_fig30 = plot(a_grid, savings',label = [L"z_1" L"z_2" L"z_3" L"z_4" L"z_5"], dpi=300, title = "Savings Decision for 30")
xaxis!(L"a")
yaxis!(L"a' - a")
savefig(sav_fig30,"savpol_fig30.png")

savings70 = Kg[:,:,70] .- a_grid'
sav_fig70 = plot(a_grid, savings70',label = [L"z_1" L"z_2" L"z_3" L"z_4" L"z_5"], dpi=300, title = "Savings Decision for 70")
xaxis!(L"a")
yaxis!(L"a' - a")
savefig(sav_fig70,"savpol_fig70.png")

prim = Primitives()
prod_fig = plot(1:prim.J, hh_prod', label = [L"z_1" L"z_2" L"z_3" L"z_4" L"z_5"], dpi=300, title = "Productivity over the years")
xaxis!(L"age j")
yaxis!(L"e(j,z)")
savefig(prod_fig,"efficiency.png")