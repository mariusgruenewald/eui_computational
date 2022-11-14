# Hugget 1996
# Some packages

#using Pkg
#Pkg.add("Distributions")
#Pkg.add("LaTeXStrings")
#Pkg.add("LinearAlgebra")
#Pkg.add("Optim")
#Pkg.add("Plots")
#Pkg.add("Random")
#Pkg.add("StatsBase")
#Pkg.add("Parameters")
#Pkg.add("CSV")
#Pkg.add("DataFrames")
#Pkg.add("Interpolations")

using Distributions
using LaTeXStrings
using LinearAlgebra
using Optim
using Plots
using Random
using StatsBase
using Parameters
using CSV
using DataFrames
using Interpolations

# Hold parameters
@with_kw struct Primitives
    minval::Float64 = 0.0
    maxval::Float64 = 100.0
    na::Int64 = 100
    ρ::Float64 = 0.9
    σ_e::Float64 = 0.2
    nz::Int64 = 10
    q::Float64 = 4.0
    a_grid::Array{Float64} = LinRange(minval, maxval, na)
    N_bar::Int64 = 1
    pop_growth::Float64 = 0.1
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
    tol::Float64 = 0.00001
end

#mutable struct to hold model results
#mutable struct storage
#    trans_mat::Matrix{Float64}
#end

function Tauchen(prim::Primitives)

    trans_mat = zeros(prim.nz, prim.nz) #preallocate value function as a vector of zeros
    uncond_sd = sqrt(prim.σ_e^2/(1-prim.ρ^2)) #calculate unconditional std. dev.
    z_grid = LinRange(-log(prim.q*uncond_sd), log(prim.q*uncond_sd), prim.nz) # set up grid
    #z_grid = log.(z_grid)
    step_size = (prim.q*uncond_sd+prim.q*uncond_sd)/(prim.nz-1) # step-size

    for i in 1:(prim.nz)
        for j in 1:(prim.nz)
                # Border probabilities differ - we need to make two exceptions
                # First exception
            if j == 1
                trans_mat[i,j] = cdf(Normal(),(z_grid[j] + step_size/2 - prim.ρ*z_grid[i])/uncond_sd)

            # Second exception
            elseif j == (prim.nz)
                trans_mat[i,j] = 1 - cdf(Normal(),(z_grid[j] - step_size/2 - prim.ρ*z_grid[i])/uncond_sd)

            # Regular case
            else
                trans_mat[i,j] = cdf(Normal(),(z_grid[j] + step_size/2 - prim.ρ*z_grid[i])/uncond_sd) - cdf(Normal(),(z_grid[j] - step_size/2 - prim.ρ*z_grid[i])/uncond_sd)
            end
        end
    end

    z_grid = exp.(z_grid)
    z_grid = z_grid./mean(z_grid)

    return trans_mat, z_grid
end

path = "LifeTables.txt"
s_j= Array(CSV.File(path, header=0) |> DataFrame)

function population_distribution(prim::Primitives, survival_rates::Matrix{Float64} = s_j)

    ϕ = zeros(length(survival_rates))
    ϕ[1] = prim.N_bar
    for (i, survival_rate) in enumerate(survival_rates[:,1])
        if i < prim.J
            ϕ[i+1] = survival_rate*ϕ[i]/(1+prim.pop_growth)
        end
    end
    ϕ = ϕ./sum(ϕ)

    return ϕ
end

function income(prim::Primitives, ϕ::Vector{Float64}, z_grid::Vector{Float64})

    j_vector = LinRange(1,prim.J,prim.J)
    hh_prod = zeros(prim.J, prim.nz)

    for (i,z) in enumerate(z_grid)
        for (t,age) in enumerate(j_vector)
            if age < prim.J_r
                hh_prod[t,i] = z*(prim.λ_0 + prim.λ_1*age + prim.λ_2*age^2)
            else
                hh_prod[t,i] = 0
            end

        end
    end

    Agg_L = ϕ'*hh_prod

    return hh_prod, j_vector, Agg_L
end


function firm_foc(prim::Primitives, agg_l::Vector{Float64}, r_g::Float64)

    agg_k::Vector{Float64} = ((r_g + prim.δ)/prim.α)^(1/(prim.α-1)).*agg_l

    wage::Vector{Float64} = (1-prim.α).*(agg_k./agg_l).^(prim.α)

    return agg_k, wage
end

function government(prim::Primitives, agg_l::Vector{Float64}, ϕ::Vector{Float64},
     wage::Vector{Float64}, z_grid::Vector{Float64})
    
    θ::Float64 = prim.ω * (sum(ϕ[prim.J_r:prim.J])/sum(ϕ[1:prim.J_r-1]))
    b::Vector{Float64} = similar(z_grid)
    for i in 1:length(z_grid)
        b[i] = θ*wage[i]*agg_l[i]/sum(ϕ[prim.J_r:prim.J])
    end

    return θ, b
end

function non_fin_inc(prim::Primitives, j_vector::Vector{Float64}, z_grid::Vector{Float64}, θ::Float64, hh_prod::Matrix,
    wage::Vector{Float64}, T_g::Float64, b::Vector{Float64})

    d_matrix = zeros(length(j_vector), length(z_grid))
    for (j, age) in enumerate(j_vector)

        if age < prim.J_r
            for (i,z) in enumerate(z_grid)
                d_matrix[j,i] = (1-θ)*hh_prod[j,i]*wage[i] + T_g
            end
        else
            d_matrix[j,:] = b .+ T_g
        end
    end

    return d_matrix
end


function egm(prim::Primitives, z_grid::Vector{Float64}, ϕ::Vector{Float64}, r_g::Float64, j_vector::Vector{Float64}, d_matrix::Matrix{Float64}, Kg::Array{Float64})
    
    # Policy decision of the last period (j=J) is zero anyways
    pol_a = zeros(length(j_vector), length(z_grid), length(prim.a_grid))

    for (j,_) in enumerate(reverse(j_vector))

        if j == prim.J
            for (i,_) in enumerate(z_grid)
                for (q,a) in enumerate(prim.a_grid)
    
                    pol_a[j,i,q] = (1/r_g)*((ϕ[j] * prim.β * r_g)^(-1/prim.σ) * trans_mat[i,:]'*(
                        d_matrix[j,:] .+ r_g*a) - d_matrix[j,i] + a)
                
                end
            
            Kg[j,i,:] = LinearInterpolation(prim.a_grid, pol_a[j,i,:], extrapolation_bc=Line())
    
            end
            
        else

            for (i,_) in enumerate(z_grid)
                for (q,a) in enumerate(prim.a_grid)

                    pol_a[j,i,q] = (1/r_g)*((ϕ[j] * prim.β * r_g)^(-1/prim.σ) * trans_mat[i,:]'*(
                        d_matrix[j,:] .+ r_g*a .+ pol_a[j+1,i,q]) - d_matrix[j,i] + a)
                
                end
            
            Kg[j,i,:] = LinearInterpolation(prim.a_grid, pol_a[j,i,:], extrapolation_bc=Line())

            end
    
        end

    
    end

    return Kg, pol_a
end

prim = Primitives()
trans_mat, z_grid = Tauchen(prim)
ϕ = population_distribution(prim)

#(1/(1 + (prim.ϕ * prim.β * r_g)^(-prim.σ) * r_g))
hh_prod, j_vector, Agg_L = income(prim, ϕ, z_grid);
# Plot lifetime labour supply to check
r_g = 0.02;
Agg_K, Wage = firm_foc(prim, Array(Agg_L'), r_g);
θ, b = government(prim, Array(Agg_L'), ϕ, Wage, z_grid);
T_g = 1.2;

d_matrix = non_fin_inc(prim, Array(j_vector), z_grid, θ, hh_prod, Wage, T_g, b)


Kg = zeros(length(j_vector), length(z_grid), length(prim.a_grid))
Kg, pol_a = egm(prim, z_grid, ϕ, r_g, Array(j_vector), d_matrix, Kg)


Kg[70,1,:] .- prim.a_grid

plot(j_vector, [hh_prod[:,1], hh_prod[:,2], hh_prod[:,3], hh_prod[:,4], hh_prod[:,5], hh_prod[:,6], hh_prod[:,7], hh_prod[:,8], hh_prod[:,9], hh_prod[:,10]])

plot(prim.a_grid, [pol_a[71,1,:], pol_a[70,1,:], pol_a[71,5,:], pol_a[70,5,:]])

function solve_model()