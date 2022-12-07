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

using QuantEcon
using BenchmarkTools
using Plots
using Interpolations
using Random
using Parameters
using LaTeXStrings
using Statistics
using StatsBase
using TensorCore
using GLM
using DataFrames

Random.seed!(1234)

@with_kw struct Primitives
    
    Z::Float64 = 1.0
    β::Float64 = 0.96
    σ::Float64 = 1.0
    α::Float64 = 0.33
    δ::Float64 = 0.05
    A::Float64 = 1.0

    nz::Int64 = 2
    z_grid::Vector{Float64} = [0.1, 1.0]
    trans_mat::Array{Float64, 2} = [0.9 0.1; 0.1 0.9]

    Z_mat::Array{Float64, 2} = [0.5 0.5; 0.1 0.9]
    Z_agg_grid::Vector{Float64} = [0.99, 1.01]

    minval_agg::Float64 = 5.33*0.9 # 5.33 is the GE capital from previous exercise
    maxval_agg::Float64 = 5.33*1.1
    n_agg::Int64 = 5
    K_agg_grid::Vector{Float64} = exp.(LinRange(log(minval_agg+1),log(maxval_agg+1), n_agg)).-1

    minval::Float64 = 0.0
    maxval::Float64 = 30.0
    na::Int64 = 200
    a_grid_lin::Vector{Float64} = collect(range(minval, maxval, na))
    a_grid_log::Vector{Float64} = exp.(LinRange(log(minval+1),log(maxval+1),na)).-1

    tol::Float64 = 1e-5
    maxiter::Int64 = 200
end


function agg_labour(prim::Primitives)
    
    @unpack nz, trans_mat, z_grid = prim

    Phi_sd = ones(1,nz)/nz
    diff = 1
    tol = 0.0000001;
    while abs(diff) > tol
        Phi_sd1 = Phi_sd*trans_mat
        diff = (Phi_sd1-Phi_sd)[argmax(Phi_sd1-Phi_sd)]
        Phi_sd = Phi_sd1
    end

    L_Agg = Phi_sd*z_grid

    return Phi_sd, L_Agg[1]
end

function Z_agg_dist(prim::Primitives)
    
    @unpack Z_mat, Z_agg_grid = prim

    Phi_sd = ones(1,nz)/nz
    diff = 1
    tol = 0.0000001;
    while abs(diff) > tol
        Phi_sd1 = Phi_sd*trans_mat
        diff = (Phi_sd1-Phi_sd)[argmax(Phi_sd1-Phi_sd)]
        Phi_sd = Phi_sd1
    end

    L_Agg = Phi_sd*z_grid

    return Phi_sd, L_Agg[1]
end

# Rework and solve for 
function firm_decision(prim::Primitives, Agg_L::Float64, K_agg::Float64, Z_agg::Float64)
    
    @unpack α, δ, β = prim

    r = α .* Z_agg * K_agg'.^(α-1).* (Agg_L)^(1-α) .- δ
    wage = (1-α).* Z_agg *(K_agg').^(α) .* (Agg_L)^(-α)
    
   return r, wage
end


function util_(prim::Primitives, c)
    @unpack σ = prim
    u = σ == 1 ? x -> log.(x) : x -> (x.^(1 - σ) .- 1) ./ (1 - σ)
    return u(c)
end


function agg_capital_hh(prim::Primitives, dist_fin::Matrix{Float64}, Kg::Matrix{Float64}, Agg_L::Float64, Z_now::Float64)
    
    @unpack A, α = prim

    agg_temp = sum(dist_fin.* Kg)

    Y = Z_now*(sum(agg_temp))^(α) * (Agg_L)^(1 - α)

    return agg_temp, Y
end


# Call this function for transition
function EGM_transition(prim::Primitives, wage::Float64, r::Float64, cpol::Matrix{Float64})

    # Here, we do not find convergence. We just take the previous results and compute the responses.
    # This gives inputs for next period. We simulate the economy sequentionally.

    @unpack nz, na, β, σ, z_grid, trans_mat, a_grid_log  = prim
    
    a_grid = copy(a_grid_log)

    c_impl = ((1+r) * β * Z_mat ⊡ (trans_mat * cpol.^(-σ))).^(-1/σ)
    
    k_impl =  (c_impl + ones(nz,1)*a_grid' - z_grid*wage*ones(1,na))./(1+r)
    kpol_egm_trans = zeros(nz, na)

    for (z,_) in enumerate(z_grid)
        nodes = (vec(k_impl[z,:]),) # Define the nodes
        itp = interpolate(nodes, a_grid, Gridded(Linear())) # Perform Interpolations
        etpf = extrapolate(itp, Line()) # Set up environment for extrapolation
        kpol_egm_trans[z,:] = etpf(a_grid) # Perform extrapolation
    end

    # Make sure boundaries are kept
    kpol_egm_trans[(kpol_egm_trans .< 0)] .= 0
    kpol_egm_trans[(kpol_egm_trans .> a_grid[na])] .= a_grid[na]


    cpol1_trans = (1+r)*ones(nz,1).*a_grid' - kpol_egm_trans[:,:] + z_grid*wage*ones(1,na) 


    return kpol_egm_trans, cpol1_trans
end

function sim_Z_shocks(prim::Primitives, n_sim::Int64)

    return simulate(MarkovChain(prim.Z_mat), n_sim),  simulate(MarkovChain(prim.trans_mat), n_sim)
end

function K_state(Z_seq::Vector{Int64}, β₀ₗ::Float64, β₁ₗ::Float64, β₀ₕ::Float64, β₁ₕ::Float64, n_sim::Int64)

    mc = QuantEcon.tauchen(nz,ρ,σ_e,μ,2)
    z_logs = mc.state_values
    trans_mat = mc.p

    # Initial guess 1
    K_seq = ones(n_sim+1)

    for (i,Z) in enumerate(Z_seq)
        if Z == 1
            K_seq[i+1] = β₀ₗ + β₁ₗ*K_seq[i]
        else 
            K_seq[i+1] = β₀ₕ + β₁ₕ*K_seq[i]
        end
    end


    return K_seq
end

# Call this function for transition
function young_2010_transition(prim::Primitives, kpol::Matrix{Float64}, dist::Matrix{Float64})
 
    # Needs to be run for every period
    @unpack nz, na, trans_mat, a_grid_log = prim
    a_grid = copy(a_grid_log)
    # Locate policy function in the grid, i.e. find indeces above and below
    ind_low = ones(nz,na)
    for a in 2:na
        for z in 1:nz
            ind_low[z,findall(x -> x >= a_grid[a], kpol[z,:])] .=  a
            ind_low[z,findall(x -> x >= na, ind_low[z,:])] .=  na-1
        end
    end
    
    ind_up = ind_low .+ 1

    wabove = ones(nz, na)
    wbelow = ones(nz, na)
    
    # However, policy functions may not be on the grid. Assign share of distribution to gridpoints, what percent of population 
    # will end up above or below policy function. Shouldn't change much => if it does, make grid finer.
    for z in 1:nz
        for i in 1:na
            wabove[z,i] =  (kpol[z,i] - a_grid[Int(ind_low[z,i])]) / (a_grid[Int(ind_low[z,i]) + 1] - a_grid[Int(ind_low[z,i])])
            wabove[z,i] = min(wabove[z,i],1)
            wabove[z,i] = max(wabove[z,i],0)
            wbelow[z,i] = 1-wabove[z,i]
        end
    end 

    # Compute distributional matrix. Here, we do not find invariant distribution but just iteration one period further
    Γ = zeros(nz, na)
    for z in 1:nz
        for i in 1:na
            # Based on matrix today, where will the lower index be tomorrow
            Γ[:, Int.(ind_low[z,i])] = Γ[:, Int.(ind_low[z,i])] .+ dist[z,i]*wbelow[z,i]*trans_mat[z,:]
            # Based on matrix today, where will the upper index be tomorrow
            Γ[:, Int.(ind_up[z,i])] = Γ[:, Int.(ind_up[z,i])] .+ dist[z,i]*wabove[z,i]*trans_mat[z,:]
        end
    end

    return Γ
end


function krusell_smith(n_sim)

    count = 0
    error = 1

    prim = Primitives()

    # Storage
    wage = zeros(n_sim)
    r = zeros(n_sim)
    K_agg = zeros(n_sim)

    # Initial Guess of distribution
    dist = (1/(prim.nz * prim.na)) * ones(prim.nz, prim.na, n_sim)
    kpol = ones(prim.nz, prim.na)
    cpol = ones(prim.nz, prim.na)
    
    K_agg[1] = sum(dist[:,:,t].*kpol[:,:])

    _, L_Agg = agg_labour(prim)
    Z_index, _ = sim_Z_shocks(prim, n_sim)

    # Guess ALM
    KZ_mat = zeros(n_sim, 3)
    β₀ₗ, β₁ₗ, β₀ₕ, β₁ₕ = 0.1, 0.1, 0.9, 0.9
    β_vec = [0.1, 0.1, 0.9, 0.9]

    while error > prim.tol && count < prim.maxiter

        count = count + 1
        # Compute Aggregate Capital Supply by Households
        
        for t in 1:n_sim-1

            if Z_index[t] == 1
                K_agg[t+1] = β₀ₗ + β₁ₗ*K_agg[t]
            else
                K_agg[t+1] = β₀ₕ + β₁ₕ*K_agg[t]
            end
            # Capital Demand & Wage
            r[t], wage[t] = firm_decision(prim, L_Agg, K_agg[t], prim.Z_agg_grid[Int(Z_index[t])])
            kpol, cpol = EGM_transition(prim, wage[t], r[t], cpol)
            dist[:,:,t+1] = young_2010_transition(prim, kpol, dist[:,:,t])

        end

        KZ_mat[:,1] = K_agg
        KZ_mat[:,2] = Z_index
        KZ_mat[2:10000, 3] = K_agg[1:10000-1]

        KZ_low = KZ_mat[KZ_mat[:,2] .== 1, :]
        KZ_high = KZ_mat[KZ_mat[:,2] .== 2, :]

        KZ_low_df = DataFrame(KZ_low, :auto)
        KZ_high_df = DataFrame(KZ_high, :auto)
        ols_low = lm(@formula(x3~x1), KZ_low_df)
        ols_high = lm(@formula(x3~x1), KZ_high_df)
        β_0l = GLM.coef(ols_low)[1]
        β_1l = GLM.coef(ols_low)[2]
        β_0h = GLM.coef(ols_high)[1]
        β_1h = GLM.coef(ols_high)[2]
        β_vec_model = [GLM.coef(ols_low)[1], GLM.coef(ols_low)[2], GLM.coef(ols_high)[1], GLM.coef(ols_high)[2]]

        error = maximum(abs.(β_vec - β_vec_model))
        println("Current Error:", error, " at iteration:", count)

        β₀ₗ = 0.9*β₀ₗ + 0.1*β_0l
        β₁ₗ = 0.9*β₁ₗ + 0.1*β_1l
        β₀ₕ = 0.9*β₀ₕ + 0.1*β_0h
        β₁ₕ = 0.9*β₁ₕ + 0.1*β_1h
        
    end

    return dist, KZ_mat, K_agg, r, Z_index, β_vec_model
end


dist, KZ_mat, K_agg, r, Z_index, β_vec_model = krusell_smith(10000)


# We are interested in kpol and agg_k_hh
dist, kpol, agg_k_hh, r, wage = solve_general_model(0.03)

