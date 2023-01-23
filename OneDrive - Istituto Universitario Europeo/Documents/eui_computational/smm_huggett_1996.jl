# This is a replication attempt at estimating Hugget 1996 JME
# Problem Set 3, Life Cycle Course
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
using StatsBase
using JuMP
using Ipopt
using LinearAlgebra
using NLsolve
using Optim

Random.seed!(1234)
@with_kw struct Primitives_Q3
    
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
    len_panel::Int64 = 1000

end

function grids(prim)
    
    # use Tauchen to discretize
    @unpack na, minval, maxval, nz, ρ, σ_e, μ, q = prim 

    a_grid = zeros(na)
    a_grid = collect(range(minval, maxval, length=na))

    mc = QuantEcon.tauchen(nz,ρ,σ_e,μ,2)
    z_logs = mc.state_values
    trans_mat = mc.p

    z_grid = zeros(nz)
    for i = 1:nz
        z_grid[i] = exp(z_logs[i])# - 0.5*(σ_e^2/(1-ρ^2)))
    end

    return a_grid, trans_mat, z_grid
end

# Compute Invariant Distribution
function invariant_prod_dist(prim, Phi::Matrix{Float64}, z_grid::Vector{Float64})
    S::Int64 = prim.nz
    Phi_sd = ones(1,S)/S
    diff = 1
    tol::Float64 = 0.0000001;
    while abs(diff) > tol
        Phi_sd1 = Phi_sd*Phi
        diff = (Phi_sd1-Phi_sd)[argmax(Phi_sd1-Phi_sd)]
        Phi_sd = Phi_sd1
    end

    z_grid = z_grid./(z_grid' *Phi_sd[1,:])

    return Phi_sd[1,:], z_grid
end

function population_distribution(prim)
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

function efficiency(prim, z_grid::Vector{Float64})

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


function firm(prim, Agg_L::Float64, r_g::Float64)

    @unpack α, δ, A = prim
    # Aggregate Capital from FOC of firm
    agg_k = ( (r_g + δ)/((α*A)*(Agg_L)^(1-α)) )^(1/(α-1))
    # Wage from other FOC
    wage = (1-α)*A*(agg_k)^(α)* (Agg_L)^(-α)

    return agg_k, wage
end


function government(prim, Ψ::Vector{Float64}, Wage::Float64, Agg_L::Float64)
    @unpack ω, J, J_r = prim

    θ = ω*sum(Ψ[J_r:J])/sum(Ψ[1:(J_r-1)])

    b_t = θ*Wage*Agg_L/(sum(Ψ[prim.J_r:prim.J]))
    b = zeros(J)
    b[J_r:J] = ones(J - J_r+1)*b_t

    return θ, b
end

# Change function
function egm(prim, r_g::Float64, T_g::Float64, wage::Float64, b::Array{Float64,1}, e::Matrix{Float64}, θ::Float64, s_j::Vector{Float64},
     trans_mat::Matrix{Float64}, z_grid::Vector{Float64}, a_grid::Vector{Float64})
    
    @unpack nz, na, J, J_r, β, σ  = prim

    Kg = zeros(nz, na, J)
    cpol = zeros(nz, na, J)
    d1 = zeros(nz, J)

    for j in reverse(1:J-1)

        for (z,_) in enumerate(z_grid)

            d1[z,j] = (1-θ)*e[z,j]*wage + b[j] + T_g
            d2 = (1-θ)*e[:,j+1]*wage .+ b[j+1] .+ T_g

            pol_a = zeros(na)
            for (q,a) in enumerate(a_grid)

                pol_a[q] =(( β*(1+r_g)*s_j[j] * trans_mat[z,:]'*((1+r_g)*a .+ d2 - Kg[:,q,j+1]).^(-σ))^(-1/σ) + a - d1[z,j])/(1+r_g)
                
            end
                    
            nodes = (pol_a,) # Define the nodes
            itp = interpolate(nodes, a_grid, Gridded(Linear())) # Perform Interpolations
            etpf = extrapolate(itp, Line()) # Set up environment for extrapolation
            kg_temp = etpf(a_grid) # Perform extrapolation

            # Make sure boundaries are kept
            kg_temp[(kg_temp .< 0)] .= 0
            #kg_temp[(kg_temp .> a_grid[na])] .= a_grid[na]
            Kg[z,:,j] = kg_temp

        end

        cpol[:,:,j] = (1+r_g)*ones(nz,1).*a_grid' - Kg[:,:,j] + d1[:,j]*ones(1,na)
    end

    

    return Kg, d1, cpol
end


function simulation_shocks(prim, trans_mat::Matrix{Float64}, n_sim::Int64)

    @unpack J, nz = prim
    z_path_index = ones(n_sim, J) # storage with first element set

    for sim in 1:n_sim

        for age in 2:J-1
            z_path_index[sim, age] = sample(1:nz, Weights(trans_mat[Int.(z_path_index[sim, age-1]),:]))
        end
    end
    
    return z_path_index
end


function capital_simulation(prim, z_path_index::Matrix{Float64}, Kg::Array{Float64, 3}, n_sim::Int64)

    @unpack nz, na, J = prim

    capital_path = zeros(n_sim, J)
    for j = 2:J
        for n = 1:n_sim
            nodes       = (a_grid,)
            itp         = interpolate(nodes, Kg[Int.(z_path_index[n,j]),:,j-1], Gridded(Linear()))
            extrp       = extrapolate(itp,Line())
            capital_path[n,j]  = extrp(capital_path[n,j-1])
        end
    end
    return capital_path
end


function young_2010(prim, a_grid::Vector{Float64}, Kg::Array{Float64, 3}, trans_mat::Matrix{Float64}, Φ::Vector{Float64})

    @unpack J, nz, na = prim

    #Γ = zeros(nz*na, nz*na, J)

    # Intialize at lowest point of a grid
    #indbelow = ones(nz, na, J)
    #indabove = ones(nz, na, J)
    # Capital index that is below value of value in policy function
 
    #for i=2:na
    #    cart = findall(Kg[:,i,:] .>= a_grid[i])
    #    for index in cart
    #        indbelow[index[1], i, index[2]] = i
    #    end
    #end
    #indabove = indbelow .+ 1

    ind_low = ones(nz,na, J)
    ind_low .= 1
    for j = 2:J
        for a in 2:na
            for z in 1:nz
                ind_low[z,findall(x -> x >= a_grid[a], Kg[z,:,j]), j] .=  a
                ind_low[z,findall(x -> x >= na, ind_low[z,:,j]), j] .=  na-1
            end
        end
    end
    ind_up = ind_low .+ 1

    wabove = ones(nz, na, J)
    wbelow = ones(nz, na, J)

    for j in 1:J
        for z in 1:nz
            for i in 1:na
                wabove[z,i,j] =  (Kg[z,i,j] - a_grid[Int(ind_low[z,i,j])]) / (a_grid[Int(ind_low[z,i,j]) + 1] - a_grid[Int(ind_low[z,i,j])])
                wabove[z,i,j] = min(wabove[z,i,j],1)
                wabove[z,i,j] = max(wabove[z,i,j],0)
                wbelow[z,i,j] = 1-wabove[z,i,j]
            end
        end 
    end

    Γ = zeros(nz*na, nz*na, J)
    dist_fin = zeros(nz*na, J)
    dist_fin[1:5, 1] = Φ

    for j in 1:J
        for z in 1:nz
            for i in 1:na
                Γ[Int.((i-1)*nz+z), Int.((ind_low[z,i,j]-1)*nz+1:ind_low[z,i,j]*nz), j] = wbelow[z,i,j]*trans_mat[z,:]
                Γ[Int.((i-1)*nz+z), Int.((ind_up[z,i,j]-1)*nz+1:ind_up[z,i,j]*nz), j] = wabove[z,i,j]*trans_mat[z,:]
            end
        end

        if (j > 1) & (j < J)
            dist_fin[:,j] = dist_fin[:,j-1]'*Γ[:,:,j]
        end
    end


    dist_fin = reshape(dist_fin, (nz, na, J))

    return dist_fin, Γ
end


function agg_capital_hh(prim, dist_fin::Array{Float64, 3}, r_g::Float64, Kg::Array{Float64, 3}, Ψ::Vector{Float64}, s_j::Vector{Float64},
    Agg_L::Float64)
    
    @unpack na, nz, J, A, α = prim

    agg_temp = zeros(J)
    agg_acc_b = zeros(J)
    for j in 1:J
        agg_temp[j] = Ψ[j]*sum(dist_fin[:,:,j].*Kg[:,:,j])
        agg_acc_b[j] = sum(sum(((1 + r_g).*Kg[:,:,j].*dist_fin[:,:,j]) .* Ψ[j] .* (1 .- s_j[j])))
    end

    Y = A*(sum(agg_temp))^(α) * (Agg_L)^(1 - α)

    return sum(agg_temp), sum(agg_acc_b), Y
end


# Data (True) Moments
function data_moments(prim, cpol::Array{Float64, 3}, Kg::Array{Float64, 3}, dist_fin::Array{Float64, 3}, s_j::Vector{Float64})

    @unpack J, a_grid, len_panel= prim
    #dist_path = ones(1000, 71) # everybody starts in the same low state productivity
    capital_path = zeros(len_panel, J) 
    cap_cnstr = zeros(len_panel, J) 
    death_age = zeros(len_panel)
    c_path = zeros(len_panel, J) 

    # Survival of individuals according to probs of survival

    s_rand = rand(Random.seed!(1234),len_panel,J)

    # create identifier where they die
    death_age[:] .= 71
    for sim = 1:len_panel
        for age = 2:J
            if s_rand[sim,age] > s_j[age]
                death_age[sim] = age
                break
            end
        end
    end

    # Simulate Capital and Consumption
    for sim in 1:len_panel
        for age in 2:J-1
            
            capital_path[sim, age+1] = sample(reshape(Kg[:, :, age], 500), Weights(reshape(dist_fin[:, :, age], 500)))
            if capital_path[sim, age+1] <= a_grid[1] # keep track of constraint agents
                cap_cnstr[sim, age] = 1
            end
            #dist_path[sim, age] = sample(reshape(dist_fin[:, :, age], 500), Weights(reshape(dist_fin[:, :, age], 500)))
            c_path[sim, age] = sample(reshape(cpol[:, :, age], 500), Weights(reshape(dist_fin[:, :, age], 500)))
        end
    end

    #capital_panel = individuals.*capital_path
    #probs_panel = individuals.*dist_path
    #cons_panel = individuals.*c_path

    # Consumption Growth by age => we have 71 moments, 
    #cons_growth = zeros(J)
    #for age in 2:J
    #    cons_growth[age] = mean(cons_panel[:, age-1]/cons_panel[:, age])
    #end

    return death_age, capital_path, cap_cnstr, c_path
end


function solve_model(prim, r_g::Float64, T_g::Float64)

    # Step 1-6
    a_grid, trans_mat, z_grid = grids(prim)
    Φ, z_grid = invariant_prod_dist(prim, trans_mat, z_grid)
    s_j, Ψ = population_distribution(prim)
    P, hh_prod = efficiency(prim, z_grid)
    Agg_L = sum(P.*Ψ)

    # Storage Step 7
    Agg_K = 0
    Wage = 0
    θ = 0
    b = 0

    # Storage Step 8, 9 & 10
    Y = 0
    dist_fin = zeros(prim.nz, prim.na, prim.J)
    Kg = zeros(prim.nz, prim.na, prim.J)
    agg_k_hh = 0
    T_g_1 = 0
    cpol = zeros(prim.nz, prim.na, prim.J)
    Kg = zeros(prim.nz, prim.na, prim.J)
    Γ = 0
    d1 = zeros(prim.nz, prim.J)

    # initiate errors
    count_r = 0
    err_r  = 1.0

    while (err_r > 1e-5) && count_r < 1000
        # New Firm and Government decision
        Agg_K, Wage = firm(prim, Agg_L, r_g)
        θ, b = government(prim, Ψ, Wage, Agg_L)

        count_t = 0
        err_t   = 1.0
        while (err_t > 1e-5) && count_t < 1000


            count_t += 1
            # Get policy function with updated r_g & T_g
            Kg, d1, cpol = egm(prim, r_g, T_g, Wage, b, hh_prod, θ, s_j, trans_mat, z_grid, a_grid)
            
            # Calculate distribution with new policy function
            dist_fin, Γ = young_2010(prim, a_grid, Kg, trans_mat, Φ)
            # Get aggregate capital supply by HH and aggregate accidental bequests
            Agg_K_hh, T_g_1, _ = agg_capital_hh(prim, dist_fin, r_g, Kg, Ψ, s_j, Agg_L)
            agg_k_hh = copy(Agg_K_hh)
            # calculate error and update guess
            err_t = abs(T_g_1- T_g)
            T_g = copy(T_g_1)
        end 

        # Back out implied interst rate
        r_g_1 = prim.α*(agg_k_hh^(prim.α-1))*(Agg_L^(1-prim.α)) - prim.δ            
        err_r = abs(r_g_1-r_g)    
        println("err_r = ", round.(err_r; digits=6)," at iteration ", count_r," with r_g_1 = ",round.(r_g_1; digits=4)," and r_g = ",round.(r_g; digits=4))                 
        r_g = 0.5*r_g + 0.5*r_g_1
        

    end 

    # Storage for data_moments
    capital_path = zeros(prim.len_panel, prim.J) 
    cap_cnstr = zeros(prim.len_panel, prim.J) 
    death_age = zeros(prim.len_panel)
    c_path = zeros(prim.len_panel, prim.J) 
    individuals = ones(prim.len_panel, prim.J)

    # Get true moments
    death_age, capital_path, cap_cnstr, c_path = data_moments(prim, cpol, Kg, dist_fin, s_j)

    return a_grid, Φ, s_j, Ψ, P, hh_prod, d1, Agg_L, Agg_K, Wage, θ, b, Kg, dist_fin, agg_k_hh, T_g_1, r_g, cpol, death_age, capital_path, cap_cnstr, c_path
end

# Run the economy
prim= Primitives_Q3()
_, _, _, _, _, _, _, _, _, Wage, θ, b, Kg, _, agg_k_hh, T_g_1, r_g, cpol, death_age, capital_path, cap_cnstr, c_path = solve_model(prim, 0.02, 1.2)



# We want to estimate σ and β (2 moments) and therefore define new struct
@with_kw struct Primitives_SMM
    
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
    #σ::Float64 = 2.0
    #β::Float64 = 0.96
    a_bar::Float64 = 0.0
    α::Float64 = 0.36
    δ::Float64 = 0.08
    A::Float64 = 1.0
    J::Int64 = 71
    ω::Float64 = 0.5
    μ::Float64 = 0
    len_panel::Int64 = 1000

end

# Change function
function egm_smm(prim, r_g::Float64, T_g::Float64, wage::Float64, b::Array{Float64,1}, e::Matrix{Float64}, θ::Float64, s_j::Vector{Float64},
    trans_mat::Matrix{Float64}, z_grid::Vector{Float64}, a_grid::Vector{Float64}, x::Vector{Float64})
   
   @unpack nz, na, J, J_r  = prim

    # Get parameters
    β = x[1]
    σ = x[2]

    Kg = zeros(nz, na, J)
    cpol = zeros(nz, na, J)
    d1 = zeros(nz, J)

    for j in reverse(1:J-1)

        for (z,_) in enumerate(z_grid)

            d1[z,j] = (1-θ)*e[z,j]*wage + b[j] + T_g
            d2 = (1-θ)*e[:,j+1]*wage .+ b[j+1] .+ T_g

            pol_a = zeros(na)
            for (q,a) in enumerate(a_grid)

                pol_a[q] =(( β*(1+r_g)*s_j[j] * trans_mat[z,:]'*((1+r_g)*a .+ d2 - Kg[:,q,j+1]).^(-σ))^(-1/σ) + a - d1[z,j])/(1+r_g)

            end
                    
            nodes = (pol_a,) # Define the nodes
            itp = interpolate(nodes, a_grid, Gridded(Linear())) # Perform Interpolations
            etpf = extrapolate(itp, Line()) # Set up environment for extrapolation
            kg_temp = etpf(a_grid) # Perform extrapolation

            # Make sure boundaries are kept
            kg_temp[(kg_temp .< 0)] .= 0
            #kg_temp[(kg_temp .> a_grid[na])] .= a_grid[na]
            Kg[z,:,j] = kg_temp

        end

        cpol[:,:,j] = (1+r_g)*ones(nz,1).*a_grid' - Kg[:,:,j] + d1[:,j]*ones(1,na)
    end

   
    return Kg, d1, cpol
end

function GMM(prim, death_age::Vector{Float64}, c_sim::Matrix{Float64}, cap_cnstr::Matrix{Float64}, s_j::Vector{Float64}, guess::Vector{Float64}, W)

    @unpack len_panel, J  = prim
    # Get parameters
    β = guess[1]
    σ = guess[2]
    # Set storage
    # Subtrack one period because we're looking at consumption growth 
    m_bar_tmp = zeros(len_panel, J-1)
    G1_tmp    = zeros(len_panel, J-1)
    G2_tmp    = zeros(len_panel, J-1)

    # Discard more time periods because in simulation everybody starts off at zero consumption (Robert told me to do so)
    m_bar     = zeros(J-5-1)
    G         = zeros(J-5-1, 2) # because two moments
    VC_tmp   = zeros(J-5-1, J-5-1)

    for sim in 1:len_panel
        for age in 2:death_age[sim]
            # Exclude constraint agents in MM condition
            if cap_cnstr[Int.(sim), Int.(age)-1] == 1
                m_bar_tmp[sim, Int(age-1)] = 0.0
                G1_tmp[sim, Int(age)-1]    = 0.0
                G2_tmp[sim, Int(age)-1]    = 0.0

            else
                # If not constraint, EE
                m_bar_tmp[sim, Int(age)-1] = β*(1+r_g)*s_j[Int(age)]*(c_sim[sim, Int(age)]/c_sim[sim, Int(age)-1])^(-σ) - 1

                # take FOC with respect to parameters β and σ
                G1_tmp[sim, Int(age)-1]    = (1+r_g)*s_j[Int(age)]*(c_sim[sim, Int(age)]/c_sim[sim, Int(age)-1])^(-σ)
                G2_tmp[sim, Int(age)-1]    = β*(1+r_g)*s_j[Int(age)]*(c_sim[sim, Int(age)]/c_sim[sim, Int(age)-1])^(-σ)*(log(c_sim[sim, Int(age)]) - log(c_sim[sim, Int(age)-1]))
            end

        end

        VC_tmp[:,:] += m_bar_tmp[sim,5:end-1]*m_bar_tmp[sim,5:end-1]'

    end

    m_bar[:] = sum(m_bar_tmp[:,5:end-1], dims=1)[:]/len_panel
    G[:,1]   = sum(G1_tmp[:,5:end-1], dims=1)[:]/len_panel
    G[:,2]   = sum(G2_tmp[:,5:end-1], dims=1)[:]/len_panel

    error = m_bar'*W*m_bar

    Opt_weigh_matrix = VC_tmp./len_panel

    return G, Opt_weigh_matrix, error
end


function SMM()

    prim= Primitives_Q3()

    # Run model with true parameters 
    _, _, s_j, _, _, _, _, _, _, _, _, _, _, _, _, T_g_1, r_g, _, death_age, _, cap_cnstr, c_path = solve_model(prim, 0.05, 0.6)

    # Define guess of weighting matrix
    W = I
    # Initial guess of parameters
    guess = [0.89, 2]
    lb     = [0.85, 1] # first beta, second sigma
    ub     = [0.99, 3]
    GMM_estim(guess) = GMM(prim, death_age, c_path, cap_cnstr, s_j, guess, W)[3]
    opt_contr = Optim.Options(f_calls_limit = 100000, iterations = 100000, show_trace=true)
    res = optimize(GMM_estim, lb, ub, guess, Fminbox(LBFGS()), opt_contr)
    θ_hat = Optim.minimizer(res)
    G_hat, Ψ_hat = GMM(prim, death_age, c_path, cap_cnstr, s_j, guess, W)[1:2] # Results converge to lower bound, why?

    # Get standard errors
    asymVar = (1/prim.len_panel)*inv(G_hat'*G_hat)*G_hat'*Ψ_hat*G_hat*inv(G_hat'*G_hat)
    SEbeta = sqrt(asymVar[1,1])
    SEsigma = sqrt(asymVar[2,2])

    # Optimal Weighting Matrix - I don't find the bug here yet
    W_opt = inv(Ψ_hat)
    GMM_estim2(guess) = GMM(prim, death_age, c_path, cap_cnstr, s_j, guess, W_opt)[3]
    opt_contr_opt = Optim.Options(f_calls_limit = 100000, iterations = 100000, show_trace=true)
    res_opt = optimize(GMM_estim2, lb, ub, guess, Fminbox(LBFGS()), opt_contr_opt)
    θ_hat_opt = Optim.minimizer(res_opt)


    return θ_hat, θ_hat_opt
end



#### Pseudocode c) for question 3)
""" 
Modify function 'data_moments' by replacing 
    1. cpol with kpol
    2. Compute Average wealth by sum(young*kpol) over dimension 2 and 3
    3. Run same code but let converge on average wealth per age
end

"""