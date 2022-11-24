# This is a replication attempt at Hugget 1996 JME
# Problem Set 1, Life Cycle Course
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

Random.seed!(1234)
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
        z_grid[i] = exp(z_logs[i])# - 0.5*(σ_e^2/(1-ρ^2)))
    end

    return a_grid, trans_mat, z_grid
end

# Compute Invariant Distribution
function invariant_prod_dist(prim::Primitives, Phi::Matrix{Float64}, z_grid::Vector{Float64})
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


function firm(prim::Primitives, Agg_L::Float64, r_g::Float64)

    @unpack α, δ, A = prim
    # Aggregate Capital from FOC of firm
    agg_k = ( (r_g + δ)/((α*A)*(Agg_L)^(1-α)) )^(1/(α-1))
    # Wage from other FOC
    wage = (1-α)*A*(agg_k)^(α)* (Agg_L)^(-α)

    return agg_k, wage
end


function government(prim::Primitives, Ψ::Vector{Float64}, Wage::Float64, Agg_L::Float64)
    @unpack ω, J, J_r = prim

    θ = ω*sum(Ψ[J_r:J])/sum(Ψ[1:(J_r-1)])

    b_t = θ*Wage*Agg_L/(sum(Ψ[prim.J_r:prim.J]))
    b = zeros(J)
    b[J_r:J] = ones(J - J_r+1)*b_t

    return θ, b
end


function egm(prim::Primitives, r_g::Float64, T_g::Float64, wage::Float64, b::Array{Float64,1}, e::Matrix{Float64}, θ::Float64, s_j::Vector{Float64},
     trans_mat::Matrix{Float64}, z_grid::Vector{Float64}, a_grid::Vector{Float64})
    
    @unpack nz, na, J, J_r, β, σ  = prim

    Kg = zeros(nz, na, J)
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


    end

    return Kg, d1
end


function simulation_shocks(prim::Primitives, trans_mat::Matrix{Float64}, n_sim::Int64)

    @unpack J, nz = prim
    z_path_index = ones(n_sim, J) # storage with first element set

    for sim in 1:n_sim

        for age in 2:J-1
            z_path_index[sim, age] = sample(1:nz, Weights(trans_mat[Int.(z_path_index[sim, age-1]),:]))
        end
    end
    
    return z_path_index
end


function capital_simulation(prim::Primitives, z_path_index::Matrix{Float64}, Kg::Array{Float64, 3}, n_sim::Int64)

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




function young_2010(prim::Primitives, a_grid::Vector{Float64}, Kg::Array{Float64, 3}, trans_mat::Matrix{Float64}, Φ::Vector{Float64})

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

    return dist_fin
end


function agg_capital_hh(prim::Primitives, dist_fin::Array{Float64, 3}, r_g::Float64, Kg::Array{Float64, 3}, Ψ::Vector{Float64}, s_j::Vector{Float64},
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

function solve_model(r_g::Float64, T_g::Float64)

    # Step 1-6
    prim = Primitives()
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
            Kg, d1 = egm(prim, r_g, T_g, Wage, b, hh_prod, θ, s_j, trans_mat, z_grid, a_grid)
            
            # Calculate distribution with new policy function
            dist_fin = young_2010(prim, a_grid, Kg, trans_mat, Φ)
            # Get aggregate capital supply by HH and aggregate accidental bequests
            Agg_K_hh, T_g_1, Y = agg_capital_hh(prim, dist_fin, r_g, Kg, Ψ, s_j, Agg_L)
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

    return a_grid, Φ, s_j, Ψ, P, hh_prod, d1, Agg_L, Agg_K, Wage, θ, b, Kg, dist_fin, Agg_K_hh, T_g_1, r_g, Y
end

# Run the economy
a_grid, Φ, s_j, Ψ, P, hh_prod, d1, Agg_L, Agg_K, Wage, θ, b, Kg, dist_fin, Agg_K_hh, T_g_1, r_g, Y = solve_model(0.02, 1.2)

## Calculate and Plot Results
gr()
savings = Kg[:,:,30] .- a_grid'
sav_fig30 = plot(a_grid, savings', label = [L"z_1" L"z_2" L"z_3" L"z_4" L"z_5"], dpi=300, title = "Savings Decision for 30")
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

dist_graph_20 = plot(a_grid, dist_fin[:,:,20]', label = [L"z_1" L"z_2" L"z_3" L"z_4" L"z_5"], dpi=300, title = "Distribution at Age 20")
xaxis!("Capital Grid")
yaxis!("Density at age 20")
savefig(dist_graph_20,"dist_20.png")

plot(a_grid, dist_fin[:,:,60]', label = [L"z_1" L"z_2" L"z_3" L"z_4" L"z_5"], dpi=300, title = "Distribution at Age 60")

# Getting the Histogram of capital
hist = zeros(prim.na)
for i in 1:prim.na
    hist[i] = sum(dist_fin[:,i,:])/sum(dist_fin[:,:,:])
end

hist_cap = bar(a_grid[1:50], hist[1:50], color = :darkgreen)
xaxis!("Capital grid")
yaxis!("Probability")
title!("Histogram of Assets")
savefig(hist_cap,"histcap.png")

###### Calculate cross sectional distribution in absolute values

mean_j = zeros(prim.J)
cumsum_j = zeros(prim.na)
median_j = zeros(prim.J)
quant_25 = zeros(prim.J)
quant_75 = zeros(prim.J)

for j in 1:prim.J-1

    mean_j[j] = sum(dist_fin[:,:,j].*Kg[:,:,j])
    cumsum_j = cumsum(vec(sum(dist_fin[:,:,j], dims=1)))
    median_j[j] = a_grid[findfirst(x -> x >= 0.5, cumsum_j)]
    quant_25[j] = a_grid[findfirst(x -> x >= 0.25, cumsum_j)]
    quant_75[j] = a_grid[findfirst(x -> x >= 0.75, cumsum_j)]

end

medianplot = plot(1:prim.J, [mean_j median_j quant_25 quant_75], dpi=300, label = ["Mean" "Median" "25 Percentile" "75 Percentile"])
xaxis!("Age")
yaxis!("Assets")
vline!([41], label= "Retirement", color="black", line=(:dot,2))
title!("Distributional Aspects of Assets across Age")
savefig(medianplot,"distribution.png")

##### Calculate Gini Coefficient
Kg_gini = sum(Kg, dims=[1,3])
dist_fin_gini = sum(dist_fin, dims=[1,3])./sum(dist_fin)
long_pol = vec(Kg_gini)
long_dist = vec(dist_fin_gini)
argsort = sortperm(long_pol)
long_pol = sort(long_pol)
long_dist = long_dist[argsort]

cumulative=zeros(na+1)
for i=1:na
    cumulative[i+1] = cumulative[i] + long_pol[i]*long_dist[i]
end

temp=zeros(na,1)
for i=1:na
    temp[i]=(cumulative[i+1]+cumulative[i])*long_dist[i]
end
gini = 1-sum(temp)/cumulative[na+1]


function Gini_2(Kg, dist_fin, j, na=100)

    long_pol = vec(reshape(Kg[:,:, j], (1, 500)))
    long_dist = vec(reshape(dist_fin[:,:, j], (1, 500)))
    argsort = sortperm(long_pol)
    long_pol = sort(long_pol)
    long_dist = long_dist[argsort]

    cumulative=zeros(na+1)
    for i=1:na
        cumulative[i+1] = cumulative[i] + long_pol[i]*long_dist[i]
    end

    temp=zeros(na,1)
    for i=1:na
        temp[i]=(cumulative[i+1]+cumulative[i])*long_dist[i]
    end
    gini = 1-sum(temp)/cumulative[na+1]

    return gini
end

gini_j = zeros(prim.J)

for j in 1:prim.J
    gini_j[j] = Gini_2(Kg, dist_fin, j)
end

age_gini = plot(1:prim.J, gini_j, dpi=300, legend = false)
xaxis!("Age")
yaxis!("Gini Index")
title!("Gini Index across the Ages")
vline!([41], label= "Retirement", color="black", line=(:dot,2))
savefig(age_gini,"age_gini.png")

##### Add Euler Equation error plot

function calc_EEE(prim::Primitives, na_precise::Int64, Kg::Array{Float64,3}, r_g::Float64, T_g::Float64, 
    θ::Float64, hh_prod::Matrix{Float64}, trans_mat::Matrix{Float64}, s_j::Vector{Float64}, Wage::Float64, b::Vector{Float64})
    
    @unpack minval, maxval, nz, na, J, J_r, β, σ, = prim
    
    # make fine grid
    f_grid = zeros(na_precise)
    f_grid = collect(range(minval, maxval, length=na_precise))

    # interpolate g_pol at nkk_precise points 
    EE_error  = zeros(nz,na_precise,J)

    for j = 1:J-1

        g_prec = zeros(nz,na_precise)
        for z = 1:nz

            # interpolate on fine grid
            nodes   = (a_grid,)
            itp     = interpolate(nodes, Kg[z,:,j], Gridded(Linear()))
            extrp  = extrapolate(itp,Line())
            g_prec[z,:] = extrp(f_grid)

            g_p_prec    = zeros(nz,na_precise)
            for z_p = 1:nz
                # interpolate the future decision
                nodes_g  = (a_grid,)
                itp_g    = interpolate(nodes_g, Kg[z_p,:,j+1], Gridded(Linear()))
                extrp_g  = extrapolate(itp_g,Line())
                g_p_prec[z_p,:]=extrp_g(g_prec[z,:])
            end

            # non financial income
            d_0 = (1-θ)*hh_prod[z,j]  *Wage + b[j] + T_g
            d_1 = (1-θ)*hh_prod[:,j+1]*Wage .+ b[j+1] .+ T_g
            for a = 1:na_precise
                # consumption today
                c  = (f_grid[a]*(1+r_g) + d_0 - g_prec[z,a])
                #consumption tomorrow
                Ec_p = β*(1+r_g)*s_j[j]*trans_mat[z,:]'*((g_prec[z,a]*(1+r_g)*ones(nz) + d_1 - g_p_prec[:,a]).^(-σ))
                # use formula to calc realtive EEE in terms of consumption 
                EE_error[z,a,j] = log(abs(c.^(-σ) - Ec_p)/c)
            end
        end
    end
    return EE_error, f_grid
end

EE_error, f_grid = calc_EEE(prim, 10000, Kg, r_g, T_g, θ, hh_prod, trans_mat, s_j, Wage, b)

# Plots with partial equilibrium and starting guesses
plot_pe_20 = plot(f_grid, EE_error[:,:,20]', label = [L"z_1" L"z_2" L"z_3" L"z_4" L"z_5"], dpi=300)
xaxis!("Fine capital grid")
yaxis!("Euler Equation Error")
title!("Euler Equation Error PE at age 20")
savefig(plot_pe_20,"pe_20.png")

plot_pe_60 = plot(f_grid, EE_error[:,:,60]', label = [L"z_1" L"z_2" L"z_3" L"z_4" L"z_5"], dpi=300)
xaxis!("Fine capital grid")
yaxis!("Euler Equation Error")
title!("Euler Equation Error PE at age 60")
savefig(plot_pe_60,"pe_60.png")

# Plots with general equilibrium results
plot_ge_20 = plot(f_grid, EE_error[:,:,20]', label = [L"z_1" L"z_2" L"z_3" L"z_4" L"z_5"], dpi=300)
xaxis!("Fine capital grid")
yaxis!("Euler Equation Error")
title!("Euler Equation Error GE at age 20")
savefig(plot_ge_20,"ge_20.png")

plot_ge_60 = plot(f_grid, EE_error[:,:,60]', label = [L"z_1" L"z_2" L"z_3" L"z_4" L"z_5"], dpi=300)
xaxis!("Fine capital grid")
yaxis!("Euler Equation Error")
title!("Euler Equation Error GE at age 60")
savefig(plot_ge_60,"ge_60.png")



# Plot some more results
plot_income = plot(1:prim.J-1, d1[:,1:70]', label = [L"z_1" L"z_2" L"z_3" L"z_4" L"z_5"], dpi=300, title = "Income")
xaxis!("Age")
yaxis!("Income")
vline!([41], label= "Retirement", color="black", line=(:dot,2))
savefig(plot_income,"income.png")

d1_gini = zeros(prim.J)

for j in 1:prim.J 

    d1_red = d1[:,j]
    prod_dist = Φ
    long_pol = vec(d1_red)
    long_dist = vec(prod_dist)
    argsort = sortperm(long_pol)
    long_pol = sort(long_pol)
    long_dist = long_dist[argsort]

    cumulative=zeros(length(d1_red)+1)
    for i in 1:length(d1_red)
        cumulative[i+1] = cumulative[i] + long_pol[i]*long_dist[i]
    end

    temp=zeros(length(d1_red),1)
    for i in 1:length(d1_red)
        temp[i]=(cumulative[i+1]+cumulative[i])*long_dist[i]
    end
    d1_gini[j] = 1-sum(temp)/cumulative[length(d1_red)+1]
end

plot_income_gini = plot(1:prim.J-1, d1_gini[1:70], legend=false, dpi=300, title = "Income Inequality")
xaxis!("Age")
yaxis!("Gini Coefficient")
vline!([41], label= "Retirement", color="black", line=(:dot,2))
savefig(plot_income_gini,"income_gini.png")


###### Consumption
cpol_j = zeros(prim.nz,prim.na,prim.J)
mean_c_j = zeros(prim.J)
for j in 1:prim.J 
    cpol_j[:,:,j] = (1+r_g)*ones(prim.nz,1).*a_grid' - Kg[:,:,j] + (1-θ)*hh_prod[:,j]*Wage*ones(1,prim.na) .+ T_g .+ b[j]
    mean_c_j[j] = sum(dist_fin[:,:,j].*cpol_j[:,:,j])
    #bpol_j[:,:,j] = 
end

gini_cons_j = zeros(prim.J)

for j in 1:prim.J
    gini_cons_j[j] = Gini_2(cpol_j, dist_fin, j)
end

cons_plot = plot(1:prim.J-1, mean_c_j[1:70], dpi=300, title="Average Consumption")
xaxis!("Age")
yaxis!("Average Coefficient")
vline!([41], label= "Retirement", color="black", line=(:dot,2))
savefig(cons_plot, "cons_plot.png")

cons_gini = plot(1:prim.J, gini_cons_j, dpi=300, title="Consumption Inequality")
xaxis!("Age")
yaxis!("Gini Coefficient")
vline!([41], label= "Retirement", color="black", line=(:dot,2))
savefig(cons_gini,"cons_gini.png")
