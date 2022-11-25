# This is a replication attempt at the Aiyagari Model with VFI and EGM
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

@with_kw struct Primitive
    
    minval::Float64 = 0.0
    maxval::Float64 = 200.0
    na::Int64 = 200
    Z::Float64 = 1.0
    β::Float64 = 0.96
    σ::Float64 = 2.0
    α::Float64 = 0.33
    δ::Float64 = 0.05
    nz::Int64 = 2
    z_grid::Vector{Float64} = [0.1, 1.0]
    trans_mat::Matrix{Float64} = [0.1 0.9; 0.1 0.9]
    A::Float64 = 1.0
    a_grid::Vector{Float64} = collect(range(minval, maxval, length=na))
end

function agg_labour(prim::Primitive)
    
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

    return Phi_sd, L_Agg
end

function firm_decision(prim::Primitive, L::Vector{Float64}, r::Float64)
    
    @unpack α, A, δ, β = prim

    k_firm = (α*(1+r)*A / (1-(1+r)+δ*(1+r)))^(1/(1-α))*L
    wage = (1-α)*A^(1/(1-α)) * (α*(1+r)/(1-(1+r)+δ*(1+r)))^(α/(1-α))
    
   return k_firm, wage
end

# Generate a Assets Grid
function asset_grid(prim::Primitive, wage::Float64, r::Float64, b::Int64=0, a_max::Int64 = 30)
    
    # Find minimum
    a_min = -1*min(b, wage*0.25/(1-(1+r)))
    
    # Set up grid
    a_grid = collect(LinRange(a_min, a_max, prim.na))
    
    return a_grid
end

# Consumption function

function fcons(prim::Primitive, wage::Float64, a::Float64, r::Float64, a_grid::Vector{Float64})

    a_grid_dim = repeat(a_grid, 1, prim.nz)
    return wage*prim.z_grid .+ (1+r)*a .- a_grid_dim'
end


function util_(prim::Primitive, c::Matrix{Float64})
    return (c.^(1-prim.σ))./(1-prim.σ)
end


function VFI(prim::Primitive, wage::Float64, r::Float64)
    
    @unpack trans_mat, β, σ, na, nz, z_grid, a_grid = prim

    v_now = zeros(nz, na)
    v_next = zeros(nz, na)
    policy = zeros(nz, na)
    error = 1
    tol = 10e-6
    util = 0
    c = 0
    
    while error > tol
        
        for a in 1:na
            
            c = fcons(prim, wage, a_grid[a], r, a_grid) # calculate consumption
            util = util_(prim, c) # calculate util
            util[findall(x-> x<=0, c)] .= -1000000 # replace unreasonable combinations
            
            v_temp = util + β.* trans_mat * v_next # value function
            v_now[:,a] = maximum(v_temp, dims=2) # find max
            
            for z in 1:nz
                policy[z, a] = findmax(v_temp, dims=2)[2][z][2] # policy decision
            end
        
        end
        error = maximum(abs.(v_now - v_next))
        v_next = copy(v_now)
    end
    # set up capital-state x capita-state matrix that incorporates the prob of being in a certain capital-state point
    # tomorrow based on today's decision
    return policy, v_now
end


function EGM(prim::Primitive, wage::Float64, r::Float64)

    @unpack nz, na, β, σ, z_grid, trans_mat, a_grid  = prim
    kpol_egm = zeros(nz,na)
    cpol = ones(nz,na)
    tol_pol = 0.00001
    err  = 1
    #cpol = (z_grid'*wage .+ r*a_grid)
    while err > tol_pol

        Ec = trans_mat * (cpol).^(-σ)
        
        c_impl = ((1+r)*β*Ec).^(-1/σ)
        
        k_impl =  (c_impl + ones(nz,1).*a_grid' - z_grid*wage*ones(1,na))./(1+r)
                    
        for (z,_) in enumerate(z_grid)
            nodes = (vec(k_impl[z,:]),) # Define the nodes
            itp = interpolate(nodes, a_grid, Gridded(Linear())) # Perform Interpolations
            etpf = extrapolate(itp, Line()) # Set up environment for extrapolation
            kpol_egm[z,:] = etpf(a_grid) # Perform extrapolation
        end

        # Make sure boundaries are kept
        kpol_egm[(kpol_egm .< 0)] .= 0
        kpol_egm[(kpol_egm .> a_grid[na])] .= a_grid[na]


        cpol1 = (1+r)*ones(nz,1).*a_grid' - kpol_egm[:,:] + z_grid*wage*ones(1,na) 
        err = maximum(abs.(cpol-cpol1))
        cpol = copy(cpol1)

    end

    return kpol_egm, cpol
end


function calc_EEE(prim::Primitive, na_precise::Int64, Kg::Matrix{Float64}, r_g::Float64, Wage::Float64)
    
    @unpack minval, maxval, nz, na, β, σ, trans_mat, a_grid, z_grid = prim
    
    # make fine grid
    f_grid = zeros(na_precise)
    f_grid = collect(range(minval, maxval, length=na_precise))

    # interpolate g_pol at nkk_precise points 
    EE_error  = zeros(nz,na_precise)


    g_prec = zeros(nz,na_precise)
    for z = 1:nz

        # interpolate on fine grid
        nodes   = (a_grid,)
        itp     = interpolate(nodes, Kg[z,:], Gridded(Linear()))
        extrp  = extrapolate(itp,Line())
        g_prec[z,:] = extrp(f_grid)

        g_p_prec    = zeros(nz,na_precise)
        for z_p = 1:nz
            # interpolate the future decision
            nodes_g  = (a_grid,)
            itp_g    = interpolate(nodes_g, Kg[z_p,:], Gridded(Linear()))
            extrp_g  = extrapolate(itp_g,Line())
            g_p_prec[z_p,:]=extrp_g(g_prec[z,:])
        end

        for a = 1:na_precise
            # consumption today
            c  = (f_grid[a]*(1+r_g) + z*Wage - g_prec[z,a]).^(-σ)
            #consumption tomorrow
            Ec_p = β*(1+r_g)*trans_mat[z,:]'*((g_prec[z,a]*(1+r_g)*ones(nz) + z_grid*Wage - g_p_prec[:,a]).^(-σ))
            # use formula to calc realtive EEE in terms of consumption 
            EE_error[z,a] = log(abs( c/Ec_p -1))
        end
    end
    
    return EE_error, f_grid
end

function young_2010(prim::Primitive, kpol::Array{Float64, 3})

    @unpack nz, na, trans_mat, a_grid = prim

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

    for z in 1:nz
        for i in 1:na
            wabove[z,i] =  (kpol[z,i] - a_grid[Int(ind_low[z,i])]) / (a_grid[Int(ind_low[z,i]) + 1] - a_grid[Int(ind_low[z,i])])
            wabove[z,i] = min(wabove[z,i],1)
            wabove[z,i] = max(wabove[z,i],0)
            wbelow[z,i] = 1-wabove[z,i]
        end
    end 

    Γ = zeros(nz*na, nz*na)


    for z in 1:nz
        for i in 1:na
            Γ[Int.((i-1)*nz+z), Int.((ind_low[z,i]-1)*nz+1:ind_low[z,i]*nz)] = wbelow[z,i]*trans_mat[z,:]
            Γ[Int.((i-1)*nz+z), Int.((ind_up[z,i]-1)*nz+1:ind_up[z,i]*nz)] = wabove[z,i]*trans_mat[z,:]
        end
    end

    probst = (1/(nz*na))*ones(1,nz*na)
    err = 1                      
    while err > 1e-5      
       probst1 = probst*Γ          
       err = maximum(abs.(probst1-probst))
       probst = probst1
    end


    dist_fin = reshape(probst, (nz, na))

    return dist_fin
end

function bisection(prim::Primitive, VFI::Bool)
    
    @unpack α = prim

    while (err > 1e-5) && (iter<100)
    
        # Guess
        q_now = (q_min + q_max)/2
        
        # Value function iteration
        k_now, wages_2 = firm_decision(alpha, A, delta, q_now, L_agg)

        if VFI == 1    
            a_now, policy_a_2, pol_a_2, statio_2, v0_2, a_grid_2 = VFI(wages_2, q_now, s, sigma, Phi, beta, 100)
        else
            kpol, cpol = EGM(wages_2, q_now, s, sigma, Phi, beta, 100)
        end

        q[1,iter] = q_now
        
        # Store temp. demand and supply
        K_agg_2[iter] = k_now
        a_mean_2[iter] = a_now
        
        # Idea: 
        # Solving FOC of firm w.r.t q where q = 1/(1+r) => condition that has to hold 
        q_2 = 1/(alpha*A*max(0.000001, a_now)^(alpha-1)-delta+1)
        q[2,iter] = q_2
        
        # error
        err = abs(q_2-q_now)
        # Supply > Demand
        if k_now > a_now
            q_max = copy(q_now)
        # Demand > Supply
        else
            q_min = copy(q_now)
        end
        iter = iter +1 
    end
    
    return
end


function solve_partial_model(VFI_bool::Bool)

    prim = Primitive()
    Phi_sd, L_Agg = agg_labour(prim)

    r = 0.03
    k_firm, wage = firm_decision(prim, L_Agg, r)

    #a_grid = asset_grid(prim, wage, r)
    if VFI_bool == true
        policy_a, v_now = VFI(prim, wage, r)
        cons_levels = prim.z_grid*wage .+ (1+r)*policy_a .- prim.a_grid'
        EE_error_vfi, f_grid_vfi = calc_EEE(prim, 10000, policy_a, r, wage)
        return policy_a, v_now, cons_levels, EE_error_vfi, f_grid_vfi
    else
        kpol, cpol = EGM(prim, wage, r)
        EE_error_egm, f_grid_egm = calc_EEE(prim, 10000, kpol, r, wage)
        return kpol, cpol, EE_error_egm, f_grid_egm 
    end
end

#### Value Function Iteration
policy_a, v_now, cons_levels, EE_error_vfi, f_grid_vfi = solve_model(true)


## Euler Equation Error


gr()
prim = Primitive()
value_vfi = plot(prim.a_grid, [v_now[1,:], v_now[2,:]], label = [L"z_1" L"z_2"], dpi=300, title = "Value Functions")
xaxis!(L"a")
yaxis!("Value Function")
savefig(value_vfi,"value_vfi.png")

cons_vfi = plot(prim.a_grid, [cons_levels[1,:], cons_levels[2,:]], label = [L"z_1" L"z_2"], dpi=300, title = "Consumption Functions")
xaxis!(L"a")
yaxis!("Consumption Function")
savefig(cons_vfi,"cons_vfi.png")

pol_vfi = plot(prim.a_grid, [policy_a[1,:], policy_a[2,:]], label = [L"z_1" L"z_2"], dpi=300, title = "Policy Functions")
xaxis!(L"a")
yaxis!("Policy Function")
savefig(pol_vfi,"pol_vfi.png")


#### Endogenous Grid Method
kpol, cpol, EE_error_egm, f_grid_egm  = solve_model(0.0)

cons_egm = plot(prim.a_grid, [cpol[1,:], cpol[2,:]], label = [L"z_1" L"z_2"], dpi=300, title = "Consumption Functions")
xaxis!(L"a")
yaxis!("Consumption Function")
savefig(cons_egm,"cons_egm.png")

pol_egm = plot(prim.a_grid, [kpol[1,:], kpol[2,:]], label = [L"z_1" L"z_2"], dpi=300, title = "Policy Functions")
xaxis!(L"a")
yaxis!("Policy Function")
savefig(pol_egm,"pol_egm.png")




