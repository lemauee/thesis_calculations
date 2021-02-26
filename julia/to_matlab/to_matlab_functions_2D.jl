using IncrementalInference, KernelDensityEstimate, DistributedFactorGraphs, MAT, CircStats

mutable struct KdeExport
    means::Array{Float64,2}
    bandwidths::Array{Float64,2}
end

mutable struct NormalExport
    mu::Vector{Float64}
    sigma::Matrix{Float64}
end

mutable struct VarExport
    label::String
    kde::KdeExport
    normal::NormalExport
    max::Vector{Float64}
end

mutable struct VarsExport
    traj::Vector{VarExport}
    lm::Vector{VarExport}
end

function VarsExport()
    VarsExport(Vector{VarExport}(undef,0),Vector{VarExport}(undef,0))
end

function kde2kdeExport(kde)
    # Converts the julia-specific kde object to a struct exportable to a mat file
    return KdeExport(getPoints(kde),getBW(kde));
end

function normal2normalExport(normal)
    NormalExport(normal.μ, normal.Σ)
end

# function getKDEfitPose2(p::BallTreeDensity)
#     pp = getPoints(p)
#     ppEuclid = pp[1:2,:]
#     ppCirc = pp[3,:]
#     normalEuclid = fit(MvNormal,ppEuclid)
#     μ = zeros(3)
#     μ[1:2] = normalEuclid.μ
#     μ[3] = cmean(ppCirc)
#     Σ = zeros(3,3)
#     Σ[1:2,1:2] = normalEuclid.Σ
#     Σ[3,3] = cstd(ppCirc)^2
#     MvNormal(μ,PDMat(Σ))
# end

function getKDEfitPose2(p::BallTreeDensity)
    pp = getPoints(p)
    n = size(pp, 2)
    μ = vec([mean(pp[1:2,:],dims=2); cmean(pp[3,:])])
    z = pp .- μ
    z[3,:] = wrapRad.(z[3,:])
    Σ = 1/n.*z*z'
    if !ishermitian(Σ)
        # workaround to ensure Σ is Hermitian
        @warn "non-hermitian covarince matrix fit: $(Σ)"
        Σ += Σ'
        Σ ./= 2
        @warn "non-hermitian covarince matrix corrected to: $(Σ)"
    end
    MvNormal(μ,PDMat(Σ))
end

function vars2varsExport(vars)
    varsExport = VarsExport()
    for kVar in 1:length(vars)
        global VarsExport
        var = vars[kVar]
        label = string(var.label)
        kde = kde2kdeExport(getKDE(var))
        if var isa DFGVariable{Point2}
            normal = normal2normalExport(getKDEfit(getKDE(var)))
        elseif var isa DFGVariable{Pose2}
            normal = normal2normalExport(getKDEfitPose2(getKDE(var)))
        end
        max = getKDEMax(getKDE(var))
        varExport = VarExport(label,kde,normal,max)
        if occursin("x",label)
            push!(varsExport.traj,varExport)
        elseif occursin("l",label)
            push!(varsExport.lm,varExport)
        else
            error("unknown variable type")
        end
    end
    return varsExport
end

function saveMat(filename,varname,var)
    file = matopen(filename, "w")
    write(file, varname, var)
    close(file)
end

function getAllVars(fg)
    symsString = string.(ls(fg))
    symsStringP = symsString[occursin.("x",symsString)]
    symsStringL = symsString[occursin.("l",symsString)]
    idsP = sort(parse.(Int,replace.(symsStringP,"x"=>"")))
    idsL = sort(parse.(Int,replace.(symsStringL,"l"=>"")))
    vars = Array{Any,1}(undef,length(symsString))
    for i in 1:length(idsP)
        vars[i] = getVariable(fg, Symbol("x", idsP[i]))
    end
    for i = 1:length(idsL)
        vars[i+length(idsP)] = getVariable(fg, Symbol("l", idsL[i]))
    end
    return vars
end
