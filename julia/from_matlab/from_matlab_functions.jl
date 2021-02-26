using MAT,PDMats,Distributions,RoME,IncrementalInference #Caesar

# Incremental

function loadMeasurementMat2D(filename,varname)
    file = matopen(filename)
    measurements = read(file, varname)
    close(file)
    return measurements
end

function getKeysInMeasurementMat2D(measurements)
    return keys(measurements["traj"]), keys(measurements["lm"])
end

function getNumberPosesInMeasurementMat2D(measurements)
    return Int(maximum(measurements["traj"]["jTf"]))
end

function parseSinglePoseWithMeasFromMeasurementMat2D!(fg, measurements, trajKeys, lmKeys, poseIdx, isStartPose, nullhypo, nKernels; graphinitTrajTf=true, graphinitLmTf=true, graphinitLmBearing=true, graphinitLmRanging=true, graphinitLmBearingAndRanging=true, factorThreadmodel=SingleThreaded, forceOdomInit=false)
    mTraj = measurements["traj"]
    nTraj = measurements["trajNoise"]

    if !isStartPose && haskey(mTraj,"tf") && "tf" in trajKeys
        mTTfTrans = mTraj["tf"]["translation"]
        mTTfRot = mTraj["tf"]["rotation"]
        mTiTf = mTraj["iTf"]
        mTjTf = mTraj["jTf"]
        mTTfIdx = findfirst(x -> x == poseIdx, mTjTf)
        if !isnothing(mTTfIdx)
            mTTfIdx = mTTfIdx[1]
            # @info "mTTfIdx is $(mTTfIdx)"
            # TODO: implement for non-gaussian case
            if isa(nTraj["tf"]["type"],String) # one noise for all meas
                nTrajTf = nTraj["tf"]
            else # individual noise for all meas
                nTrajTf = Dict("sigma"=>nTraj["tf"]["sigma"][mTTfIdx],"type"=>nTraj["tf"]["type"][mTTfIdx],"mu"=>nTraj["tf"]["mu"][mTTfIdx])
            end
            mDist = parseMeasurement(
                [mTTfTrans[mTTfIdx,1], mTTfTrans[mTTfIdx,2], mTTfRot[mTTfIdx,1]],
                nTrajTf)
            from_pose = Symbol("x", Int(mTiTf[mTTfIdx]))
            to_pose = Symbol("x", Int(mTjTf[mTTfIdx]))
            @info "adding traj tf measurement between $(from_pose) and $(to_pose)"
            # Make sure both variables are in the FG. Otherwise add them.
            addVarIfMissing!(fg, from_pose, Pose2, [:POSE], nKernels)
            addVarIfMissing!(fg, to_pose, Pose2, [:POSE], nKernels)
            rel_pose_factor = Pose2Pose2(mDist)
            addFactor!(fg, [from_pose, to_pose], rel_pose_factor, graphinit = graphinitTrajTf, threadmodel=factorThreadmodel)
            if forceOdomInit
                initManual!(fg, to_pose, [Symbol("x",Int(mTiTf[mTTfIdx]),"x",Int(mTjTf[mTTfIdx]),"f",1)])
                # doautoinit!(fg,to_pose)
            end
        end
    end



    mLm = measurements["lm"]
    nLm = measurements["lmNoise"]

    mLiPose = mLm["iPose"]
    mLjLm = mLm["jLm"]
    mLIdx = findall(mLiPose .== poseIdx) # only meas for current pose

    if !isnothing(mLIdx)

        mLIdx = map(x->x[1], mLIdx)

        if haskey(mLm,"tf") && "tf" in lmKeys
            # mLTfTrans = mLm["tf"]["translation"]
            # mLTfRot = mLm["tf"]["rotation"]
            mLTf = mLm["tf"]
            nLTf = nLm["tf"]
            @info "nullhypo is $(nullhypo)"
            for kMeas = mLIdx
                # mDist = parseMeasurement(
                #     [mLTfTrans[kMeas,1], mLTfTrans[kMeas,2], mLTfRot[kMeas,1]],
                #     nLm["tf"])
                mhFlag, idLm, pLm = parseJLm(mLjLm,kMeas)
                from_pose = Symbol("x", Int(mLiPose[kMeas]))
                to_lm = Symbol.(["l"], idLm)
                @info "adding lm tf measurement between $(from_pose) and $(to_lm)"
                # Make sure both variables are in the FG. Otherwise add them.
                addVarIfMissing!(fg, from_pose, Pose2, [:POSE], nKernels)
                addVarIfMissing!.([fg], to_lm, [Pose2], [[:LANDMARK]], nKernels)
                # rel_pose_factor = Pose2Pose2(mDist)
                rel_pose_factor = parseTf(mLTf,nLTf,kMeas)
                if !mhFlag
                    addFactor!(fg, vcat(from_pose,to_lm), rel_pose_factor, graphinit = graphinitLmTf, threadmodel = factorThreadmodel, nullhypo=nullhypo)
                else
                    addFactor!(fg, vcat(from_pose,to_lm), rel_pose_factor, multihypo = vcat(1,pLm), graphinit = graphinitLmTf, threadmodel = factorThreadmodel, nullhypo=nullhypo)
                end
            end
        end

        if haskey(mLm,"bearingAndRanging")  && "bearingAndRanging" in lmKeys
            mLBr = mLm["bearingAndRanging"]
            nLBr = nLm["bearingAndRanging"]
            mLiPose = mLm["iPose"]
            mLjLm = mLm["jLm"]
            for kMeas = mLIdx
                mhFlag, idLm, pLm = parseJLm(mLjLm,kMeas)
                from_pose = Symbol("x", Int(mLiPose[kMeas]))
                to_lm = Symbol.(["l"], idLm)
                @info "adding lm bearingAndRanging measurement between $(from_pose) and $(to_lm)"
                # Make sure both variables are in the FG. Otherwise add them.
                addVarIfMissing!(fg, from_pose, Pose2, [:POSE], nKernels)
                addVarIfMissing!.([fg], to_lm, [Point2], [[:LANDMARK]], nKernels)
                rel_pose_factor = parseBearingAndRanging(mLBr,nLBr,kMeas)
                if !mhFlag
                    addFactor!(fg, vcat(from_pose,to_lm), rel_pose_factor, graphinit = graphinitLmTf, threadmodel = factorThreadmodel)
                else
                    addFactor!(fg, vcat(from_pose,to_lm), rel_pose_factor, multihypo = vcat(1,pLm), graphinit = graphinitLmTf, threadmodel = factorThreadmodel)
                end
            end
        end
    end
end

# General

function addVarIfMissing!(fg,symbol,type,tags,nKernels)
    if (symbol in ls(fg)) == false
        @info "adding missing variable $(symbol) with N=$(nKernels)"
        addVariable!(fg, symbol, type, tags = tags, N=nKernels)
    end
end

function parseTf(mLTf,nLTf,kMeas)
    mLTfTrans = mLTf["translation"]
    mLTfRot = mLTf["rotation"]
    if haskey(mLTf,"p") # multimodal
        mLTfP = mLTf["p"]
        mDist = Vector{SamplableBelief}(undef,length(mLTfP[kMeas]))
        for kMode = 1:length(mLTfP[kMeas])
            global mDist[kMode] = parseMeasurement([mLTfTrans[kMeas][kMode,1], mLTfTrans[kMeas][kMode,2], mLTfRot[kMeas][kMode,1]],nLTf)
        end
        p = vec(mLTfP[kMeas])
        p ./= sum(p) # make sure its really a vaild probability fcn
        return rel_pose_factor = Mixture(Pose2Pose2,mDist,p)
    else # unimodal
        mDist = parseMeasurement(
            [mLTfTrans[kMeas,1], mLTfTrans[kMeas,2], mLTfRot[kMeas,1]],
            nLTf)
        return rel_pose_factor = Pose2Pose2(mDist)
    end
end

function parseBearingAndRanging(mLBr,nLBr,kMeas)
    if nLBr["type"] == "gaussian"
        mLBrBear = mLBr["bearing"]
        mLBrRange = mLBr["ranging"]
        # unimodal bearing is assumed
        mDistBear = Normal(mLBrBear[kMeas,1]+nLBr["mu"][1],
            sqrt(nLBr["sigma"][1][1]))
        if isa(mLBrRange,Dict) # multimodal range
            mLBrRangeP = mLBrRange["p"]
            if length(mLBrRangeP[kMeas]) == 1 # unimodal in disguise
                mDistRange = Normal(mLBrRange["value"][kMeas][1,1]+nLBr["mu"][2],
                sqrt(nLBr["sigma"][end][end]))
            else # really multimodal
                muSigmaRange = Vector{Tuple{Float64,Float64}}(undef,length(mLBrRangeP[kMeas]))
                for kMode = 1:length(mLBrRangeP[kMeas]) #TODO: generalize for other measurement noises
                    global muSigmaRange[kMode] = (mLBrRange["value"][kMeas][kMode,1]+nLBr["mu"][2], 
                        sqrt(nLBr["sigma"][end][end]))
                end
                pRange = vec(mLBrRangeP[kMeas])
                pRange ./= sum(pRange) # make sure its really a vaild probability fcn
                mDistRange = MixtureModel(Normal,muSigmaRange,pRange)
            end
        else # unimodal range
            mDistRange = Normal(mLBrRange[kMeas,1]+nLBr["mu"][2],
                sqrt(nLBr["sigma"][end][end]))
        end
        return rel_pose_factor = Pose2Point2BearingRange(mDistBear,mDistRange)
    elseif noiseDict["type"] == "gaussian_mixture"
        error("gaussian mixture not implemented yet") # TODO
    else
        error("unsupported noise type")
    end
end

function parseMeasurement(meas,noiseDict)
    if noiseDict["type"] == "gaussian"
        if meas isa Array || meas isa Vector
            factor = parseMvNormal(vec(meas)+vec(noiseDict["mu"]), noiseDict["sigma"])
        else
            factor = Normal(meas+noiseDict["mu"], sqrt(noiseDict["sigma"]))
        end
    elseif noiseDict["type"] == "gaussian_mixture"
        error("gaussian mixture not implemented yet") # TODO
    else
        error("unsupported noise type")
    end
end

function parseMvNormal(mu,sigma)
    return MvNormal(vec(mu),parseSigmaMvNormal(sigma))
end

function parseSigmaMvNormal(sigma)
    if size(sigma,1) == 1
        sigmaPDMat = PDiagMat(vec(sigma))
    else
        if !ishermitian(sigma)
            # workaround to ensure sigma is Hermitian
            @warn "non-hermitian covarince matrix fit: $(sigma)"
            sigma += sigma'
            sigma ./= 2
            @warn "non-hermitian covarince matrix corrected to: $(sigma)"
        end
        sigmaPDMat = PDMat(sigma)
    end
    return sigmaPDMat
end

function parseJLm(jLm,idx)
    if isa(jLm,Array)
        return (false, [Int(jLm[idx])],[1])
    elseif isa(jLm,Dict)
        p = vec(jLm["p"][idx])
        if any(p.≈1) # single hypo in disguise
            return (false, vec(Int.(jLm["idx"][idx][p.≈1])),[1])
        else # real multihypo
            p ./= sum(p) # make sure its really a vaild probability fcn
            return (true, vec(Int.(jLm["idx"][idx])), p)
        end
    else
        @error "invalid landmark measurement"
    end
end
