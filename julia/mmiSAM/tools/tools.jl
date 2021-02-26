using Printf

function measurementsFoldername(trajKeys,lmKeys)
    global str = ""
    str *= "_traj"
    for trajKey = trajKeys
        global str *= "-" * trajKey
    end
    str *= "_lm"
    for lmKey = lmKeys
        global str *= "-" * lmKey
    end
    return str
end

function logpath(path, file, framework, trajKeys, lmKeys ,startPoseIdx,endPoseIdx,nullhypo,tukey,nKernels,spreadNH,inflation,suffix)
    return path * "/" * replace(file,".mat" => "") * "_startpose-" * lpad(startPoseIdx,5,'0') * "_endpose-" * lpad(endPoseIdx,5,'0') * "_" * framework * measurementsFoldername(trajKeys,lmKeys) * replace((@sprintf "_nh-%0.3f" nullhypo),"."=>"-")  * replace((@sprintf "_tukey-%0.3f" tukey),"."=>"-")* "_nKernels-" * lpad(nKernels,5,'0') * replace((@sprintf "_spreadNH-%0.3f" spreadNH),"."=>"-") * replace((@sprintf "_inflation-%0.3f" inflation),"."=>"-") * "_" * suffix
end

function plotSave(fg,tree,prefix)
    prefix = lpad(prefix,5,'0')

    # save
    saveDFG(fg, "$(getLogPath(fg))/$(prefix)_fg")
    saveTree(tree, "$(getLogPath(fg))/$(prefix)_tree.jld2")
    saveMat("$(getLogPath(fg))/$(prefix)_variables.mat","variables",vars2varsExport(getAllVars(fg)))

    # plot
    # poses
    # pl1 = plotSLAM2DPoses(fg)
    # draw(SVG("$(getLogPath(fg))/$(prefix)_poses.pdf", 20cm, 20cm), pl1)
    # poses and landmarks
    pl2 = plotSLAM2D(fg;drawContour=false)
    push!(pl2, Coord.cartesian(fixed=true),style(background_color=RGB(1,1,1)))
    @async draw(PNG("$(getLogPath(fg))/$(prefix)_poses_landms.png", 20cm, 20cm), pl2)
    # tree
    drawTree(tree, show=false, filepath="$(getLogPath(fg))/$(prefix)_bt.pdf")
end

function str2mat(Type,str)
    expr = Meta.parse(str)
    if isa(expr,Number)
        # scalar case
        return expr
    else
        return mat = Type.(expr.args)
    end
end
