#info
@info "using $(Threads.nthreads()) Threads"
@info "using $(Distributed.nprocs()) Processes"
#packages
@everywhere using Pkg
@everywhere Pkg.activate($(Pkg.project().path))
include("../tools/include.jl")
@info "packages added"

# include funtions to read from measurements matfile
include("../../from_matlab/from_matlab_functions.jl")
# include funtions save to variables/results matfile
include("../../to_matlab/to_matlab_functions_2D.jl")
# include funtions to plot etc.
include("../tools/tools.jl")

function parseCommandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--path"
            help = "path to dataset"
            arg_type = String
            required = true
        "--file"
            help = "filename of dataset 'xxx.mat'"
            arg_type = String
            required = true
        "--trajKeys"
            help = "trajectory measurements to use, colon separated 'translation:rotation'"
            arg_type = String
            required = true
        "--lmKeys"
            help = "lm measurements to use, colon separated 'bearing:ranging'"
            arg_type = String
            required = true
        "--startPoseIdx"
            help = "start pose idx"
            arg_type = Int
            default = 1
        "--endPoseIdx"
            help = "end pose idx, -1 for all"
            arg_type = Int
            required = true
        "--startPoseVal"
            help = "start pose value [x; y; theta], as julia array '[0;0;1.57]'"
            arg_type = String
            required = true
        "--plotSaveFinal"
            help = "plot final solution?"
            arg_type = Bool
            required = true
        "--plotSaveIter"
            help = "plot every iteration?"
            arg_type = Bool
            required = true
        "--nRuns"
            help = "how many times should a solution be obtained?"
            arg_type = Int
            default = 1
        "--suffix"
            help = "a descriptive suffix to differ between solutions"
            arg_type = String
            default = "solution"
        "--useMsgLikelihoods"
            help = "set useMsgLikelihoods of solver to?"
            arg_type = Bool
            default = false
        "--nullhypo"
            help = "set nullhypo of landmark factors to?"
            arg_type = Float64
            required = true
        "--tukey"
            help = "set tukey parameter of landmark factors to? (only for logpath generation, not used in julia)"
            arg_type = Float64
            required = true
        "--nKernels"
            help = "number of KDE kernels for every variable"
            arg_type = Int
            required = true
        "--spreadNH"
            help = "set spreadNH of solver to?"
            arg_type = Float64
            required = true
        "--inflation"
            help = "set inflation of solver to?"
            arg_type = Float64
            required = true
    end

    return parse_args(s)
end

parsedArgs = parseCommandline()
trajKeys = split(parsedArgs["trajKeys"],':')
lmKeys = split(parsedArgs["lmKeys"],':')
startPoseIdx = parsedArgs["startPoseIdx"]
endPoseIdx = parsedArgs["endPoseIdx"]
startPoseVal = str2mat(Float64,parsedArgs["startPoseVal"])
plotSaveFinal = parsedArgs["plotSaveFinal"]
plotSaveIter = parsedArgs["plotSaveIter"]
nRuns = parsedArgs["nRuns"]
nullhypo = parsedArgs["nullhypo"]
tukey = parsedArgs["tukey"]
nKernels = parsedArgs["nKernels"]
spreadNH = parsedArgs["spreadNH"]
inflation = parsedArgs["inflation"]

# load matfile
filename = parsedArgs["path"] * "/" * parsedArgs["file"]
varname = "measurements";
measurements = loadMeasurementMat2D(filename,varname)

# select where to log & write results
if parsedArgs["useMsgLikelihoods"]
    framework = "juliaML"
else
    framework = "julia"
end

nPoses = getNumberPosesInMeasurementMat2D(measurements)
if endPoseIdx == -1
    endPoseIdx = nPoses
end
dataLogpathBase = logpath(parsedArgs["path"], parsedArgs["file"], framework, trajKeys, lmKeys, startPoseIdx, endPoseIdx, nullhypo, tukey, nKernels,spreadNH,inflation,parsedArgs["suffix"])
tRun = Matrix{Float64}(undef,nRuns,endPoseIdx)

if !isdir(dataLogpathBase)
    for kRun = 1:nRuns
        @info "###### run $(kRun) ######"
        dataLogpath = "$(dataLogpathBase)/$(lpad(Int(kRun),5,'0'))"

        # Create initial factor graph with specified logging path.
        global fg = LightDFG{SolverParams}(solverParams=SolverParams(logpath=dataLogpath))
        getSolverParams(fg).useMsgLikelihoods = parsedArgs["useMsgLikelihoods"] # only works for Pose2Pose2
        getSolverParams(fg).N = nKernels
        getSolverParams(fg).spreadNH = spreadNH
        getSolverParams(fg).inflation = inflation

        ## solve
        # Add initial variable with a prior measurement to anchor the graph.
        tRun[kRun,1] = @elapsed begin
            global tree, smt, hist, fg
            @info "### iteration for pose $(startPoseIdx) ###"
            addVariable!(fg, Symbol('x',startPoseIdx), Pose2; N=nKernels)
            initial_pose = MvNormal(startPoseVal, Matrix(Diagonal([0.1;0.1;0.05].^2)))
            addFactor!(fg, [Symbol('x',startPoseIdx)], PriorPose2(initial_pose))
            parseSinglePoseWithMeasFromMeasurementMat2D!(fg, measurements, trajKeys, lmKeys, startPoseIdx, true, nullhypo, nKernels; factorThreadmodel=MultiThreaded)
            tree, smt, hist = solveTree!(fg; multithread=true)
        end
        if plotSaveIter
            plotSave(fg,tree,startPoseIdx);
        end

        # Solve incrementally from here ...
        for kPose = (startPoseIdx+1):endPoseIdx
            tRun[kRun,kPose] = @elapsed begin
                global tree, smt, hist, fg
                @info "### iteration for pose $(kPose) ###"
                parseSinglePoseWithMeasFromMeasurementMat2D!(fg, measurements, trajKeys, lmKeys, kPose, false, nullhypo, nKernels; factorThreadmodel=MultiThreaded, graphinitTrajTf=false, forceOdomInit=true)
                tree, smt, hist = solveTree!(fg, tree; multithread=true)
            end
            if plotSaveIter # & (kPose < getNumberPosesInMeasurementMat2D(measurements))
                plotSave(fg, tree, kPose)
            end
        end

        ## final plot
        if parsedArgs["plotSaveFinal"]
            plotSave(fg, tree, endPoseIdx)
        end
    end
    saveMat("$(dataLogpathBase)/tRun.mat","tRun",tRun)
    @info "### inference finished ###"
else
    @warn "julia solution already created"
end

## Run the garbage collector.
GC.gc()
