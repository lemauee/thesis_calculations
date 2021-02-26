# thesis_calculations

Examples to reproduce RoME/IIF Solver freezing up.

## How to run

Instantiate the environment described in julia/Project.toml and julia/Manifest.toml.
Take a look at the `multimodal_calculations.bash` and `outlier_calculations.bash` scripts. Feel free to vary the number of threads and processes used in there.
The scripts should be run from this repositorys root (otherwise relative file paths will not work I think).

## Output

Any output is written to a folder in the dataset directory. If it runs well I'd be happy to get those :)

## Further notes

This is the state of my code before any changes were made to it due to to RoME #418 . Feel free to change the julia/mmiSAM/evaluation/solve2DIncremental.jl to turn of any multithreading features or so.

Also feel free to play with the nullhypo, spreadNH or inflation parameter to get a better solution if it runs well. Thats what I intended to do before it freezed everytime.

Also some data in the outliers folder is not used by the bash-scripts yet because I could not get a plausible solution from it, feel free to try this and get something going with different parameters.

The --tukey parameter does not change anything, its just for logpath consistency with GTSAM ;) 
