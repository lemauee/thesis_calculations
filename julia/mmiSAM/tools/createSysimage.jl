using PackageCompiler
pkgString = open(f->read(f, String), "include.jl")
pkgString = replace(pkgString,"using " => "")
pkgString = replace(pkgString,"\n" => "")
pkgString = replace(pkgString,"RoMEPlotting," => "") # fix because starting with it fails ...
pkgNames = split(pkgString,',')
pkgSyms = Symbol.(pkgNames)
create_sysimage(pkgSyms; sysimage_path="$(homedir())/.julia/sysimage_RoME.so")
