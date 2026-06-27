"""
    trend_runtime_load_phase(mode) -> String

Map EnvConfig mode to the artifact phase used by runtime trend classifiers.
`test` maps to `test`; all other modes map to `training`.
"""
function trend_runtime_load_phase(mode)::String
    key = lowercase(String(Symbol(mode)))
    if key == "test"
        return "test"
    end
    # In production mode runtime inference must use training artifacts.
    return "training"
end

"""
    trend_runtime_folder_from_spec(spec, mode) -> String

Resolve the artifact folder for runtime trend classifier loading. A literal
`folder` in spec has priority, otherwise it is derived from config reference.
"""
function trend_runtime_folder_from_spec(spec::NamedTuple, mode)::String
    if hasproperty(spec, :folder)
        return String(getproperty(spec, :folder))
    end
    if hasproperty(spec, :config_ref)
        return "Trend-$(String(getproperty(spec, :config_ref)))-$(trend_runtime_load_phase(mode))"
    end
    if hasproperty(spec, :configname)
        return "Trend-$(String(getproperty(spec, :configname)))-$(trend_runtime_load_phase(mode))"
    end
    error("missing classifier folder information in spec; expected folder or config_ref/configname")
end

"""
    runtime_loadclassifier(build_classifier, nn_fileprefix, build_args...; search_folders)

Search `search_folders` for `nn_fileprefix`, load the NN, and build a runtime
classifier via `build_classifier(nn, build_args...)`.
"""
function runtime_loadclassifier(
    build_classifier::Function,
    nn_fileprefix::AbstractString,
    build_args...;
    search_folders::AbstractVector{<:AbstractString},
)
    for folder in unique(String.(search_folders))
        EnvConfig.setlogpath(folder)
        nnpath = nnfilename(nn_fileprefix)
        if isfile(nnpath)
            try
                nn = loadnn(nn_fileprefix)
                return build_classifier(nn, build_args...)
            catch err
                shorterr = sprint(showerror, err)
                error("classifier file found but could not be loaded: nnpath=$nnpath. Cause=$shorterr. Likely artifact compatibility mismatch (Flux/Optimisers/BSON versions).")
            end
        end
    end
    error("classifier file not found for fileprefix=$nn_fileprefix, checked folders=$(collect(search_folders))")
end