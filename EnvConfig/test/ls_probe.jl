module EnvConfigLSProbe

using EnvConfig
import EnvConfig: checkfolders, setpairquote!

const PROBE_BASE_PATH = joinpath("alpha", "beta")
const PROBE_LOCAL_CHECK = checkfolders
const PROBE_LOCAL_MUTATOR = setpairquote!

end