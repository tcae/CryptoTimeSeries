
# include("../EnvConfig/test/runtests.jl")
# include("../Ohlcv/test/runtests.jl")
# include("../Features/test/runtests.jl")
# include("../Targets/test/runtests.jl")
# include("../Classify/test/runtests.jl")
# include("../CryptoXch/test/runtests.jl")
# include("../Assets/test/runtests.jl")
# include("../Trade/test/trade_test.jl")
using Pkg

Pkg.test(["EnvConfig", "Ohlcv", "Features", "Targets", "Classify", "CryptoXch", "Assets"]; coverage=true)