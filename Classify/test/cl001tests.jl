
module ClassifyTest
using CategoricalArrays
using CategoricalDistributions, StatisticalMeasures
import CategoricalDistributions:classes
using Dates, DataFrames
using Test
using EnvConfig, Classify

EnvConfig.init(training)

@testset "Classifier001 tests" begin
cls = Classify.Classifier001Set()
cid = Classify.clsid(cls, STDREGRWINDOW, STDGAINTHRSHLD, 0, 10*24*60)
@test Classify.clcfg.clid == cid

end

end  # module