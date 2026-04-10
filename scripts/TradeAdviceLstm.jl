include("TrendLstm.jl")
const TradeAdviceLstm = TrendLstm

if abspath(PROGRAM_FILE) == @__FILE__
    TrendLstm.main(ARGS)
end
