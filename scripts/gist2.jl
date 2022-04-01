
# include("../src/Binance.jl")
using JSON, HTTP, DataFrames
using MyBinance

# println(@__FILE__)
# println(dirname(@__FILE__))
# println(dirname(dirname(@__FILE__)))
# println(pwd())

# ei = MyBinance.getExchangeInfo()
# open(pwd() * "/HTTP-log.json","a") do io
#     JSON.print(io, ei, 4)
# end
# println(pwd())
# println(length(ei["rateLimits"]))
# df = DataFrames.DataFrame(ei["rateLimits"][1])
# # println(names(ei["rateLimits"][1]))
# for entry in ei["rateLimits"]
#     push!(df, entry)
# end
# println(df)

# for ix in 1:400
#     res = HTTP.request("GET", "https://www.binance.com/api/v1/exchangeInfo")
#     if res.status != 200
#         println("done $ix: $(res.status)")
#         break
#     else
#         print("$ix, ")
#     end
# end

# df = DataFrame(base=String[], xch=Int32[], manual=Bool[])
df = DataFrame()
sa = ["abc", "def", "ghi"]
ia = [3, 4, 5]
ba = [true, false, true]

df[!, :base] = sa
df[!, :ia] = ia
df[!, :ba] = ba
println(df)

# df.base = sa
# df.ia = ia
# df.ba = ba

# println(df)

