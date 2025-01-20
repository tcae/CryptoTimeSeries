using Bybit, EnvConfig, Test, Dates, DataFrames

EnvConfig.init(production)
bc = Bybit.BybitCache()

function order()
    now = Bybit.get24h(bc, "BTCUSDT")
    println("$now")
    # oocreate = Bybit.createorder(bc, "BTCUSDT", "Buy", 0.00001, now.lastprice)
    oocreate = Bybit.createorder(bc, "BTCUSDT", "Buy", 0.00015, 98201 ) # now.lastprice)
    println("create order = $(isnothing(oocreate) ? string(oocreate) : oocreate)")

    oo = Bybit.order(bc, oocreate.orderid)
    println(oo)
end

function leverage()
    println(Bybit.HttpPrivateRequest(bc, "POST", "/v5/spot-margin-trade/set-leverage", Dict("leverage" => "2"), "alltransactions"))
end

function checkbaseames()
    df = Bybit.exchangeinfo(bc)
    len= 0
    sorted = true
    lastbase = first(df[!, :basecoin])
    for base in df[!, :basecoin]
        len = max(len, length(base))
        if lastbase > base
            sorted = false
        end
    end
    println("length=$len, sorted=$sorted, #coins=$(length(df[!, :basecoin]))")
end

checkbaseames()
# leverage()
# order()
# println("balances: $(Bybit.balances(bc))")
