
function rdylgn(v)
    # c = v <= 0.5 ? (v > 1 ? 0xff0000 : (v * 2 * 0x0000ff) << 16) : (v < 0 ? 0 : ((1 - v) * 2 * 0x0000ff) << 8)
    c = 0xffffff
    if v < 0.5
        if v <= 0
            c = 0xff0000  # only red
        else
            c = (UInt(round(v * 2 * 0x0000ff)) << 8) + 0xff0000
        end
    else  # v >= 0.5
        if v >= 1
            c = 0x00ff00  # only green
        else
            c = (UInt(round((1 - v) * 2 * 0x0000ff)) << 16) + 0x00ff00
        end
    end
    c += 150
end

function palette(N::Int=100)
    N -= 1
    c = [rdylgn(ix/N) for ix in 0:N]
end

function discrete_background_color_bins(df; n_bins=5, columns="all")
    if n_bins < 1
        return Dict[]
    end

    bounds = [(i-1) * (1.0 / n_bins) for i in 1:n_bins+1]
    nme = names(df, Number)
    if columns == "all"
        df_numeric_columns = df[!,nme]
    else
        df_numeric_columns = df[!,[columns]]
    end
    if (size(df_numeric_columns, 1) == 0) || (size(df_numeric_columns, 2) == 0)
        return Dict[]
    end

    styles = Dict[]
    ps = ["#" * string(c, base=16, pad=6) for c in palette(n_bins)]

    vals = Float64[]
    for column in names(df_numeric_columns, Number)
        for v in df_numeric_columns[!, column]
            ismissing(v) && continue
            push!(vals, (v))
        end
    end
    isempty(vals) && return Dict[]

    df_max = maximum(vals)
    df_min = minimum(vals)
    df_max = maximum([df_max, abs(df_min)])
    df_min = -df_max
    ranges = [((df_max - df_min) * i) + df_min for i in bounds]

    # for (ix, c) in enumerate(ps)
    #     println("$ix $c")
    # end
    for column in names(df_numeric_columns, Number)
        for i in 1:length(bounds)-1
            min_bound = ranges[i]
            max_bound = ranges[i+1]
            backgroundColor = ps[i]
            # backgroundColor = string("#",lowercase(hex.(colormap("RdBU",n_bins))[i]))
            color = "black"  # i > (length(bounds) / 2.) ? "white" : "inherit"
            chk = i < (length(bounds) - 1) ? " && {$column} < $max_bound" : ""
            push!(styles, Dict(
               "if" => Dict(
                    "filter_query" => string("{$column} >= $min_bound", chk),
                    "column_id"=> column
                ),
                "backgroundColor" => backgroundColor,
                "color"=> color
            ))
        end
    end

    return styles
end

