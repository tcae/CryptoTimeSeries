using Dash

using CSV, DataFrames, Colors

wide_data = [
    Dict("Firm" =>  "Acme", "2017" =>  13, "2018" =>  5, "2019" =>  10, "2020" =>  4),
    Dict("Firm" =>  "Olive", "2017" =>  3, "2018" =>  3, "2019" =>  13, "2020" =>  3),
    Dict("Firm" =>  "Barnwood", "2017" =>  6, "2018" =>  7, "2019" =>  3, "2020" =>  6),
    Dict("Firm" =>  "Henrietta", "2017" =>  -3, "2018" =>  -10, "2019" =>  -5, "2020" =>  -6),
]

df = vcat(DataFrame.(wide_data)...)

app = dash()

colcolor(col, val) = val / (maximum(col) - minimum(col))
# palette(N::Int=100) = diverging_palette(
#     360,
#     360,
#     N;
#     mid=0.5,
#     c=0.88,
#     s=0.6,
#     b=0.75,
#     w=0.15,
#     d1=1.0,
#     d2=1.0,
#     wcolor=RGB(1,1,0),
#     dcolor1=RGB(1,0,0),
#     dcolor2=RGB(0,1,0),
#     logscale=false)

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
    bounds = [(i-1) * (1.0 / n_bins) for i in 1:n_bins+1]
    nme = names(df, Number)
    if columns == "all"
        df_numeric_columns = df[!,nme]
    else
        df_numeric_columns = df[!,[columns]]
    end
    df_max = maximum(Array(df_numeric_columns))
    df_min = minimum(Array(df_numeric_columns))
    ranges = [
        ((df_max - df_min) * i) + df_min
        for i in bounds
    ]
    styles = Dict[]
    legend = Component[]
    ps = ["#" * string(c, base=16, pad=6) for c in palette(n_bins)]
    # for (ix, c) in enumerate(ps)
    #     println("$ix $c")
    # end
    for i in 1:length(bounds)-1
        min_bound = ranges[i]
        max_bound = ranges[i+1]
        backgroundColor = ps[i]
        # backgroundColor = string("#",lowercase(hex.(colormap("RdBU",n_bins))[i]))
        color = "black"  # i > (length(bounds) / 2.) ? "white" : "inherit"
        for column in names(df_numeric_columns, Number)
          chk = i < (length(bounds) - 1) ? " && {$column} < $max_bound" : ""
            push!(styles, Dict(
               "if" => Dict(
                    "filter_query" => string("{$column} >= $min_bound", chk),
                    "column_id"=> column
                ),
                "backgroundColor" => backgroundColor,
                "color"=> color
            )
        )
        end
        push!(legend,
            html_div(style=Dict("display"=> "inline-block", "width"=> "60px"), children=[
                html_div(
                    style=Dict(
                        "backgroundColor"=> backgroundColor,
                        "borderLeft"=> "1px rgb(50, 50, 50) solid",
                        "height"=> "10px"
                    )
                ),
                # html_small(round(min_bound, digits=2), style=Dict("paddingLeft"=> "2px"))
                html_small(backgroundColor, style=Dict("paddingLeft"=> "2px"))
            ])
        )
      end

    return (styles, html_div(legend, style=Dict("padding"=> "5px 0 5px 0")))
end


(styles, legend) = discrete_background_color_bins(df, n_bins=31, columns="all")
app.layout = html_div([
    html_div(children=[legend], style=Dict("float" => "right")),
    dash_datatable(
        data=Dict.(pairs.(eachrow(df))),
        sort_action="native",
        columns=[Dict("name" =>c, "id" => c) for c in names(df)],
        style_data_conditional=styles
    )
])


run_server(app, "0.0.0.0", debug=true)