using Dash

using CSV, DataFrames

df_gapminder = CSV.read(download("https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv"), DataFrame)
df = df_gapminder[1:500,:]
# println(df)
df.id = [i for i in 1:500]

app = dash()

app.layout =  dash_datatable(
        data=Dict.(pairs.(eachrow(df))),
        sort_action="native",
        columns=[Dict("name" =>c, "id" => c) for c in names(df)],
        style_data_conditional=[Dict(
          "if" =>  Dict("filter_query" =>  "{id} = 3", "column_id" => "pop"),
          "backgroundColor" =>  "#85144b",
          "color" =>  "white"
        )]
          # ),
          # Dict(
          #   "if" =>  Dict("filter_query" =>  "{id} = 5", "column_id" => "pop"),
          #   "background" =>  Dict("linear-gradient" =>
          #     Dict("white" => "0%" , "white" => "50%", "{color_above}" => "50%", "{color_above}" => "10%", "white" => "10%", "white" => "100%"))
          # ),
          # "color" =>  "white"]
    )

run_server(app, "0.0.0.0", debug=true)
