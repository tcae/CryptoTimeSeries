using Dash, PlotlyJS, JSON3

    app = dash(external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"])

    xdata= rand(1:10,3)
    ydata= rand(1:10,3)
    matrix= rand(1:10,3,3)



    function vline(xposition,ydata)

        scatter(Dict(
                    :mode => "lines",
                    :x => fill(xposition,2),
                    :y => [minimum(ydata),maximum(ydata)],

                ),
                showlegend=false,
                marker_line_width = 0.5,
                marker_opacity = 0.8,
                line_color= "#da0a72b6"
        )
    end

                                            #Main layout

    app.layout =    html_div() do
        html_div(
            children=[
                dcc_graph(
                    id="Hidden-Heatmap",
                    figure = (
                        Plot(
                            heatmap(
                                Dict(
                                    :type => "heatmap",
                                    :x => xdata,
                                    :y => ydata,
                                    :z => [matrix[i,:] for i in 1:size(matrix)[1]]
                                ),
                                zmax= 10,
                                zmin= 0 ,
                                colorscale= "Viridis"
                            ),
                            Layout(
                                uirevision = "zoom"
                            )
                        )
                    ),style= Dict(
                        :display => "none"
                    )
                 ),
                html_div(
                    children =[
                        dcc_graph(id="Heatmap",
                        ),
                        html_div(
                            id="Slider Div",
                            dcc_rangeslider(
                                id = "Ranged Slider",
                                step = 0.01,
                                min=0,
                                max=10,
                                value=[3,5],
                                persistence=true ,
                                allowCross=false,
                            )
                        )
                    ]
                )
            ],
            className ="twelve columns"
        )
    end

    # Vline into Graph from Hidden Graph figure
    callback!(app,
    Output("Heatmap","figure"),
    Input("Ranged Slider","value"),
    State("Hidden-Heatmap","figure")) do value,current_fig
        @show length(value)

            type = current_fig[:data][1][:type]
            zmax = current_fig[:data][1][:zmax]
            zmin = current_fig[:data][1][:zmin]

            xdata = current_fig[:data][1][:x]
            ydata = current_fig[:data][1][:y]
            zdata = current_fig[:data][1][:z]

            cur_plot =  Plot(
                            heatmap(
                                Dict(
                                    :type => type,
                                    :x    => xdata,
                                    :y    => ydata,
                                    :z    => zdata
                                ),
                                zmax = 10,
                                zmin = 1,
                                colorscale= "Viridis",
                                uirevision = "zoom"
                            ),
                            Layout(uirevision = "zoom"
                            )
                        )
                for i in eachindex(value)  # 1:length(value)
                    addtraces!(cur_plot,vline(value[i],ydata))
                end
            return figure=(cur_plot)

    end

    # Autosize range of ranged sliders to Zoom of Heatmap
    callback!(app,
    Output("Ranged Slider","min"),
    Output("Ranged Slider","max"),
    Input("Heatmap","relayoutData"),
    State("Heatmap","figure"))  do   range,figure

        #* TCAE sample code to get JSON symbols
        if !(range === nothing)
            if (try range[:autosize]  catch end === nothing)
                # myrange = JSON3.read(range, MyRelayoutData)
                JSON3.pretty(JSON3.write(range))
                println("relayout data: $(range[Symbol("xaxis.range[0]")])")
            else
                JSON3.pretty(JSON3.write(range))
                println("relayout data: autosize = $(range[:autosize])")
            end
        end
        #* ^^^

        # Set ranges of RangedSliders
        ## if relayoutData is at default
        if  (try range[:autosize]  catch end !== nothing)
            xdata_fig = figure[:data][1][:x]
            min = minimum(xdata_fig)
            max = maximum(xdata_fig)

                return min,max
        ## changed the zoom
        elseif  try range[:autosize]  catch end === nothing && typeof(range) == (NamedTuple{(Symbol("xaxis.range[0]"), Symbol("xaxis.range[1]"), Symbol("yaxis.range[0]"), Symbol("yaxis.range[1]")),NTuple{4,Float64}})
            min = range[1]
            max = range[2]
            println("min = $min  max = $max")

                return min,max
        ## set back to autosize
        elseif  try range[:autosize]  catch end === nothing && typeof(range) == (NamedTuple{(Symbol("xaxis.autorange"), Symbol("yaxis.autorange")),Tuple{Bool,Bool}})
            xdata_fig = figure[:data][1][:x]
            min = minimum(xdata_fig)
            max = maximum(xdata_fig)

                return min,max
        else
            return Dash.no_update(), Dash.no_update()
        end
    end


    run_server(app, "0.0.0.0", debug=true)