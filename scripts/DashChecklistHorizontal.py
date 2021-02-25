

# import plotly.graph_objects as go
import dash
# import dash_core_components as dcc
# import dash_html_components as html
# from dash.dependencies import Input, Output, State
# import dash_table
# import dash_table.FormatTemplate as FormatTemplate
# from dash_table.Format import Sign
# from dash_table.Format import Format, Scheme


import dash_core_components as dcc

# external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app = dash.Dash(__name__)

app.layout = dcc.Checklist(
        options=[
            {'label': 'New York City', 'value': 'NYC'},
            {'label': 'Montréal', 'value': 'MTL'},
            {'label': 'San Francisco', 'value': 'SF'}
        ],
        value=['MTL', 'SF'],
        labelStyle={'display': 'inline-block'}
    )


app.run_server()

