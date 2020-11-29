"""Instantiate a Dash app."""
import numpy as np
import pandas as pd
import dash
import dash_table
import dash_html_components as html
import dash_core_components as dcc
from .data import get_df_and_figs, make_scatter, get_df, get_fig
from dash.dependencies import Input, Output
from .layout import html_layout


def init_dashboard(server):
    """Create a Plotly Dash dashboard."""
    dash_app = dash.Dash(
        server=server,
        routes_pathname_prefix='/dashapp/',
        external_stylesheets=[
            '/static/dist/css/styles.css',
            'https://fonts.googleapis.com/css?family=Lato'
        ]
    )

    # Load starting df and figs
    cols=['Cumulative_cases','New_cases']
    df, fig1, fig2  = get_df_and_figs(cols)
    metric_cols = [col for col in df.columns[1:] if col not in ['Country_code', 'Country']]
    scat = make_scatter(df, cols)
    starter_cols = ['Country', 'Cumulative_cases','New_cases']
    columns = [{"name": i, "id": i} for i in starter_cols]
    # Custom HTML layout
    dash_app.index_string = html_layout

    # Create Layout
    dash_app.layout = html.Div(
        children=[dcc.Graph(id='map1', figure=fig1),
        dcc.Graph(id='map2', figure=fig2),
        dcc.Graph(id='scatter', figure=scat),
        html.Div([html.H2("Change the value in the dropdowns to see the maps change!")]),
        html.Div([dcc.Dropdown(
                id='xaxis-column',
                options=[{'label': i, 'value': i} for i in metric_cols],
                value=cols[0]
                )], style={'width': '48%', 'display': 'inline-block'}),
         html.Div([dcc.Dropdown(
                id='yaxis-column',
                options=[{'label': i, 'value': i} for i in metric_cols],
                value=cols[1]
                )],style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
        # create_data_table(df, cols=df.columns)
        dash_table.DataTable(
            id='my-table',
            columns=columns,
            data=df[starter_cols].to_dict('records'),
            page_size=300,
        )
        ],
        id='dash-container'
    )

    @dash_app.callback(
        Output('scatter', 'figure'),
        Output('map1', 'figure'),
        Output('map2', 'figure'),
        Output("my-table", "data"),
        Output('my-table', 'columns'),
        Input('xaxis-column', 'value'),
        Input('yaxis-column', 'value'))
    def update_scatter(x_col, y_col):
        df = get_df()
        cols = [x_col, y_col]
        scat = make_scatter(df, cols)

        cols = ['Country', x_col, y_col]
        columns = [{"name": i, "id": i} for i in cols]
        data = df[cols].to_dict('records')
        map1 = get_fig(df, x_col)
        map2 = get_fig(df, y_col)
        return scat, map1, map2, data, columns



    return dash_app.server





def create_data_table(dq, cols):
    """Create Dash datatable from Pandas DataFrame."""
    dq = dq[cols]
    starter_cols = ['Country', 'Cumulative_cases','death_rate']
    columns = [{"name": i, "id": i} for i in starter_cols]

    table = dash_table.DataTable(
        id='table',
        columns=columns,
        data=dq[starter_cols].to_dict('records'),
        sort_action="native",
        sort_mode='native',
    )
    return table
