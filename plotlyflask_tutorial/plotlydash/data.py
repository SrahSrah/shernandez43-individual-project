"""Prepare data for Plotly Dash."""
import os
os.system('pip install scikit-learn')
import pandas as pd
import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def get_df_and_figs(cols=['Cumulative_cases','death_rate']):
    df = pd.read_csv('data/data.csv')
    fig1 = px.choropleth(df, locations="Country_code",
                    color=cols[0], # lifeExp is a column of gapminder
                    hover_name="Country", # column to add to hover information
                    color_continuous_scale=px.colors.sequential.Blues,
                   title=cols[0] + ' Across the Globe')
    fig2 = px.choropleth(df, locations="Country_code",
                    color=cols[1], # lifeExp is a column of gapminder
                    hover_name="Country", # column to add to hover information
                    color_continuous_scale=px.colors.sequential.Blues,
                   title=cols[1] + ' Across the Globe')

    return df, fig1, fig2

def get_df():
    return pd.read_csv('data/data.csv')

def get_fig(df, col='Cumulative_cases'):
    fig = px.choropleth(df, locations="Country_code",
                    color=col, # lifeExp is a column of gapminder
                    hover_name="Country", # column to add to hover information
                    color_continuous_scale=px.colors.sequential.Blues,
                   title=col + ' Across the Globe')
    return fig


def make_scatter(dq, cols):
    X = dq[cols[0]]
    Y = dq[cols[1]]

    x = np.array(X).reshape((-1, 1))
    y = np.array(Y)
    model = LinearRegression().fit(x, y)
    y_pred = model.predict(x)
    r2 = r2_score(y, y_pred)

    # fig1 = px.scatter(dq, x, y, trendline="ols")
    # trendline1 = fig1.data[1] # second trace, first one is scatter
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X, y=Y, mode='markers', text=dq['Country']))
    fig.add_trace(go.Scatter(x=X, y=y_pred, mode='lines', name='OLS fit', text='R sqared: ' + str(r2)))
    fig = update_fig(fig, x=cols[0], y=cols[1],
    title=cols[1] + ' vs. ' + cols[0] + ', with an R-squared of '+ str(r2))

    return fig

def update_fig(fig, x='', y='', title='', legend=''):
    fig.update_layout(
    title=title,
    xaxis_title=x,
    yaxis_title=y,
    legend_title=legend,
    font=dict(
        family="Courier New, monospace",
        size=16))
    return fig
