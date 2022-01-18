from cProfile import label
from typing import Container, ValuesView
import dash
from dash.development.base_component import Component
from dash import dcc
from dash import html
from datetime import datetime as dt
from datetime import date
from dash.dependencies import Input, Output, State
from flask.helpers import url_for
import yfinance as yf
import pandas as pd
import pandas_datareader as data
import plotly.graph_objs as go
import plotly.express as px
from plotly.graph_objects import Layout
from plotly.validator_cache import ValidatorCache
from yfinance import ticker
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from model import prediction
from dash.exceptions import PreventUpdate
#from matplotlib.pyplot import plt

#get stock data
start='2012-1-1'
end='2022-1-3'
app = dash.Dash(__name__)
server = app.server
app.layout=html.Div(
    className="container",
    children=[
        dcc.Store(id='ticker_code',storage_type='session'),
        html.Div(children=[
            html.H1("Welcome to stock dash app!",className="start",style={'color':'#ffffff'}),
            html.Div(children=[
                html.Div([
                    dcc.Input(id="stock_code",type='text',placeholder="Enter Stock Code",className="inputbox"),
                    html.Button('Submit',id='submit_button',className="submit_button",style={'color':'#fffbf5'})
                ]),
                html.Div(
                children=[
                # Date range picker input
                dcc.DatePickerRange(
                    id='date_range_picker',
                    min_date_allowed=date(1990,12,31),
                    initial_visible_month=date(2021,12,3),
                    end_date=dt.now(), 
                )
                ]),
                html.Div(
                className='button',
                children=[
                # Stock price button
                html.Button('Stock Price',id='Stock_price_button',className="stock_p_b",style={'color':'#fffbf5'}),
                html.Button('Forcast',id='Forcast_button',className="forecast_b",style={'color':'#fffbf5'}),
                
                ]
                ),
                html.Div(children=[
                    dcc.Input(id="days",type='number',placeholder="Number of Days",className="inputbox"),
                    html.Button('Indicator',id='Indicator_button',className="indicator_b",style={'color':'#fffbf5'})
                ]
                )
            ])
            ],
                 className='nav'),
        html.Div(
        [
            html.Div(
                children=[ 
                html.Img('',id='logo',className='company_logo'),
                html.H1('',id='c_name',className='company_name',style={'color':'#064635'})
                ],
                className="header"),
            html.Div( #Description
                     html.P('',id='description', className='c_desc',style={'color':'#064635'}),),
            html.Div(
                children=[
                    dcc.Graph(id='stock_graph'),
                    dcc.Graph(id='indicator_graph'),
                    dcc.Graph(id='prediction_graph'),
                    
                ]
                ),
            
            ],
        className="content"
    )
    ]
)
@app.callback(
    [
        Output(component_id="logo",component_property='src'),
        Output(component_id="c_name",component_property='children'),
        Output(component_id="description",component_property='children'),
        
    ],
    Input(component_id='stock_code',component_property='value'),
    State(component_id='submit_button',component_property='n_clicks')
)
def update_data(arg1, arg2):
    global stock_name
    stock_name=arg1
    ticker = yf.Ticker(arg1)
    inf = ticker.info
    df = pd.DataFrame().from_dict(inf, orient="index").T
    #print(stock_name)
    return df['logo_url'],df['shortName'],df['longBusinessSummary']

@app.callback(
    Output("stock_graph",'figure'),
    [
    Input("date_range_picker",'start_date'),
    Input("date_range_picker",'end_date'),
    #Input("ticker_code",'name')
    ],
    State('Stock_price_button','n_clicks')
)
def stock_graph_show(start_date,end_date,input_param):
    #print(start_date,end_date,input_param)
    df = yf.download(stock_name,start_date,end_date)
    df.reset_index(inplace=True)
    title="Closing and Opening Price vs Date"
    fig=px.line(df,x='Date',y=['Open','Close'],title=title)
    
    return fig

@app.callback(
    Output("indicator_graph",'figure'),
    [
    Input("date_range_picker",'start_date'),
    Input("date_range_picker",'end_date'),
    ],
    State('Indicator_button','n_clicks')
)
def get_more(start_date,end_date,input_param):
    df=yf.download(stock_name,start_date,end_date)
    #df=data.DataReader(stock_name,'yahoo',start_date,end_date)
    df=df.reset_index()
    #print(df)
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    #print(start_date,end_date,input_param,stock_name,df['EWA_20'])
    fig = px.line(df,x= 'Date',y= ['EMA_20','Close'],title="Exponential Moving Average vs Date")
    #fig.update_traces(opacity=0.7,mode='markers')
    return fig

# callback for forecast
@app.callback(
    Output("prediction_graph", "figure"),
    Input("days", "value"),
    State("Forcast_button", "n_clicks"))
def forecast(n_days, n):
    print(n,n_days,stock_name)
    if stock_name == None:
        raise PreventUpdate
    fig = prediction(stock_name, int(n_days) + 1)
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
    