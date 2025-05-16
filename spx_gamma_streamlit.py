import streamlit as st
import pandas as pd
import numpy as np
import scipy
from scipy.stats import norm
import plotly.graph_objects as go
import requests
from datetime import datetime

pd.options.display.float_format = '{:,.4f}'.format


# Obtener datos de CBOE
try:
    response = requests.get(url="https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json")
    response.raise_for_status()  # Verifica si la solicitud fue exitosa
    data_json = response.json()
    spotPrice = data_json["data"]["close"]
    options_data = data_json["data"]["options"]
except Exception as e:
    st.error(f"Error al obtener datos de CBOE: {e}")
    st.stop()

# Convertir datos JSON a DataFrame
df = pd.DataFrame(options_data)
df['strike'] = df['option'].str.extract(r'SPX\d{6}[CP](\d{6})').astype(float) / 10000  # Extraer strike del código de opción
df['optType'] = df['option'].str[-7].replace({'C': 'call', 'P': 'put'})
df['open_interest'] = df['open_interest'].astype(float)
df['gamma'] = df['gamma'].astype(float)

# Calcular Gamma Exposure
df['GEX'] = df.apply(lambda row: row['gamma'] * row['open_interest'] * 100 * spotPrice * spotPrice * 0.01 * (-1 if row['optType'] == 'put' else 1), axis=1)
df['TotalGamma'] = df['GEX'] / 10**9  # Convertir a miles de millones

# Agrupar por strike
dfAgg = df.groupby('strike').agg({'TotalGamma': 'sum'}).reset_index()

# Rango de strikes
fromStrike = 0.8 * spotPrice
toStrike = 1.2 * spotPrice

# Filtrar datos para los gráficos
dfPlot = dfAgg[(dfAgg['strike'] >= fromStrike) & (dfAgg['strike'] <= toStrike)]
threshold = dfPlot['TotalGamma'].abs().max() * 0.01
dfPlot = dfPlot[dfPlot['TotalGamma'].abs() > threshold]

# Chart 1: Absolute Gamma Exposure con Plotly
fig1 = go.Figure()

# Añadir barras para la exposición total a Gamma
fig1.add_trace(go.Histogram(
    x=dfPlot['strike'],
    y=dfPlot['TotalGamma'],
    histfunc='sum',
    name='Exposición a Gamma',
    marker_color='gray',
    opacity=0.7,
    xbins=dict(start=fromStrike, end=toStrike, size=6)  # Tamaño de las barras similar al original
))

# Añadir línea vertical para el SPX Spot
fig1.add_vline(x=spotPrice, line=dict(color='red', width=2, dash='dash'), annotation_text=f"SPX Spot: {spotPrice:,.0f}", annotation_position="top right")

# Configurar el diseño del gráfico
fig1.update_layout(
    title=f"Gamma Total: ${df['TotalGamma'].sum():,.2f} MM por 1% Movimiento del SPX",
    xaxis_title="Strike",
    yaxis_title="Exposición a Gamma Spot ($ miles de millones/1% movimiento)",
    xaxis=dict(range=[fromStrike, toStrike], tickformat=","),
    yaxis=dict(tickformat=".2f"),
    showlegend=True,
    template="plotly_dark",  # Tema oscuro para un aspecto más moderno
    font=dict(size=14),
    margin=dict(l=50, r=50, t=50, b=50)
)

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig1, use_container_width=True)

# Chart 2: Absolute Gamma Exposure by Calls and Puts con Plotly
fig2 = go.Figure()

# Filtrar por Calls y Puts
df_calls = df[df['optType'] == 'call'].groupby('strike').agg({'GEX': 'sum'}).reset_index()
df_puts = df[df['optType'] == 'put'].groupby('strike').agg({'GEX': 'sum'}).reset_index()
dfPlot_calls = df_calls[(df_calls['strike'] >= fromStrike) & (df_calls['strike'] <= toStrike)]
dfPlot_puts = df_puts[(df_puts['strike'] >= fromStrike) & (df_puts['strike'] <= toStrike)]
threshold = max(dfPlot_calls['GEX'].abs().max(), dfPlot_puts['GEX'].abs().max()) * 0.05
dfPlot_calls = dfPlot_calls[dfPlot_calls['GEX'].abs() > threshold]
dfPlot_puts = dfPlot_puts[dfPlot_puts['GEX'].abs() > threshold]

# Añadir barras para Calls
fig2.add_trace(go.Histogram(
    x=dfPlot_calls['strike'],
    y=dfPlot_calls['GEX'] / 10**9,
    histfunc='sum',
    name='Gamma de Calls',
    marker_color='green',
    opacity=0.7,
    xbins=dict(start=fromStrike, end=toStrike, size=1)
))

# Añadir barras para Puts
fig2.add_trace(go.Histogram(
    x=dfPlot_puts['strike'],
    y=dfPlot_puts['GEX'] / 10**9,
    histfunc='sum',
    name='Gamma de Puts',
    marker_color='red',
    opacity=0.7,
    xbins=dict(start=fromStrike, end=toStrike, size=1)
))

# Añadir línea vertical para el SPX Spot
fig2.add_vline(x=spotPrice, line=dict(color='blue', width=2, dash='dash'), annotation_text=f"SPX Spot: {spotPrice:,.0f}", annotation_position="top right")

# Añadir anotaciones para los strikes más significativos
top_call_strikes = dfPlot_calls['GEX'].nlargest(3).index.tolist()
top_put_strikes = dfPlot_puts['GEX'].nsmallest(3).index.tolist()

for strike in top_call_strikes:
    gamma_value = dfPlot_calls.loc[strike, 'GEX'] / 10**9
    fig2.add_annotation(
        x=strike, y=gamma_value,
        text=f"{strike:.0f}: {gamma_value:.2f}B",
        showarrow=True, arrowhead=1, ax=0, ay=-30,
        bgcolor="white", bordercolor="green", opacity=0.7
    )

for strike in top_put_strikes:
    gamma_value = dfPlot_puts.loc[strike, 'GEX'] / 10**9
    fig2.add_annotation(
        x=strike, y=gamma_value,
        text=f"{strike:.0f}: {gamma_value:.2f}B",
        showarrow=True, arrowhead=1, ax=0, ay=30,
        bgcolor="white", bordercolor="red", opacity=0.7
    )

# Configurar el diseño del gráfico
fig2.update_layout(
    title=f"Gamma Total: ${df['TotalGamma'].sum():,.2f} MM por 1% Movimiento del SPX",
    xaxis_title="Strike",
    yaxis_title="Exposición a Gamma Spot ($ miles de millones/1% movimiento)",
    xaxis=dict(range=[fromStrike, toStrike], tickformat=","),
    yaxis=dict(tickformat=".2f"),
    showlegend=True,
    template="plotly_dark",
    font=dict(size=14),
    margin=dict(l=50, r=50, t=50, b=50),
    barmode='overlay'  # Superponer barras para mejor visualización
)

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig2, use_container_width=True)
