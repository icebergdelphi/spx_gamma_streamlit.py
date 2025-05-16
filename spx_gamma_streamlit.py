import streamlit as st
import pandas as pd
import numpy as np
import scipy
from scipy.stats import norm
import plotly.graph_objects as go
import requests
from datetime import datetime
import sys

pd.options.display.float_format = '{:,.4f}'.format

# Black-Scholes European-Options Gamma
def calcGammaEx(S, K, vol, T, r, q, optType, OI):
    if T == 0 or vol == 0:
        return 0
    dp = (np.log(S/K) + (r - q + 0.5*vol**2)*T) / (vol*np.sqrt(T))
    dm = dp - vol*np.sqrt(T)
    if optType == 'call':
        gamma = np.exp(-q*T) * norm.pdf(dp) / (S * vol * np.sqrt(T))
        return OI * 100 * S * S * 0.01 * gamma
    else:  # Gamma is same for calls and puts. This is just to cross-check
        gamma = K * np.exp(-r*T) * norm.pdf(dm) / (S * S * vol * np.sqrt(T))
        return OI * 100 * S * S * 0.01 * gamma

# Selección del índice por parte del usuario
index = st.selectbox("Seleccione el índice:", ["SPX", "NDX"])

# Obtener datos de CBOE
try:
    response = requests.get(url=f"https://cdn.cboe.com/api/global/delayed_quotes/options/_" + index + ".json")
    response.raise_for_status()  # Verifica si la solicitud fue exitosa
    data_json = response.json()
    spotPrice = data_json["data"]["close"]
    options_data = data_json["data"]["options"]
except Exception as e:
    st.error(f"Error al obtener datos de CBOE para {index}: {e}")
    st.stop()

# Convertir datos JSON a DataFrame
df = pd.DataFrame(options_data)
df['strike'] = df['option'].str.extract(r'SPX\d{6}[CP](\d{6})' if index == "SPX" else r'NDX\d{6}[CP](\d{6})').astype(float) / 10000  # Extraer strike según el índice
df['optType'] = df['option'].str[-7].replace({'C': 'call', 'P': 'put'})
df['open_interest'] = df['open_interest'].astype(float)
df['iv'] = df['iv'].astype(float)  # Implied Volatility from JSON

# Calcular tiempo hasta la expiración (T) usando last_trade_time y la fecha actual (16/05/2025)
current_date = datetime(2025, 5, 16)
df['expiration'] = pd.to_datetime(df['option'].str.extract(r'SPX\d{2}(\d{2}\d{2}\d{2})' if index == "SPX" else r'NDX\d{2}(\d{2}\d{2}\d{2})')[0], format='%y%m%d')
df['T'] = (df['expiration'] - current_date).dt.days / 365.0  # T en años

# Parámetros para calcGammaEx
r = 0.05  # Tasa libre de riesgo (5% asumido)
q = 0.02  # Dividendo (2% asumido)

# Calcular Gamma usando calcGammaEx
df['gamma_calc'] = df.apply(lambda row: calcGammaEx(
    S=spotPrice,
    K=row['strike'],
    vol=row['iv'],
    T=row['T'] if row['T'] > 0 else 0.0001,  # Evitar T=0
    r=r,
    q=q,
    optType=row['optType'],
    OI=row['open_interest']
), axis=1)

# ---=== CALCULATE SPOT GAMMA ===---
# Separar Calls y Puts para calcular GEX
df_calls = df[df['optType'] == 'call'].copy()
df_puts = df[df['optType'] == 'put'].copy()

# Calcular CallGEX y PutGEX
df_calls['CallGEX'] = df_calls['gamma_calc']
df_puts['PutGEX'] = df_puts['gamma_calc'] * -1  # Negativo para puts

# Combinar Calls y Puts de nuevo
df = pd.concat([df_calls, df_puts], ignore_index=True)

# Calcular TotalGamma
df['TotalGamma'] = df['CallGEX'].fillna(0) + df['PutGEX'].fillna(0)
df['TotalGamma'] = df['TotalGamma'] / 10**9  # Convertir a miles de millones

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
    xbins=dict(start=fromStrike, end=toStrike, size=6)
))

# Añadir línea vertical para el Spot
fig1.add_vline(x=spotPrice, line=dict(color='red', width=2, dash='dash'), annotation_text=f"{index} Spot: {spotPrice:,.0f}", annotation_position="top right")

# Configurar el diseño del gráfico
fig1.update_layout(
    title=f"Gamma Total: ${df['TotalGamma'].sum():,.2f} MM por 1% Movimiento del {index}",
    xaxis_title="Strike",
    yaxis_title="Exposición a Gamma Spot ($ miles de millones/1% movimiento)",
    xaxis=dict(range=[fromStrike, toStrike], tickformat=",", automargin=True),
    yaxis=dict(tickformat=".2f", automargin=True),
    showlegend=True,
    template="plotly_dark",
    font=dict(size=14),
    margin=dict(l=20, r=20, t=50, b=50),
    width=1200,
    height=500
)

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig1, use_container_width=True)

# Chart 2: Absolute Gamma Exposure by Calls and Puts con Plotly
fig2 = go.Figure()

# Filtrar por Calls y Puts
df_calls = df[df['optType'] == 'call'].groupby('strike').agg({'CallGEX': 'sum'}).reset_index()
df_puts = df[df['optType'] == 'put'].groupby('strike').agg({'PutGEX': 'sum'}).reset_index()
dfPlot_calls = df_calls[(df_calls['strike'] >= fromStrike) & (df_calls['strike'] <= toStrike)]
dfPlot_puts = df_puts[(df_puts['strike'] >= fromStrike) & (df_puts['strike'] <= toStrike)]
threshold = max(dfPlot_calls['CallGEX'].abs().max() if not dfPlot_calls.empty else 0, dfPlot_puts['PutGEX'].abs().max() if not dfPlot_puts.empty else 0) * 0.05
dfPlot_calls = dfPlot_calls[dfPlot_calls['CallGEX'].abs() > threshold] if not dfPlot_calls.empty else dfPlot_calls
dfPlot_puts = dfPlot_puts[dfPlot_puts['PutGEX'].abs() > threshold] if not dfPlot_puts.empty else dfPlot_puts

# Añadir barras para Calls
fig2.add_trace(go.Histogram(
    x=dfPlot_calls['strike'],
    y=dfPlot_calls['CallGEX'] / 10**9,
    histfunc='sum',
    name='Gamma de Calls',
    marker_color='green',
    opacity=0.7,
    xbins=dict(start=fromStrike, end=toStrike, size=1)
))

# Añadir barras para Puts
fig2.add_trace(go.Histogram(
    x=dfPlot_puts['strike'],
    y=dfPlot_puts['PutGEX'] / 10**9,
    histfunc='sum',
    name='Gamma de Puts',
    marker_color='red',
    opacity=0.7,
    xbins=dict(start=fromStrike, end=toStrike, size=1)
))

# Añadir línea vertical para el Spot
fig2.add_vline(x=spotPrice, line=dict(color='blue', width=2, dash='dash'), annotation_text=f"{index} Spot: {spotPrice:,.0f}", annotation_position="top right")

# Añadir anotaciones para los strikes más significativos
top_call_strikes = dfPlot_calls['CallGEX'].nlargest(3).index.tolist() if not dfPlot_calls.empty else []
top_put_strikes = dfPlot_puts['PutGEX'].nsmallest(3).index.tolist() if not dfPlot_puts.empty else []

for strike in top_call_strikes:
    gamma_value = dfPlot_calls.loc[strike, 'CallGEX'] / 10**9
    fig2.add_annotation(
        x=strike, y=gamma_value,
        text=f"{strike:.0f}: {gamma_value:.2f}B",
        showarrow=True, arrowhead=1, ax=0, ay=-30,
        bgcolor="white", bordercolor="green", opacity=0.7
    )

for strike in top_put_strikes:
    gamma_value = dfPlot_puts.loc[strike, 'PutGEX'] / 10**9
    fig2.add_annotation(
        x=strike, y=gamma_value,
        text=f"{strike:.0f}: {gamma_value:.2f}B",
        showarrow=True, arrowhead=1, ax=0, ay=30,
        bgcolor="white", bordercolor="red", opacity=0.7
    )

# Configurar el diseño del gráfico
fig2.update_layout(
    title=f"Gamma Total: ${df['TotalGamma'].sum():,.2f} MM por 1% Movimiento del {index}",
    xaxis_title="Strike",
    yaxis_title="Exposición a Gamma Spot ($ miles de millones/1% movimiento)",
    xaxis=dict(range=[fromStrike, toStrike], tickformat=",", automargin=True),
    yaxis=dict(tickformat=".2f", automargin=True),
    showlegend=True,
    template="plotly_dark",
    font=dict(size=14),
    margin=dict(l=20, r=20, t=50, b=50),
    width=1200,
    height=500,
    barmode='overlay'
)

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig2, use_container_width=True)
