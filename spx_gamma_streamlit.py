import streamlit as st
import pandas as pd
import numpy as np
import scipy
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import requests
from datetime import datetime

pd.options.display.float_format = '{:,.4f}'.format

# Título de la aplicación
st.title("SPX Gamma Exposure Dashboard")

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

# Chart 1: Absolute Gamma Exposure
plt.figure(figsize=(12, 7))
plt.grid(True, linestyle='--', alpha=0.7)
width = 6
plt.bar(dfPlot['strike'].values, dfPlot['TotalGamma'].to_numpy(), width=width, linewidth=0.1, edgecolor='k', label="Exposición a Gamma")
plt.xlim([fromStrike, toStrike])
chartTitle = "Gamma Total: $" + str("{:.2f}".format(df['TotalGamma'].sum())) + " MM por 1% Movimiento del SPX"
plt.title(chartTitle, fontweight="bold", fontsize=20)
plt.xlabel('Strike', fontweight="bold", fontsize=14)
plt.ylabel('Exposición a Gamma Spot ($ miles de millones/1% movimiento)', fontweight="bold", fontsize=14)
plt.axvline(x=spotPrice, color='r', lw=2, label="SPX Spot: " + str("{:,.0f}".format(spotPrice)))
plt.gca().xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
plt.gca().yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.2f}'))
plt.legend(fontsize=12)
st.pyplot(plt)

# Chart 2: Absolute Gamma Exposure by Calls and Puts
plt.figure(figsize=(12, 7))
plt.grid(True, linestyle='--', alpha=0.7)
# Filtrar por Calls y Puts
df_calls = df[df['optType'] == 'call'].groupby('strike').agg({'GEX': 'sum'}).reset_index()
df_puts = df[df['optType'] == 'put'].groupby('strike').agg({'GEX': 'sum'}).reset_index()
dfPlot_calls = df_calls[(df_calls['strike'] >= fromStrike) & (df_calls['strike'] <= toStrike)]
dfPlot_puts = df_puts[(df_puts['strike'] >= fromStrike) & (df_puts['strike'] <= toStrike)]
threshold = max(dfPlot_calls['GEX'].abs().max(), dfPlot_puts['GEX'].abs().max()) * 0.05
dfPlot_calls = dfPlot_calls[dfPlot_calls['GEX'].abs() > threshold]
dfPlot_puts = dfPlot_puts[dfPlot_puts['GEX'].abs() > threshold]
width = 1
offset = 0.2
plt.bar(dfPlot_calls['strike'].values - offset, dfPlot_calls['GEX'].to_numpy() / 10**9, width=width, linewidth=0.1, edgecolor='k', color='green', alpha=0.7, label="Gamma de Calls")
plt.bar(dfPlot_puts['strike'].values + offset, dfPlot_puts['GEX'].to_numpy() / 10**9, width=width, linewidth=0.1, edgecolor='k', color='red', alpha=0.7, label="Gamma de Puts")
plt.xlim([fromStrike, toStrike])
chartTitle = "Gamma Total: $" + str("{:.2f}".format(df['TotalGamma'].sum())) + " MM por 1% Movimiento del SPX"
plt.title(chartTitle, fontweight="bold", fontsize=20)
plt.xlabel('Strike', fontweight="bold", fontsize=14)
plt.ylabel('Exposición a Gamma Spot ($ miles de millones/1% movimiento)', fontweight="bold", fontsize=14)
plt.axvline(x=spotPrice, color='blue', lw=2, label="SPX Spot: " + str("{:,.0f}".format(spotPrice)))
plt.gca().xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
plt.gca().yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.2f}'))
top_call_strikes = dfPlot_calls['GEX'].nlargest(3).index.tolist()
top_put_strikes = dfPlot_puts['GEX'].nsmallest(3).index.tolist()
for strike in top_call_strikes:
    gamma_value = dfPlot_calls.loc[strike, 'GEX'] / 10**9
    plt.annotate(f"{strike:.0f}: {gamma_value:.2f}B", xy=(strike - offset, gamma_value), xytext=(0, 5), textcoords='offset points', ha='center', fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.7))
for strike in top_put_strikes:
    gamma_value = dfPlot_puts.loc[strike, 'GEX'] / 10**9
    plt.annotate(f"{strike:.0f}: {gamma_value:.2f}B", xy=(strike + offset, gamma_value), xytext=(0, -15), textcoords='offset points', ha='center', fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.7))
plt.legend(fontsize=12)
st.pyplot(plt)
