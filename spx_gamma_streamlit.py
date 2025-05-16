import streamlit as st
import pandas as pd
import numpy as np
import scipy
from scipy.stats import norm
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta, date

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

def isThirdFriday(d):
    return d.weekday() == 4 and 15 <= d.day <= 21

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
data_df = pd.DataFrame(options_data)
data_df['CallPut'] = data_df['option'].str.slice(start=-9, stop=-8)
data_df['ExpirationDate'] = data_df['option'].str.slice(start=-15, stop=-9)
data_df['ExpirationDate'] = pd.to_datetime(data_df['ExpirationDate'], format='%y%m%d')
data_df['Strike'] = data_df['option'].str.slice(start=-8, stop=-3)
data_df['Strike'] = data_df['Strike'].str.lstrip('0')

data_df_calls = data_df.loc[data_df['CallPut'] == "C"]
data_df_puts = data_df.loc[data_df['CallPut'] == "P"]
data_df_calls = data_df_calls.reset_index(drop=True)
data_df_puts = data_df_puts.reset_index(drop=True)

df = data_df_calls[['ExpirationDate', 'option', 'last_trade_price', 'change', 'bid', 'ask', 'volume', 'iv', 'delta', 'gamma', 'open_interest', 'Strike']]
df_puts = data_df_puts[['ExpirationDate', 'option', 'last_trade_price', 'change', 'bid', 'ask', 'volume', 'iv', 'delta', 'gamma', 'open_interest', 'Strike']]
df_puts.columns = ['put_exp', 'put_option', 'put_last_trade_price', 'put_change', 'put_bid', 'put_ask', 'put_volume', 'put_iv', 'put_delta', 'put_gamma', 'put_open_interest', 'put_strike']

df = pd.concat([df, df_puts], axis=1)

df['check'] = np.where((df['ExpirationDate'] == df['put_exp']) & (df['Strike'] == df['put_strike']), 0, 1)

if df['check'].sum() != 0:
    st.warning("PUT CALL MERGE FAILED - OPTIONS ARE MISMATCHED.")
    st.stop()

df.drop(['put_exp', 'put_strike', 'check'], axis=1, inplace=True)

df.columns = ['ExpirationDate', 'Calls', 'CallLastSale', 'CallNet', 'CallBid', 'CallAsk', 'CallVol',
              'CallIV', 'CallDelta', 'CallGamma', 'CallOpenInt', 'StrikePrice', 'Puts', 'PutLastSale',
              'PutNet', 'PutBid', 'PutAsk', 'PutVol', 'PutIV', 'PutDelta', 'PutGamma', 'PutOpenInt']

df['ExpirationDate'] = pd.to_datetime(df['ExpirationDate'], format='%a %b %d %Y')
df['ExpirationDate'] = df['ExpirationDate'] + timedelta(hours=16)
df['StrikePrice'] = df['StrikePrice'].astype(float)
df['CallIV'] = df['CallIV'].astype(float)
df['PutIV'] = df['PutIV'].astype(float)
df['CallGamma'] = df['CallGamma'].astype(float)
df['PutGamma'] = df['PutGamma'].astype(float)
df['CallOpenInt'] = df['CallOpenInt'].astype(float)
df['PutOpenInt'] = df['PutOpenInt'].astype(float)

# Ajustar volatilidad implícita si es 0
df['CallIV'] = df['CallIV'].replace(0.0, 0.2)
df['PutIV'] = df['PutIV'].replace(0.0, 0.2)

# Get Today's Date
current_date = datetime.now()  # Obtiene la fecha y hora actuales dinámicamente

# Calcular tiempo hasta la expiración (T)
df['daysTillExp'] = (df['ExpirationDate'] - current_date).dt.total_seconds() / (365.25 * 24 * 60 * 60)
df['daysTillExp'] = df['daysTillExp'].clip(lower=1e-5)  # Evitar T <= 0

# ---=== CALCULATE SPOT GAMMA ===---
# Gamma Exposure = Unit Gamma * Open Interest * Contract Size * Spot Price
# To further convert into 'per 1% move' quantity, multiply by 1% of spotPrice
df['CallGEX'] = df['CallGamma'] * df['CallOpenInt'] * 100 * spotPrice * spotPrice * 0.01
df['PutGEX'] = df['PutGamma'] * df['PutOpenInt'] * 100 * spotPrice * spotPrice * 0.01 * -1

df['TotalGamma'] = (df['CallGEX'] + df['PutGEX']) / 10**9
dfAgg = df.groupby(['StrikePrice']).sum(numeric_only=True)

# Rango de strikes
fromStrike = 0.8 * spotPrice
toStrike = 1.2 * spotPrice

# Chart 1: Absolute Gamma Exposure
fig1 = go.Figure()
fig1.add_trace(go.Bar(
    x=dfAgg.index,
    y=dfAgg['TotalGamma'],
    width=6,
    marker_color='gray',
    opacity=0.7,
    name='Gamma Exposure'
))
fig1.add_vline(x=spotPrice, line=dict(color='red', width=2, dash='dash'), annotation_text=f"{index} Spot: {spotPrice:,.0f}", annotation_position="top right")
fig1.update_layout(
    title=f"Total Gamma: ${df['TotalGamma'].sum():,.2f} Bn per 1% {index} Move",
    xaxis_title="Strike",
    yaxis_title="Spot Gamma Exposure ($ billions/1% move)",
    xaxis=dict(range=[fromStrike, toStrike], tickformat=",", automargin=True),
    yaxis=dict(tickformat=".2f", automargin=True),
    showlegend=True,
    template="plotly_dark",
    font=dict(size=14),
    margin=dict(l=20, r=20, t=50, b=50),
    width=1200,
    height=500
)
st.plotly_chart(fig1, use_container_width=True)

# Chart 2: Open Interest by Calls and Puts
fig2 = go.Figure()
fig2.add_trace(go.Bar(
    x=dfAgg.index,
    y=dfAgg['CallOpenInt'],
    width=6,
    marker_color='green',
    opacity=0.7,
    name='Call OI'
))
fig2.add_trace(go.Bar(
    x=dfAgg.index,
    y=-1 * dfAgg['PutOpenInt'],
    width=6,
    marker_color='red',
    opacity=0.7,
    name='Put OI'
))
fig2.add_vline(x=spotPrice, line=dict(color='blue', width=2, dash='dash'), annotation_text=f"{index} Spot: {spotPrice:,.0f}", annotation_position="top right")
fig2.update_layout(
    title=f"Total Open Interest for {index}",
    xaxis_title="Strike",
    yaxis_title="Open Interest (number of contracts)",
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
st.plotly_chart(fig2, use_container_width=True)
