import streamlit as st
import pandas as pd
import numpy as np
import scipy
from scipy.stats import norm
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta, date

st.set_page_config(page_icon="🧊", layout="wide")

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
#Agregar para calcular la tasa de dividendos anual por ahora en el codigo esta fijo
#dividend_yield = st.number_input("Tasa de dividendos (% anual)", min_value=0.0, max_value=20.0, value=1.5, step=0.1)

# Función para calcular el precio estimado de ES o NQ
def calculate_future_price(index_price, r=0.04,d=0.015, index="SPX"):
    current_date = datetime.now()
    # Asumimos que el próximo vencimiento es el tercer viernes de junio 2025 (20/06/2025)
    next_expiry = datetime(2025, 6, 20, 16, 0)  # Cierre del mercado
    T = (next_expiry - current_date).total_seconds() / (365.25 * 24 * 60 * 60)
    future_price = index_price * np.exp((r - d) * T)
    return future_price

# Inicialización en el estado de la sesión
if 'last_request_time' not in st.session_state:
    st.session_state['last_request_time'] = datetime.now()
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'current_selection' not in st.session_state:
    st.session_state['current_selection'] = None

# Función para obtener datos de la API
def fetch_data(index):
    try:
        response = requests.get(f"https://cdn.cboe.com/api/global/delayed_quotes/options/_{index}.json")
        response.raise_for_status()
        data_json = response.json()
        return data_json
    except Exception as e:
        st.error(f"Error al obtener datos de CBOE para {index}: {e}")
        return None

# Selección del índice o conversión y tasa de interés
selection = st.selectbox("Seleccione el índice o conversión:", ["SPX", "NDX", "SPX=>ES", "NDX=>NQ"], key="selection")
risk_free_rate = st.number_input("Tasa de interés libre de riesgo (% anual)", min_value=0.0, max_value=20.0, value=5.0, step=0.1, key="risk_free_rate")

# Verificar si la selección cambió
if st.session_state['current_selection'] != selection:
    st.session_state['current_selection'] = selection
    st.session_state['data'] = None  # Forzar una nueva solicitud al cambiar la selección
    st.session_state['last_request_time'] = datetime.now()

# Verificar si han pasado 5 minutos (300 segundos) desde el último request o si no hay datos
current_time = datetime.now()
if (current_time - st.session_state['last_request_time']).total_seconds() > 300 or st.session_state['data'] is None:
    st.session_state['last_request_time'] = current_time
    data_json = fetch_data(selection.split('=>')[0] if '=>' in selection else selection)
    if data_json:
        st.session_state['data'] = data_json
else:
    st.write("Esperando la siguiente actualización en menos de 5 minutos...")
    data_json = st.session_state['data']

# Procesar datos si existen
if data_json:
    spotPrice = data_json["data"]["close"]
    futurePrice = calculate_future_price(spotPrice, r=risk_free_rate/100, d=0.015, index=selection.split('=>')[0] if '=>' in selection else selection)
    options_data = data_json["data"]["options"]

    # Determinar qué precio usar para los cálculos (spotPrice para SPX/NDX, futurePrice para SPX=>ES/NDX=>NQ)
    basePrice = futurePrice if "=>" in selection else spotPrice
    display_label = selection.replace("=>", " to ") if "=>" in selection else selection

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
    current_date = datetime.now()

    # Calcular tiempo hasta la expiración (T)
    df['daysTillExp'] = (df['ExpirationDate'] - current_date).dt.total_seconds() / (365.25 * 24 * 60 * 60)
    df['daysTillExp'] = df['daysTillExp'].clip(lower=1e-5)  # Evitar T <= 0

    # ---=== CALCULATE SPOT GAMMA ===---
    # Gamma Exposure = Unit Gamma * Open Interest * Contract Size * Spot Price
    # To further convert into 'per 1% move' quantity, multiply by 1% of basePrice
    df['CallGEX'] = df['CallGamma'] * df['CallOpenInt'] * 100 * basePrice * basePrice * 0.01
    df['PutGEX'] = df['PutGamma'] * df['PutOpenInt'] * 100 * basePrice * basePrice * 0.01 * -1

    df['TotalGamma'] = (df['CallGEX'] + df['PutGEX']) / 10**9
    dfAgg = df.groupby(['StrikePrice']).sum(numeric_only=True)

    # Rango de strikes
    fromStrike = 0.8 * basePrice
    toStrike = 1.2 * basePrice

  # Chart 1: Absolute Gamma Exposure
fig1 = go.Figure()

# Positive Gamma Exposure (light blue)
positive_gamma = dfAgg[dfAgg['TotalGamma'] >= 0]
fig1.add_trace(go.Bar(
    x=positive_gamma.index,
    y=positive_gamma['TotalGamma'],
    width=6,
    marker_color='#66B3FF',  # Light blue
    opacity=0.7,
    name='Positive Gamma'
))

# Negative Gamma Exposure (orange)
negative_gamma = dfAgg[dfAgg['TotalGamma'] < 0]
fig1.add_trace(go.Bar(
    x=negative_gamma.index,
    y=negative_gamma['TotalGamma'],
    width=6,
    marker_color='#FF8C00',  # Orange
    opacity=0.7,
    name='Negative Gamma'
))

fig1.add_vline(x=basePrice, line=dict(color='red', width=2, dash='dash'), annotation_text=f"{display_label}: {basePrice:,.0f}", annotation_position="top right")
fig1.update_layout(
    title=f"Total Gamma: ${df['TotalGamma'].sum():,.2f} Bn per 1% {display_label} Move",
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
    fig2.add_vline(x=basePrice, line=dict(color='blue', width=2, dash='dash'), annotation_text=f"{display_label}: {basePrice:,.0f}", annotation_position="top right")
    fig2.update_layout(
        title=f"Total Open Interest for {display_label}",
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
