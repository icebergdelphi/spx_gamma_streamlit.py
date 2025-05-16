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

# Función para calcular el precio estimado del ES o NQ
def calculate_futures_price(index_price, r, d, is_ndx=False):
    current_date = datetime.now()
    # Asumimos que el próximo vencimiento es el tercer viernes de junio 2025 (20/06/2025)
    next_expiry = datetime(2025, 6, 20, 16, 0)  # Cierre del mercado
    T = (next_expiry - current_date).total_seconds() / (365.25 * 24 * 60 * 60)
    dividend_yield = 0.015 if not is_ndx else 0.01  # 1.5% para SPX, 1% para NDX
    futures_price = index_price * np.exp((r - dividend_yield) * T)
    return futures_price

# Inicialización en el estado de la sesión
if 'last_request_time' not in st.session_state:
    st.session_state['last_request_time'] = datetime.now()
if 'data' not in st.session_state:
    st.session_state['data'] = None

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

# Verificar si han pasado 5 minutos (300 segundos) desde el último request
current_time = datetime.now()
if (current_time - st.session_state['last_request_time']).total_seconds() > 300 or st.session_state['data'] is None:
    st.session_state['last_request_time'] = current_time
    index = st.selectbox("Seleccione el índice:", ["SPX", "NDX", "SPX=>ES", "NDX=>NQ"], key="index")
    data_json = fetch_data(index.split("=>")[0] if "=>" in index else index)
    if data_json:
        st.session_state['data'] = data_json
else:
    st.write("Esperando la siguiente actualización en menos de 5 minutos...")
    index = st.session_state.get('index', 'SPX')  # Usar el último índice seleccionado si existe
    data_json = st.session_state['data']

# Campo editable para la tasa de interés libre de riesgo
r_rate = st.number_input("Tasa de interés libre de riesgo (% anual)", min_value=0.0, max_value=10.0, value=4.0, step=0.1) / 100

# Procesar datos si existen
if data_json:
    spotPrice = data_json["data"]["close"]
    futuresPrice = calculate_futures_price(spotPrice, r_rate, 0.015, is_ndx="NDX" in index or "NQ" in index)
    options_data = data_json["data"]["options"]

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
    # To further convert into 'per 1% move' quantity, multiply by 1% of spotPrice
    df['CallGEX'] = df['CallGamma'] * df['CallOpenInt'] * 100 * spotPrice * spotPrice * 0.01
    df['PutGEX'] = df['PutGamma'] * df['PutOpenInt'] * 100 * spotPrice * spotPrice * 0.01 * -1

    df['TotalGamma'] = (df['CallGEX'] + df['PutGEX']) / 10**9
    dfAgg = df.groupby(['StrikePrice']).sum(numeric_only=True)

    # Rango de strikes
    fromStrike = 0.8 * spotPrice
    toStrike = 1.2 * spotPrice

    # Determinar qué gráficos mostrar según la selección
    if "=>" in index:
        base_index = index.split("=>")[0]
        futures_name = index.split("=>")[1]
        title_prefix = f"Total {futures_name} Gamma: "
        spot_label = f"{base_index} Spot"
        futures_label = f"{futures_name} Est."
        futures_color = 'blue' if futures_name == 'ES' else 'green'
    else:
        title_prefix = f"Total {index} Gamma: "
        spot_label = f"{index} Spot"
        futures_label = None
        futures_color = None

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
    fig1.add_vline(x=spotPrice, line=dict(color='red', width=2, dash='dash'), annotation_text=f"{spot_label}: {spotPrice:,.0f}", annotation_position="top right")
    if futures_label:
        fig1.add_vline(x=futuresPrice, line=dict(color=futures_color, width=2, dash='dash'), annotation_text=f"{futures_label}: {futuresPrice:,.2f}", annotation_position="top left")
    fig1.update_layout(
        title=f"{title_prefix}${{df['TotalGamma'].sum():,.2f}} Bn per 1% {index} Move" + (f" ({spot_label}={spotPrice:,.0f}, {futures_label}={futuresPrice:,.2f})" if futures_label else ""),
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
    fig2.add_vline(x=spotPrice, line=dict(color='blue', width=2, dash='dash'), annotation_text=f"{spot_label}: {spotPrice:,.0f}", annotation_position="top right")
    if futures_label:
        fig2.add_vline(x=futuresPrice, line=dict(color=futures_color, width=2, dash='dash'), annotation_text=f"{futures_label}: {futuresPrice:,.2f}", annotation_position="top left")
    fig2.update_layout(
        title=f"Total
