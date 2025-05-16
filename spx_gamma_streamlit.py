import streamlit as st
import pandas as pd
import numpy as np
import scipy
from scipy.stats import norm
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta, date


st.set_page_config( 
   page_icon="🧊",
   layout="wide", 
)

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
    index = st.selectbox("Seleccione el índice:", ["SPX", "NDX"], key="index")
    data_json = fetch_data(index)
    if data_json:
        st.session_state['data'] = data_json
else:
    st.write("Esperando la siguiente actualización en menos de 5 minutos...")
    index = st.session_state.get('index', 'SPX')  # Usar el último índice seleccionado si existe
    data_json = st.session_state['data']

# Procesar datos si existen
if data_json:
    spotPrice = data_json["data"]["close"]
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
    df['PutGEX'] = df['
