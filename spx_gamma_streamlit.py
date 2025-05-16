import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta

st.set_page_config(page_icon="🧊", layout="wide")
pd.options.display.float_format = '{:,.4f}'.format

# Black-Scholes European-Options Gamma
def calcGammaEx(S, K, vol, T, r, q, optType, OI):
    if T <= 0 or vol <= 0:
        return 0
    dp = (np.log(S/K) + (r - q + 0.5*vol**2)*T) / (vol * np.sqrt(T))
    gamma = np.exp(-q*T) * norm.pdf(dp) / (S * vol * np.sqrt(T))
    return OI * 100 * S * S * 0.01 * gamma

# Detectar el tercer viernes de un mes
def isThirdFriday(d):
    return d.weekday() == 4 and 15 <= d.day <= 21

# Calcular el precio futuro de ES o NQ
def calculate_future_price(index_price, r, d):
    current_date = datetime.now()
    next_expiry = current_date + timedelta((4 - current_date.weekday()) % 7) + timedelta(weeks=2)
    T = (next_expiry - current_date).total_seconds() / (365.25 * 24 * 60 * 60)
    return index_price * np.exp((r - d) * T)

# Obtener datos de la API de CBOE
def fetch_data(index):
    try:
        response = requests.get(f"https://cdn.cboe.com/api/global/delayed_quotes/options/_{index}.json")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error al obtener datos de CBOE para {index}: {e}")
        return None

# Inicialización del estado de la sesión
if 'data' not in st.session_state:
    st.session_state['data'] = None

# Selección de índices y tasas
selection = st.selectbox("Seleccione el índice o conversión:", ["SPX", "NDX", "SPX=>ES", "NDX=>NQ"])
risk_free_rate = st.number_input("Tasa de interés libre de riesgo (% anual)", min_value=0.0, max_value=20.0, value=5.0)
dividend_yield = st.number_input("Tasa de dividendos (% anual)", min_value=0.0, max_value=20.0, value=1.5)

# Obtener datos si no están en la sesión
if st.session_state['data'] is None:
    data_json = fetch_data(selection.split('=>')[0] if '=>' in selection else selection)
    if data_json:
        st.session_state['data'] = data_json
else:
    data_json = st.session_state['data']

# Procesar datos
if data_json:
    spotPrice = data_json["data"].get("close", None)
    if spotPrice is None:
        st.error("No se pudo obtener el precio de cierre.")
        st.stop()

    futurePrice = calculate_future_price(spotPrice, risk_free_rate / 100, dividend_yield / 100)
    basePrice = futurePrice if "=>" in selection else spotPrice

    # Convertir datos JSON a DataFrame
    options_data = data_json["data"].get("options", [])
    data_df = pd.DataFrame(options_data)
    if data_df.empty:
        st.error("No hay datos de opciones disponibles.")
        st.stop()

    data_df['CallPut'] = data_df['option'].str[-9:-8]
    data_df['StrikePrice'] = data_df['option'].str[-8:-3].astype(float)
    data_df['ExpirationDate'] = pd.to_datetime(data_df['option'].str[-15:-9], format='%y%m%d')

    # Cálculo de Gamma Exposure
    current_date = datetime.now()
    data_df['daysTillExp'] = np.maximum((data_df['ExpirationDate'] - current_date).dt.total_seconds() / (365.25 * 24 * 60 * 60), 1e-5)

    data_df['Gamma'] = data_df.apply(lambda row: calcGammaEx(
        S=basePrice,
        K=row['StrikePrice'],
        vol=max(row.get('iv', 0.2), 0.01),
        T=row['daysTillExp'],
        r=risk_free_rate / 100,
        q=dividend_yield / 100,
        optType=row['CallPut'],
        OI=row.get('open_interest', 0)
    ), axis=1)

    # Gráficos de exposición gamma
    dfAgg = data_df.groupby('StrikePrice').agg({'Gamma': 'sum', 'open_interest': 'sum'})

    # Gráfico: Exposición Gamma
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dfAgg.index,
        y=dfAgg['Gamma'],
        marker_color='blue',
        name='Gamma Exposure'
    ))
    fig.add_vline(x=basePrice, line=dict(color='red', dash='dash'), annotation_text=f"Spot: {basePrice:,.0f}")
    fig.update_layout(
        title=f"Exposición Gamma ({selection})",
        xaxis_title="Strike Price",
        yaxis_title="Gamma Exposure",
        template="plotly_dark"
    )
    st.plotly_chart(fig)

    # Gráfico: Interés Abierto (Open Interest)
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=dfAgg.index,
        y=dfAgg['open_interest'],
        marker_color='green',
        name='Open Interest'
    ))
    fig2.update_layout(
        title=f"Open Interest ({selection})",
        xaxis_title="Strike Price",
        yaxis_title="Number of Contracts",
        template="plotly_dark"
    )
    st.plotly_chart(fig2)
