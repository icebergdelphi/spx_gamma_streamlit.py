import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import time

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


# Función para determinar la fecha de expiración de futuros ES/NQ
def get_next_expiry_date():
    """
    Calcula la próxima fecha de expiración de los futuros E-mini S&P 500 (ES) y Nasdaq-100 (NQ).
    Los futuros expiran el tercer viernes de marzo, junio, septiembre y diciembre.
    
    Returns:
        datetime: Fecha y hora de expiración (cierre del mercado a las 16:00)
    """
    current_date = datetime.now()
    
    # Meses de expiración: marzo (3), junio (6), septiembre (9), diciembre (12)
    expiry_months = [3, 6, 9, 12]
    
    # Encontrar el próximo mes de expiración
    current_month = current_date.month
    current_year = current_date.year
    
    next_expiry_month = None
    next_expiry_year = current_year
    
    # Determinar el próximo mes de expiración
    for month in expiry_months:
        if month > current_month:
            next_expiry_month = month
            break
    
    # Si estamos después de diciembre, ir al próximo año
    if next_expiry_month is None:
        next_expiry_month = expiry_months[0]  # Marzo
        next_expiry_year = current_year + 1
    
    # Encontrar el tercer viernes del mes de expiración
    first_day = datetime(next_expiry_year, next_expiry_month, 1)
    
    # Calcular cuántos días hay que añadir para llegar al primer viernes
    days_until_friday = (4 - first_day.weekday()) % 7
    first_friday = first_day + timedelta(days=days_until_friday)
    
    # El tercer viernes es 14 días después del primer viernes
    third_friday = first_friday + timedelta(days=14)
    
    # Si la fecha actual es posterior al tercer viernes de este mes de expiración,
    # pasar al siguiente ciclo de expiración
    if current_date.date() > third_friday.date() and current_date.month == next_expiry_month:
        # Encontrar el siguiente mes de expiración
        next_index = (expiry_months.index(next_expiry_month) + 1) % len(expiry_months)
        next_expiry_month = expiry_months[next_index]
        if next_index == 0:  # Si volvemos a marzo, incrementar el año
            next_expiry_year += 1
            
        # Recalcular el tercer viernes
        first_day = datetime(next_expiry_year, next_expiry_month, 1)
        days_until_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day + timedelta(days=days_until_friday)
        third_friday = first_friday + timedelta(days=14)
    
    # Establecer la hora de cierre del mercado (16:00)
    expiry_datetime = datetime(third_friday.year, third_friday.month, third_friday.day, 16, 0)
    
    return expiry_datetime




# Función para calcular el precio estimado de ES o NQ
def calculate_future_price(index_price, r=0.04, d=0.015, index="SPX"):
    """
    Calcula el precio estimado del futuro ES o NQ basado en el precio del índice subyacente.
    Utiliza la fórmula de costo de carry: F = S * e^((r-d)*T)
    
    Args:
        index_price (float): Precio actual del índice (SPX o NDX)
        r (float): Tasa de interés libre de riesgo anualizada (decimal)
        d (float): Rendimiento de dividendos anualizado (decimal)
        index (str): Nombre del índice ("SPX" o "NDX")
        
    Returns:
        float: Precio estimado del futuro
    """
    current_date = datetime.now()
    next_expiry = get_next_expiry_date()
    
    # Calcular la fracción de año hasta la expiración
    T = (next_expiry - current_date).total_seconds() / (365.25 * 24 * 60 * 60)
    
    # Si estamos muy cerca de la expiración, usar un valor mínimo para evitar errores
    T = max(T, 0.001)
    
    # Aplicar fórmula de precio de futuros
    future_price = index_price * np.exp((r - d) * T)
    
    return future_price

# Función para mostrar información de expiración
def display_expiry_info():
    """
    Muestra información sobre la próxima fecha de expiración de futuros.
    """
    next_expiry = get_next_expiry_date()
    expiry_str = next_expiry.strftime("%A, %d de %B de %Y a las %H:%M")
    
    days_to_expiry = (next_expiry - datetime.now()).days
    hours_to_expiry = int((next_expiry - datetime.now()).total_seconds() / 3600) % 24
    
    st.sidebar.markdown("### Información de Expiración")
    st.sidebar.markdown(f"**Próxima fecha de expiración:**  \n{expiry_str}")
    st.sidebar.markdown(f"**Tiempo hasta expiración:**  \n{days_to_expiry} días y {hours_to_expiry} horas")
    
    # Opcional: Mostrar ciclo completo de expiración
    with st.sidebar.expander("Ver ciclo de expiración"):
        current_year = datetime.now().year
        
        st.markdown("#### Ciclo de expiración de futuros")
        for year in range(current_year, current_year + 2):
            st.markdown(f"**{year}**")
            for month in [3, 6, 9, 12]:
                first_day = datetime(year, month, 1)
                days_until_friday = (4 - first_day.weekday()) % 7
                first_friday = first_day + timedelta(days=days_until_friday)
                third_friday = first_friday + timedelta(days=14)
                expiry_date = datetime(third_friday.year, third_friday.month, third_friday.day, 16, 0)
                
                if expiry_date > datetime.now():
                    st.markdown(f"- {expiry_date.strftime('%d de %B')}: **Próximo**" if expiry_date == get_next_expiry_date() else f"- {expiry_date.strftime('%d de %B')}")

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
bar_width = st.slider("Bar Width for Charts", min_value=5, max_value=20, value=6, step=1)
# Añadir información de expiración
display_expiry_info()
# Toggle for auto-refresh
auto_refresh = st.checkbox("Enable Auto-Refresh (every 5 minutes)", value=False)

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
    # Placeholder para el timer dinámico
    timer_placeholder = st.empty()
    # Simular actualización del timer durante 10 iteraciones (10 segundos)
    for _ in range(10):
        current_time = datetime.now()
        remaining_seconds = int(300 - (current_time - st.session_state['last_request_time']).total_seconds())
        remaining_minutes = remaining_seconds // 60
        remaining_seconds = remaining_seconds % 60
        timer_placeholder.write(f"Next update in: {remaining_minutes:02d}:{remaining_seconds:02d}")
        time.sleep(1)  # Esperar 1 segundo antes de la próxima actualización
    data_json = st.session_state['data']

# Auto-refresh logic
if auto_refresh:
    time.sleep(5)  # Short sleep to avoid excessive CPU usage
    if (datetime.now() - st.session_state['last_request_time']).total_seconds() > 300:
        st.experimental_rerun()  # Rerun the app to fetch new data

# Procesar datos si existen
if data_json:
    spotPrice = data_json["data"]["close"]
    futurePrice = calculate_future_price(spotPrice, r=risk_free_rate/100, d=0.015, index=selection.split('=>')[0] if '=>' in selection else selection)
    options_data = data_json["data"]["options"]

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

    df['CallIV'] = df['CallIV'].replace(0.0, 0.2)
    df['PutIV'] = df['PutIV'].replace(0.0, 0.2)

    current_date = datetime.now()
    df['daysTillExp'] = (df['ExpirationDate'] - current_date).dt.total_seconds() / (365.25 * 24 * 60 * 60)
    df['daysTillExp'] = df['daysTillExp'].clip(lower=1e-5)

    df['CallGEX'] = df['CallGamma'] * df['CallOpenInt'] * 100 * basePrice * basePrice * 0.01
    df['PutGEX'] = df['PutGamma'] * df['PutOpenInt'] * 100 * basePrice * basePrice * 0.01 * -1
    df['TotalGamma'] = (df['CallGEX'] + df['PutGEX']) / 10**9
    dfAgg = df.groupby(['StrikePrice']).sum(numeric_only=True)

    fromStrike = 0.8 * basePrice
    toStrike = 1.2 * basePrice

    # Checkbox para el Gráfico 1
    show_strike_annotations_chart1 = st.checkbox("Show Important Strike Prices on Gamma Chart", value=False)

    # Chart 1: Absolute Gamma Exposure
    fig1 = go.Figure()
    positive_gamma = dfAgg[dfAgg['TotalGamma'] >= 0]
    fig1.add_trace(go.Bar(
        x=positive_gamma.index,
        y=positive_gamma['TotalGamma'],
        width=bar_width,
        marker_color='#66B3FF',
        opacity=0.7,
        name='Positive Gamma'
    ))
    negative_gamma = dfAgg[dfAgg['TotalGamma'] < 0]
    fig1.add_trace(go.Bar(
        x=negative_gamma.index,
        y=negative_gamma['TotalGamma'],
        width=bar_width,
        marker_color='#FF8C00',
        opacity=0.7,
        name='Negative Gamma'
    ))
    fig1.add_vline(x=basePrice, line=dict(color='red', width=2, dash='dash'), annotation_text=f"{display_label}: {basePrice:,.0f}", annotation_position="top right")
    
    # Conditionally show annotations for Chart 1
    if show_strike_annotations_chart1:
        top_positive = positive_gamma.nlargest(5, 'TotalGamma')
        for strike in top_positive.index:
            fig1.add_annotation(
                x=strike,
                y=top_positive.loc[strike, 'TotalGamma'],
                text=f"{int(strike)}",
                showarrow=True,
                arrowhead=1,
                yshift=10,
                font=dict(color="white")
            )
        
        top_negative = negative_gamma.nsmallest(5, 'TotalGamma')
        for strike in top_negative.index:
            fig1.add_annotation(
                x=strike,
                y=top_negative.loc[strike, 'TotalGamma'],
                text=f"{int(strike)}",
                showarrow=True,
                arrowhead=1,
                yshift=-10,
                font=dict(color="white")
            )

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

    # Checkbox para el Gráfico 2
    show_strike_annotations_chart2 = st.checkbox("Show Important Strike Prices on Open Interest Chart", value=False)

    # Chart 2: Open Interest by Calls and Puts
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=dfAgg.index,
        y=dfAgg['CallOpenInt'],
        width=bar_width,
        marker_color='green',
        opacity=0.5,
        name='Call OI'
    ))
    fig2.add_trace(go.Bar(
        x=dfAgg.index,
        y=-1 * dfAgg['PutOpenInt'],
        width=bar_width,
        marker_color='red',
        opacity=0.5,
        name='Put OI'
    ))
    top_calls = dfAgg.nlargest(5, 'CallOpenInt')
    fig2.add_trace(go.Bar(
        x=top_calls.index,
        y=top_calls['CallOpenInt'],
        width=bar_width,
        marker_color='#00FF00',
        opacity=1.0,
        name='Top Call OI',
        showlegend=False
    ))
    top_puts = dfAgg.nlargest(5, 'PutOpenInt')
    fig2.add_trace(go.Bar(
        x=top_puts.index,
        y=-1 * top_puts['PutOpenInt'],
        width=bar_width,
        marker_color='#FF0000',
        opacity=1.0,
        name='Top Put OI',
        showlegend=False
    ))
    
    # Conditionally show annotations for Chart 2
    if show_strike_annotations_chart2:
        for strike in top_calls.index:
            fig2.add_annotation(
                x=strike,
                y=top_calls.loc[strike, 'CallOpenInt'],
                text=f"{int(strike)}",
                showarrow=True,
                arrowhead=1,
                yshift=10,
                font=dict(color="white")
            )
        for strike in top_puts.index:
            fig2.add_annotation(
                x=strike,
                y=-1 * top_puts.loc[strike, 'PutOpenInt'],
                text=f"{int(strike)}",
                showarrow=True,
                arrowhead=1,
                yshift=-10,
                font=dict(color="white")
            )

    fig2.add_vline(x=basePrice, line=dict(color='blue', width=2, dash='dash'), annotation_text=f"{display_label}: {basePrice:,.0f}", annotation_position="top right")
    fig2.update_layout(
        title=f"Total Open Interest for {display_label}",
        xaxis_title="Strike",
        yaxis_title="Open Interest (number of contracts)",
        xaxis=dict(range=[fromStrike, toStrike], tickformat=",", automargin=True),
        yaxis=dict(tickformat=".0f", automargin=True),
        showlegend=True,
        template="plotly_dark",
        font=dict(size=14),
        margin=dict(l=20, r=20, t=50, b=50),
        width=1200,
        height=500,
        barmode='overlay'
    )
    st.plotly_chart(fig2, use_container_width=True)

# Gráfico 3

# Preparar datos para el top 5 de calls y puts
# Ordenar por Open Interest para identificar los más importantes
top_calls_df = dfAgg.nlargest(5, 'CallOpenInt').reset_index()
top_puts_df = dfAgg.nlargest(5, 'PutOpenInt').reset_index()

# Añadir más información a los top calls y puts
# Para calls, recuperamos último precio de venta
top_calls_df['LastSale'] = top_calls_df['StrikePrice'].apply(
    lambda strike: df[df['StrikePrice'] == strike]['CallLastSale'].iloc[0] 
    if not df[df['StrikePrice'] == strike].empty and len(df[df['StrikePrice'] == strike]['CallLastSale']) > 0 
    else 0
)

# Para puts, recuperamos último precio de venta
top_puts_df['LastSale'] = top_puts_df['StrikePrice'].apply(
    lambda strike: df[df['StrikePrice'] == strike]['PutLastSale'].iloc[0] 
    if not df[df['StrikePrice'] == strike].empty and len(df[df['StrikePrice'] == strike]['PutLastSale']) > 0 
    else 0
)

# Crear un layout con dos columnas para mostrar los top 5 calls y puts
st.subheader(f"Top 5 Strikes de Interés para {display_label}")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Top 5 Calls")
    call_data = {
        "Strike": top_calls_df['StrikePrice'].tolist(),
        "Open Interest": top_calls_df['CallOpenInt'].tolist(),
        "Last Sale": top_calls_df['LastSale'].tolist(),
    }
    call_table = pd.DataFrame(call_data)
    # Formatear para mostrar números enteros para Strike y OI, y formato de precio para Last Sale
    call_table['Strike'] = call_table['Strike'].apply(lambda x: f"{int(x):,}")
    call_table['Open Interest'] = call_table['Open Interest'].apply(lambda x: f"{int(x):,}")
    call_table['Last Sale'] = call_table['Last Sale'].apply(lambda x: f"${x:.2f}")
    st.table(call_table)

with col2:
    st.markdown("### Top 5 Puts")
    put_data = {
        "Strike": top_puts_df['StrikePrice'].tolist(),
        "Open Interest": top_puts_df['PutOpenInt'].tolist(),
        "Last Sale": top_puts_df['LastSale'].tolist(),
    }
    put_table = pd.DataFrame(put_data)
    # Formatear para mostrar números enteros para Strike y OI, y formato de precio para Last Sale
    put_table['Strike'] = put_table['Strike'].apply(lambda x: f"{int(x):,}")
    put_table['Open Interest'] = put_table['Open Interest'].apply(lambda x: f"{int(x):,}")
    put_table['Last Sale'] = put_table['Last Sale'].apply(lambda x: f"${x:.2f}")
    st.table(put_table)

# También puedes crear un gráfico visual de estos datos
show_strike_annotations_chart3 = st.checkbox("Show Top Strikes Visual Comparison", value=False)

if show_strike_annotations_chart3:
    # Crear un gráfico visual combinado
    fig3 = go.Figure()
    
    # Usar un gráfico de barras horizontales para los calls
    fig3.add_trace(go.Bar(
        y=[f"Call {i+1}" for i in range(len(top_calls_df))],
        x=top_calls_df['CallOpenInt'],
        orientation='h',
        name='Call OI',
        marker_color='green',
        text=[f"Strike: {int(x)}<br>Last Sale: ${y:.2f}" for x, y in zip(top_calls_df['StrikePrice'], top_calls_df['LastSale'])],
        hoverinfo='text+x',
        opacity=0.7
    ))
    
    # Usar un gráfico de barras horizontales para los puts
    fig3.add_trace(go.Bar(
        y=[f"Put {i+1}" for i in range(len(top_puts_df))],
        x=top_puts_df['PutOpenInt'],
        orientation='h',
        name='Put OI',
        marker_color='red',
        text=[f"Strike: {int(x)}<br>Last Sale: ${y:.2f}" for x, y in zip(top_puts_df['StrikePrice'], top_puts_df['LastSale'])],
        hoverinfo='text+x',
        opacity=0.7
    ))
    
    fig3.update_layout(
        title=f"Top 5 Strikes by Open Interest - {display_label}",
        xaxis_title="Open Interest (number of contracts)",
        yaxis=dict(automargin=True),
        template="plotly_dark",
        font=dict(size=14),
        margin=dict(l=20, r=20, t=50, b=50),
        width=1200,
        height=500,
        barmode='group'
    )
    
    st.plotly_chart(fig3, use_container_width=True)
