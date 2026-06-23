import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
import warnings

warnings.filterwarnings('ignore')

# ================= 1. 網頁基礎設定 =================
st.set_page_config(page_title="台股 ETF 趨勢預測分析", layout="wide", page_icon="📈")

# --- 注入自訂 CSS (側邊欄字體微縮與排版優化) ---
st.markdown("""
<style>
    /* 1. 縮小側邊欄文字、拉桿標籤、勾選框、提示框字體至 12px */
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stAlert p,
    [data-testid="stSidebar"] label p,
    [data-testid="stSidebar"] .stCheckbox p {
        font-size: 12px !important;
        line-height: 1.6 !important;
    }
    
    /* 2. 側邊欄的大標題保留大小(14px)以維持層次感 */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        font-size: 14px !important;
        font-weight: 700 !important;
    }
    
    /* 3. 調整按鈕內的文字大小 */
    [data-testid="stSidebar"] .stButton button p {
        font-size: 12px !important;
    }
    
    /* 4. 乾淨的主標題顏色 */
    h1, h2, h3 {
        color: #1E3A8A !important; 
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)

if 'searched' not in st.session_state:
    st.session_state.searched = False
if 'current_ticker' not in st.session_state:
    st.session_state.current_ticker = ""

def execute_search(input_ticker):
    if input_ticker:
        st.session_state.searched = True
        st.session_state.current_ticker = input_ticker.strip().upper()

# ================= 2. 介面邏輯：首頁 =================
if not st.session_state.searched:
    st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; font-size: 3.5rem;'>台股 ETF 趨勢預測分析</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray; font-size: 1.2rem;'>基於 Prophet 演算法的 ETF 趨勢分析工具</p><br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form(key='search_form'):
            search_col1, search_col2 = st.columns([4, 1])
            with search_col1:
                main_input = st.text_input("輸入代碼", placeholder="例如: 0050, 00878, 2330...", label_visibility="collapsed")
            with search_col2:
                submit_btn = st.form_submit_button("🔍 搜尋", use_container_width=True)
            if submit_btn and main_input:
                execute_search(main_input)
                st.rerun()

# ================= 3. 儀表板模式 =================
else:
    ticker = st.session_state.current_ticker
    st.markdown(f"<h1>📊 {ticker} 智能預測與風險分析看板</h1>", unsafe_allow_html=True)
    
    st.sidebar.header("🔍 ETF搜尋")
    with st.sidebar.form(key='sidebar_search_form'):
        new_input = st.text_input("輸入新代碼", ticker, label_visibility="collapsed")
        new_submit = st.form_submit_button("重新分析", use_container_width=True)
        if new_submit and new_input:
            execute_search(new_input)
            st.rerun()
            
    st.sidebar.markdown("---")
    
    ticker_query = f"{ticker}.TW" if ticker.isdigit() else ticker
    
    # 資料下載防呆機制
    base_data = pd.DataFrame()
    with st.spinner(f"正在從雲端數據庫解析 {ticker_query} 的市場特徵，請稍候..."):
        try:
            base_data = yf.download(ticker_query, period="max", progress=False)
        except Exception as e:
            pass
            
    if base_data.empty:
        st.error(f"🚨 無法獲取 **{ticker_query}** 的歷史資料！")
        st.warning("""
        👉 **可能原因：代碼錯誤 或 Yahoo Finance 流量管制。請等待 30 秒後重試。**
        """)
        if st.button("⬅️ 返回首頁重新搜尋", use_container_width=True):
            st.session_state.searched = False
            st.rerun()
        st.stop()
            
    base_data = base_data.reset_index()
    if isinstance(base_data.columns, pd.MultiIndex):
        base_data.columns = [col[0] for col in base_data.columns]
    base_data['Date'] = pd.to_datetime(base_data['Date']).dt.tz_localize(None)
    
    # --- 背景量化指標計算 ---
    daily_returns_calc = base_data['Close'].pct_change().dropna()
    calculated_volatility = daily_returns_calc.std() * np.sqrt(252) * 100
    
    delta = base_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1]
    
    # --- 🚀 專業量化模型判定邏輯 (極致純淨數據版) ---
    if calculated_volatility > 18.0 or current_rsi > 70:
        model_name = "短線動能追蹤模型"
        auto_trend, auto_hist, auto_blind, auto_week = 0.20, 2, 30, True
        reason_text = (
            f"▪ <b>年化歷史波動率：</b>{calculated_volatility:.1f}%<br>"
            f"▪ <b>RSI 相對強弱指標：</b>{current_rsi:.1f}"
        )
    else:
        model_name = "長線均值回歸模型"
        auto_trend, auto_hist, auto_blind, auto_week = 0.05, 5, 90, False
        reason_text = (
            f"▪ <b>年化歷史波動率：</b>{calculated_volatility:.1f}%<br>"
            f"▪ <b>RSI 相對強弱指標：</b>{current_rsi:.1f}"
        )

    # --- 側邊欄：智能建議與自由調整 ---
    st.sidebar.header("🧠 參數微調 (載入智能預設)")
    
    # 使用自訂 HTML，呈現極致專業的數據戰情卡片
    st.sidebar.markdown(f"""
    <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 14px; border-radius: 6px; margin-bottom: 20px; border-left: 4px solid #1E3A8A;">
        <div style="font-size: 13px; font-weight: bold; color: #1E3A8A; margin-bottom: 8px;">
            ⚙️ 目前掛載：{model_name}
        </div>
        <div style="font-size: 12px; color: #475569; line-height: 1.8;">
            {reason_text}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    trend_flexibility = st.sidebar.slider("📈 趨勢敏感度 (Changepoint)", 0.01, 0.50, auto_trend, step=0.01)
    history_years = st.sidebar.slider("📚 訓練資料區間 (年)", 1, 10, auto_hist)
    blind_test_days = st.sidebar.slider("📚 驗證資料區間 (天)", 15, 180, auto_blind)
    predict_days = st.sidebar.slider("📚 未來推演區間 (天)", 30, 180, 90)
    week_season = st.sidebar.checkbox("開啟週間季節性 (捕捉短期波動)", value=auto_week)

    # 資料裁切
    start_date_filter = base_data['Date'].max() - pd.DateOffset(years=history_years)
    data = base_data[base_data['Date'] >= start_date_filter]
    
    max_date = data['Date'].max()
    min_date = data['Date'].min()
    cutoff_date = max_date - pd.Timedelta(days=blind_test_days)
    
    # 訓練模型
    prophet_df = data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    train_df = prophet_df[prophet_df['ds'] <= cutoff_date]
    test_df = prophet_df[prophet_df['ds'] > cutoff_date]
    
    training_days = (cutoff_date - min_date).days
    yearly_seasonality_setting = True if training_days >= 365 else False
    
    model = Prophet(changepoint_prior_scale=trend_flexibility, yearly_seasonality=yearly_seasonality_setting, weekly_seasonality=week_season, daily_seasonality=False)
    model.fit(train_df)
    future = model.make_future_dataframe(periods=blind_test_days + predict_days)
    forecast = model.predict(future)
    
    # 計算 MAPE
    merged = pd.merge(test_df, forecast[['ds', 'yhat']], on='ds', how='inner')
    mape = np.mean(np.abs((merged['y'] - merged['yhat']) / merged['y'])) * 100 if not merged.empty else 0

    # ================= 回歸經典 4 大 KPI =================
    current_price = data['Close'].iloc[-1]
    yearly_return = ((current_price - data['Close'].iloc[-252]) / data['Close'].iloc[-252]) * 100 if len(data) > 252 else 0
    rolling_max = data['Close'].cummax()
    max_drawdown = (data['Close'] / rolling_max - 1.0).min() * 100
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("最新收盤價 (TWD)", f"{current_price:.2f}")
    col2.metric("近一年總報酬率", f"{yearly_return:.2f}%")
    col3.metric("年化歷史波動率", f"{calculated_volatility:.2f}%")
    col4.metric(f"最大歷史回撤 ({history_years}年內)", f"{max_drawdown:.2f}%")
    st.markdown("<br>", unsafe_allow_html=True)

    # ================= 1️⃣ 歷史規律與週期拆解 =================
    st.markdown("<h3>1️⃣ 歷史規律與週期拆解</h3>", unsafe_allow_html=True)
    fig_comp = model.plot_components(forecast)
    axes = fig_comp.get_axes()
    axes[0].set_title("Long-term Fundamental Trend", fontsize=12, fontweight='bold')
    for ax in axes[1:]:
        if 'Day of week' in ax.get_xlabel(): ax.set_title("Weekly Trading Sentiment", fontsize=12, fontweight='bold')
        elif 'Day of year' in ax.get_xlabel(): ax.set_title("Annual Cycle & Dividend Seasonality", fontsize=12, fontweight='bold')
    fig_comp.tight_layout(pad=2.0)
    st.pyplot(fig_comp)
    st.markdown("---")

    # ================= 2️⃣ AI 預測趨勢與盲測對比 =================
    st.markdown("<h3>2️⃣ 預測趨勢圖</h3>", unsafe_allow_html=True)
    
    if mape < 5:
        st.success(f"🎯 **分析準確度極高！** 歷史軌跡完美契合，平均誤差 (MAPE) 僅 {mape:.2f}%")
    elif mape < 10:
        st.info(f"📊 **分析準確度良好。** 趨勢符合模型預期，平均誤差 (MAPE) 為 {mape:.2f}%")
    else:
        st.warning(f"⚠️ **分析結果落差顯著 (MAPE: {mape:.2f}%)**。實際價格突破長線預測，顯示近期發生極端溢價或資金狂熱。")

    fig_interactive = plot_plotly(model, forecast)
    for trace in fig_interactive['data']:
        if trace.name:
            trace.name = trace.name.replace('Actual', '實際價格').replace('Predicted', '模型預測價格').replace('yhat_lower', '下限').replace('yhat_upper', '上限')
    
    fig_interactive.add_trace(go.Scatter(
        x=test_df['ds'], y=test_df['y'], mode='markers', 
        marker=dict(color='#DC2626', size=5, symbol='circle'), 
        name='實際價格', hovertemplate='<b>實際價格</b>: %{y:.2f}<extra></extra>'
    ))
    
    fig_interactive.update_layout(
        hovermode="x unified", 
        hoverdistance=1,     # 縮小滑鼠磁鐵的吸附半徑 (避免吸到隔天的黑點)
        spikedistance=1,     # 縮小垂直參考線的判定範圍
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig_interactive, use_container_width=True)
    st.markdown("---")

    # ================= 3️⃣ 未來預測數值總表 =================
    st.markdown("<h3>3️⃣ 未來預測數值總表</h3>", unsafe_allow_html=True)
    true_future_start = data['Date'].max()
    future_table = forecast[forecast['ds'] > true_future_start][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    if not future_table.empty:
        future_table.columns = ['日期', '預測價格', '下限 (Lower)', '上限 (Upper)']
        future_table['日期'] = future_table['日期'].dt.date
        st.dataframe(future_table.set_index('日期'), use_container_width=True)
