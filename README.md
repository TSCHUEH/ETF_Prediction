# 📈 AI 智能量化預測終端 (AI Quant Prediction Terminal)

> **結合動態特徵提取與時間序列演算法，為台股與 ETF 提供機構級趨勢洞察的 Web 應用程式。**

本專題旨在運用預測模型拆解 ETF 歷史規律，並透過視覺化決策面板呈現合理價格區間，藉此排除主觀偏誤，為進場時機提供客觀的量化依據。
> https://etfprediction-azeyw6n3qn4kgq6wyjgumd.streamlit.app/

---

## ✨ 核心特色 (Core Features)

* **🧠 智能特徵提取與分類 (Heuristic Classification)**
  系統不依賴人為主觀判斷，於背景極速計算資產的 **「年化歷史波動率」** 與 **「RSI 相對強弱指標」**。依據量化特徵，自動為該標的掛載最適配的參數（切換「短線動能追蹤模型」或「長線均值回歸模型」）。
* **📊 機器學習時間序列預測 (Time-Series Forecasting)**
  底層搭載 Facebook 開源之 Prophet 預測工具，能自動拆解出資產的「長期基本面趨勢」與「週間/年度季節性週期」。
* **🎯 嚴謹的盲測驗證 (Blind Testing & MAPE)**
  為防止模型過度擬合 (Overfitting)，系統強制將末端資料保留為「驗證區間」，並實時計算 **MAPE (平均絕對百分比誤差)**，拒絕未來函數作弊，還原最真實的預測準確度。
* **💻 機構級互動終端 (Professional UI/UX)**
  採用 Streamlit 與客製化 CSS 打造極簡純淨的戰情面板，結合 Plotly 互動式圖表，提供像素級精準的數據追蹤，完美適應深色/淺色模式。

---

## 🛠️ 技術堆疊 (Tech Stack)

* **前端介面:** `Streamlit`
* **資料獲取:** `yfinance` (Yahoo Finance API)
* **資料處理:** `Pandas`, `NumPy`
* **演算法模型:** `Prophet`
* **資料視覺化:** `Plotly`, `Matplotlib`

---

## 🚀 快速啟動 (Quick Start)

### 1. 安裝環境與依賴套件
請確保您的電腦已安裝 Python 3.8 以上版本。在終端機 (Terminal) 中執行以下指令以安裝所需套件：

```bash
pip install -r requirements.txt
