# 📈 AI 智能量化預測終端 (AI Quant Prediction Terminal)

> **結合動態特徵提取與時間序列演算法，為台股與 ETF 提供機構級趨勢洞察的 Web 應用程式。**

本專案旨在解決散戶投資人常因新聞炒作而追高殺低之痛點。系統透過嚴謹的量化演算法與歷史數據回測，為使用者提供客觀的歷史規律拆解、未來價格推演，以及極端溢價的風險警示。

---

## ✨ 核心特色 (Core Features)

* **🧠 智能特徵提取與分類 (Heuristic Classification)**
  系統不依賴人為主觀判斷，於背景極速計算資產的 **「年化歷史波動率」** 與 **「RSI 相對強弱指標」**。依據量化特徵，自動為該標的掛載最適配的參數（切換「短線動能追蹤模型」或「長線均值回歸模型」）。
* **📊 機器學習時間序列預測 (Time-Series Forecasting)**
  底層搭載 Facebook 開源之 Prophet 預測引擎，能自動拆解出資產的「長期基本面趨勢」與「週間/年度季節性週期」。
* **🎯 嚴謹的盲測驗證 (Blind Testing & MAPE)**
  為防止模型過度擬合 (Overfitting)，系統強制將末端資料保留為「盲測區間」，並實時計算 **MAPE (平均絕對百分比誤差)**，拒絕未來函數作弊，還原最真實的預測準確度。
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
