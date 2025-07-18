
# 🔧 Francis-99 Fault Detection System

A **Streamlit-based AI web application** that detects early-stage faults in Francis-99 turbine blades using sensor data and a hybrid CNN-LSTM deep learning model. It also provides stage-specific visual feedback and expert-level interpretation using a large language model (LLM).

---

## 🚀 Features

- ✅ Predicts **fault stage (1–9)** from 5 turbine sensor values.
- 📉 Uses **CNN+LSTM deep learning model** trained on time-series vibration & pressure data.
- 📈 Scales input data using **RobustScaler** for improved prediction stability.
- 🔍 Shows **cropped image** of crack stage from `crackimage.jpg`.
- 🤖 Provides **LLM-based expert interpretation** using **Groq + LLaMA 3** API.

---

## 📦 File Structure

```
.
├── app.py                     # Main Streamlit application
├── cnn+LSTM_fault_model.h5   # Trained hybrid CNN+LSTM model
├── robust_scaler.pkl         # Pre-fitted RobustScaler for input normalization
├── crackimage.jpg            # Master image with 10 fault stages (5x2 layout)
├── .env                      # Contains GROQ_API_KEY (not committed)
├── requirements.txt          # Python dependencies
└── README.md                 # You're reading it!
```

---

## 🧠 Model Overview

- **Input shape**: `(30, 5)` → 30 time steps of 5 sensors
- **Sensors used**:
  - `PDT1` – Pressure downstream 1
  - `PGV2` – Guide vane vibration 2
  - `PDT3` – Pressure downstream 3
  - `ATB1` – Axial turbine bearing 1
  - `ATB2` – Axial turbine bearing 2

---

## 🖥️ Running Locally

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/francis99-fault-detection.git
cd francis99-fault-detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Add `.env` file
Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Run Streamlit App
```bash
streamlit run app.py
```

---

## 🌐 Deployment

This app can be hosted for free using **[Streamlit Community Cloud](https://streamlit.io/cloud)**:

1. Push the project to GitHub.
2. Go to [Streamlit Cloud](https://streamlit.io/cloud) and log in.
3. Click **“New App”** and connect your GitHub repo.
4. Set **GROQ_API_KEY** in **App > Settings > Secrets**.
5. Click **Deploy**.

---

## 🧠 Expert Interpretation via LLM

Uses Groq-hosted **LLaMA 3-70B** model to return JSON-formatted expert feedback such as:
- Crack length in mm
- Blade damage condition
- Recommended actions

---

## 📸 Visual Output

Each predicted stage corresponds to a **cropped slice** of a `crackimage.jpg` that contains 10 fault stages arranged in a 5x2 grid.

---

## 📋 Sample Output

```json
{
  "stage": "Stage 4 – Moderate crack growth",
  "crack_length": "72 mm",
  "description": "Crack is progressing and may affect structural integrity. Still attached but growing.",
  "recommendation": "Schedule inspection and prepare for potential maintenance."
}
```

---

## 📚 Requirements

```
streamlit
numpy
Pillow
tensorflow
python-dotenv
joblib
groq
```

---

## 📜 License

This project is open-source for research and educational use.

---

## 🙌 Acknowledgments

- Based on data from **Francis-99 Workshop**
- Visual references from simulation dataset
- Model inference accelerated via **Groq LPU Inference**
