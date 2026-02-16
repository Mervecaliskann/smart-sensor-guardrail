# 🛡️ Smart Sensor Data Quality Guardrail

A professional data validation system for smart factories. This tool checks industrial sensor data (Temperature, Pressure, Vibration) before it goes to AI models. It helps meet **"Data Validation"** requirements for Industry 4.0.

## 🚀 Project Overview
In factories, sensors can sometimes give wrong data (like a sudden 180°C jump). If this "dirty" data goes into an AI model, the model will make wrong decisions. This project creates a "Guardrail" (safety layer) to find these errors using statistics and explain them using AI.

---

## 📊 System Dashboard & AI Insights

The system monitors data in real-time and creates technical reports for engineers.

### 1. Statistical Error Detection
*The chart below shows a fake 180°C spike. The system successfully found this as a "Physical Range Violation" and a "Z-Score Outlier".*

![Dashboard Graph](assets/dashboard_preview.png)

### 2. AI-Powered Quality Report
*The AI (GPT-4o-mini) analyzes the errors and explains the risks (like Bias or Overfitting). It classifies the data as "UNSAFE" for the model.*

![AI Diagnostic Report](assets/dashboard_preview2.png)

---

## 🧠 Statistical Logic
The system uses a mathematical method called the **Z-Score** to find anomalies that do not follow the normal distribution.

**The Formula:**

$$Z = \frac{x - \mu}{\sigma}$$

* **x**: The raw sensor reading.
* **μ (mu)**: The average (mean) of the data.
* **σ (sigma)**: The standard deviation.

If the **Z-Score** is higher than 3, the system marks it as an error.

## 🛠️ Tech Stack
- **Language:** Python (OOP Principles)
- **Data/Stats:** NumPy, Pandas, Scikit-learn
- **Dashboard:** Streamlit
- **AI Integration:** OpenAI API (GPT-4o-mini)

## 📂 Project Structure
- `src/validator.py`: Statistical checks (Z-Score & Ranges).
- `src/generator.py`: Simulator for "dirty" industrial data.
- `src/describer.py`: AI data quality reporting.
- `app.py`: Main dashboard application.