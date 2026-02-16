# 🛡️ Smart Sensor Data Quality Guardrail

Ensuring AI reliability in smart manufacturing by validating industrial sensor data before it reaches the model. This project addresses the "Data Validation" requirements in modern Industry 4.0 pipelines.

## 🚀 Overview
In industrial environments, sensor failures (frozen sensors, spikes, or drift) can lead to catastrophic AI predictions. This "Guardrail" system acts as a statistical security layer that filters raw data and provides AI-powered quality insights.

## 🧠 Statistical Logic
The system employs two main validation layers:
1. **Physical Range Checks:** Validates if readings are within the equipment's physical operating limits.
2. **Statistical Outlier Detection:** Uses the **Z-Score** method to identify technical anomalies that fall outside the 3-sigma rule of the normal distribution.

$$Z = \frac{x - \mu}{\sigma}$$

*Where $x$ is the reading, $\mu$ is the mean, and $\sigma$ is the standard deviation.*

## 📊 Features
- **Real-time Validation:** Instant checks for incoming sensor streams.
- **Statistical Dashboard:** Visualizing Z-Score hits and range violations.
- **AI Quality Reports:** GPT-4o-mini interprets statistical errors into actionable engineering advice.

## 🛠️ Tech Stack
- **Languages:** Python (OOP)
- **Statistics/Data:** NumPy, Pandas, Scikit-learn
- **Interface:** Streamlit
- **LLM Integration:** OpenAI API

## 📂 Project Structure
- `src/validator.py`: Core statistical logic (Z-Score & Ranges).
- `src/generator.py`: Simulator for "dirty" industrial data.
- `src/describer.py`: AI-powered data quality reporting.
- `app.py`: Streamlit-based monitoring dashboard.

---
*Developed as part of an industrial AI portfolio focusing on smart manufacturing and data integrity.*