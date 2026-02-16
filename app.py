import streamlit as st
import pandas as pd
from src.generator import FaultyDataGenerator
from src.validator import SensorValidator
from src.describer import DataQualityDescriber

st.set_page_config(page_title="Sensor Quality Guardrail", layout="wide")
st.title("🛡️ Smart Sensor Data Quality Guardrail")

# 1. Veri Üretimi
gen = FaultyDataGenerator()
df = gen.generate_data()

st.subheader("📡 Incoming Raw Sensor Stream")
st.line_chart(df)

# 2. Validasyon Kuralları
thresholds = {'temperature': {'min': 0, 'max': 150}}
validator = SensorValidator(thresholds)

if st.button("🔍 Validate Data Quality"):
    # İstatistiksel kontroller
    range_hits = validator.check_ranges(df)
    z_hits = validator.check_z_scores(df)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Physical Range Violations", range_hits['temperature'])
    with col2:
        st.metric("Statistical Outliers (Z-Score)", z_hits['temperature'])

    # 3. AI Yorumu
    describer = DataQualityDescriber()
    with st.spinner("AI is evaluating data reliability..."):
        report = describer.describe_errors(range_hits, z_hits)
        st.info("📝 AI Data Quality Report:")
        st.write(report)