"""
[TR] İnce Streamlit arayüzü; tüm hesaplama src.pipeline içinde.
[EN] Thin Streamlit UI; all computation lives in src.pipeline.
"""
import streamlit as st

from src.pipeline import CHECK_LABELS, run_pipeline

st.set_page_config(page_title="Sensor Quality Guardrail", layout="wide")
st.title("🛡️ Smart Sensor Data Quality Guardrail")

result = run_pipeline()

st.subheader("📡 Incoming Raw Sensor Stream")
st.line_chart(result.data)

st.subheader("🔍 Checks")
cols = st.columns(len(CHECK_LABELS))
for col, (check, label) in zip(cols, CHECK_LABELS.items()):
    col.metric(label, int(result.flags[check].sum()))

st.caption("Evaluation against the generator's ground-truth labels")
st.dataframe(result.evaluation)

v = result.verdict
(st.error if v.label == "UNSAFE" else st.success)(f"Verdict: **{v.label}** — {v.reason}")

if st.button("📝 Explain with AI"):
    try:
        from src.describer import DataQualityDescriber
        with st.spinner("AI is writing the explanation..."):
            st.write(DataQualityDescriber().explain(result.summary()))
    except Exception as exc:  # missing key, network, quota ...
        st.warning(f"AI explanation unavailable: {exc}")
