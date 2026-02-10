"""
Streamlit Real-Time Dashboard for Sentinel RL.
Run: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pkg import bridge

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Sentinel RL — Mission Control",
    page_icon="🛡️",
    layout="wide",
)

st.markdown("""
<style>
    div[data-testid="stHorizontalBlock"] > div:nth-child(1) button {
        background-color: #FF4B4B; color: white; border: none; height: 50px; font-size: 16px;
    }
    div[data-testid="stHorizontalBlock"] > div:nth-child(2) button {
        background-color: #00CC00; color: white; border: none; height: 50px; font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🛡️ Sentinel RL — Mission Control")
    st.caption("Multi-Signal Runtime Protection & Auto-Remediation")
with col2:
    status_placeholder = st.empty()

# --- METRICS ROW ---
m1, m2, m3, m4, m5 = st.columns(5)
with m1: step_metric = st.empty()
with m2: loss_metric = st.empty()
with m3: reward_metric = st.empty()
with m4: entropy_metric = st.empty()
with m5: revert_metric = st.empty()

# --- CHARTS ---
chart_placeholder = st.empty()

# --- CONTROLS ---
st.divider()
st.subheader("Manual Interventions")
c1, c2 = st.columns(2)

with c1:
    if st.button("☣️ INJECT POISON (Simulate Attack)"):
        with open(bridge.COMMAND_FILE, "w") as f:
            f.write("POISON_ON")
        st.toast("Command Sent: Injecting Poison Data...", icon="☣️")

with c2:
    if st.button("⏪ EMERGENCY REVERT (Manual Override)"):
        with open(bridge.COMMAND_FILE, "w") as f:
            f.write("REVERT_NOW")
        st.toast("Command Sent: Forcing Revert...", icon="⏪")

# --- UPDATE LOOP ---
while True:
    if os.path.exists(bridge.LOG_FILE):
        try:
            df = pd.read_csv(bridge.LOG_FILE)

            if not df.empty:
                latest = df.iloc[-1]
                current_status = latest["status"]

                # Status badge
                badge = {
                    "safe": ("SYSTEM HEALTHY", "success"),
                    "warning": ("⚠️ DRIFT DETECTED", "warning"),
                    "reverted": ("🚨 REVERTING MODEL", "error"),
                    "cooldown": ("🧊 COOLING DOWN", "info"),
                    "pre_warning": ("🔮 PREDICTIVE ALERT", "warning"),
                    "warmup": ("🔥 WARMING UP", "info"),
                }
                text, method = badge.get(current_status, ("UNKNOWN", "info"))
                getattr(status_placeholder, method)(text)

                # Metrics
                step_metric.metric("Step", int(latest["step"]))
                loss_metric.metric("Loss", f"{latest['loss']:.4f}")
                reward_metric.metric("Reward", f"{latest.get('reward', 0):.1f}")
                entropy_metric.metric("Entropy", f"{latest.get('entropy', 0):.3f}")
                total_reverts = len(df[df["status"] == "reverted"])
                revert_metric.metric("Reverts", total_reverts)

                # Multi-panel chart
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=("Loss", "Reward", "Entropy", "Grad Norm"),
                    vertical_spacing=0.12,
                )

                # Loss
                df["loss_ma"] = df["loss"].rolling(window=5).mean()
                fig.add_trace(go.Scatter(
                    x=df["step"], y=df["loss"],
                    mode="lines", name="Raw Loss",
                    line=dict(color="#555", width=1), opacity=0.4,
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=df["step"], y=df["loss_ma"],
                    mode="lines", name="Smoothed Loss",
                    line=dict(color="#00CC96", width=2),
                ), row=1, col=1)

                # Reward
                if "reward" in df.columns:
                    df["reward_ma"] = df["reward"].rolling(window=5).mean()
                    fig.add_trace(go.Scatter(
                        x=df["step"], y=df["reward_ma"],
                        mode="lines", name="Reward",
                        line=dict(color="#636EFA", width=2),
                    ), row=1, col=2)

                # Entropy
                if "entropy" in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df["step"], y=df["entropy"],
                        mode="lines", name="Entropy",
                        line=dict(color="#EF553B", width=2),
                    ), row=2, col=1)

                # Grad Norm
                if "grad_norm" in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df["step"], y=df["grad_norm"],
                        mode="lines", name="Grad Norm",
                        line=dict(color="#AB63FA", width=2),
                    ), row=2, col=2)

                # Revert markers on all panels
                reverts = df[df["status"] == "reverted"]
                if not reverts.empty:
                    for row, col, metric in [
                        (1, 1, "loss"), (1, 2, "reward"),
                        (2, 1, "entropy"), (2, 2, "grad_norm"),
                    ]:
                        if metric in reverts.columns:
                            fig.add_trace(go.Scatter(
                                x=reverts["step"], y=reverts[metric],
                                mode="markers", name="Revert",
                                marker=dict(color="red", size=12, symbol="x"),
                                showlegend=(row == 1 and col == 1),
                            ), row=row, col=col)

                fig.update_layout(
                    template="plotly_dark",
                    height=550,
                    margin=dict(l=10, r=10, t=40, b=10),
                    showlegend=True,
                )
                chart_placeholder.plotly_chart(fig, use_container_width=True)

        except Exception:
            pass

    time.sleep(0.5)