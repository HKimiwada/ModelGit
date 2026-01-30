import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
import os
from pkg import bridge  # Import constants

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="ModelGit Mission Control",
    page_icon="🛡️",
    layout="wide"
)

# --- CSS STYLING ---
st.markdown("""
<style>
    /* Status Badge Styling */
    .stAlert {font-weight: bold;}
    
    /* Button Styling */
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
    st.title("🛡️ ModelGit Guardian")
    st.caption("Active Runtime Protection & Auto-Remediation System")
with col2:
    status_placeholder = st.empty()

# --- REAL-TIME METRICS ---
m1, m2, m3, m4 = st.columns(4)
with m1: step_metric = st.empty()
with m2: loss_metric = st.empty()
with m3: buffer_metric = st.empty()
with m4: revert_metric = st.empty()

# --- MAIN CHART ---
chart_placeholder = st.empty()

# --- CONTROL PANEL ---
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
            # 1. Read Data
            df = pd.read_csv(bridge.LOG_FILE)
            
            if not df.empty:
                latest = df.iloc[-1]
                current_status = latest['status']
                
                # 2. Update Status Badge
                if current_status == "safe":
                    status_placeholder.success("SYSTEM HEALTHY")
                elif current_status == "warning":
                    status_placeholder.warning("⚠️ DRIFT DETECTED")
                elif current_status == "reverted":
                    status_placeholder.error("🚨 REVERTING MODEL...")
                elif current_status == "cooldown":
                    status_placeholder.info("🧊 COOLING DOWN")

                # 3. Update Metrics
                step_metric.metric("Current Step", int(latest['step']))
                loss_metric.metric("Current Loss", f"{latest['loss']:.4f}")
                
                # Calculate simple moving average for visual clarity
                df['loss_ma'] = df['loss'].rolling(window=5).mean()
                
                # Count total interventions
                total_reverts = len(df[df['status'] == 'reverted'])
                revert_metric.metric("Auto-Interventions", total_reverts)
                
                # 4. Render Chart
                fig = go.Figure()
                
                # A. The Loss Line
                fig.add_trace(go.Scatter(
                    x=df['step'], y=df['loss'],
                    mode='lines', name='Raw Loss',
                    line=dict(color='#333333', width=1),
                    opacity=0.5
                ))
                
                fig.add_trace(go.Scatter(
                    x=df['step'], y=df['loss_ma'],
                    mode='lines', name='Smoothed Loss',
                    line=dict(color='#00CC96', width=3)
                ))

                # B. The Revert Markers (The "Red X")
                revert_events = df[df['status'] == 'reverted']
                if not revert_events.empty:
                    fig.add_trace(go.Scatter(
                        x=revert_events['step'], y=revert_events['loss'],
                        mode='markers', name='Intervention',
                        marker=dict(color='red', size=15, symbol='x')
                    ))
                
                # C. The Cooldown Zones (Blue Shading)
                # (Optional advanced visualization: shade areas where status == cooldown)
                
                fig.update_layout(
                    title="Real-Time Loss Landscape",
                    xaxis_title="Step",
                    yaxis_title="Loss",
                    template="plotly_dark",
                    height=450,
                    margin=dict(l=10, r=10, t=40, b=10),
                    yaxis_range=[0, max(10, df['loss'].max() + 5)]
                )
                
                chart_placeholder.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            # Prevent crashing on file read collisions
            pass
            
    time.sleep(0.5) # Refresh rate