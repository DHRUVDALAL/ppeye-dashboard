import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="PPEye Analytics", layout="wide")

st.markdown("""
    <style>
        .title-text {
            font-size: 40px;
            font-weight: 900;
            color: black;
            text-align: center;
        }
        .subtitle {
            font-size: 20px;
            font-weight: 600;
            text-align: center;
            color: black;
        }
        .card {
            background-color: #00000010;
            padding: 15px;
            border-radius: 12px;
            border-left: 6px solid black;
            margin-bottom: 15px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<p class='title-text'>PPEye — Supervisor Analytics</p>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Violation statistics & insights</p>", unsafe_allow_html=True)

st.write("---")

# Load data
try:
    df = pd.read_csv("violations.csv")
except:
    st.error("No violation data found. Run real-time detection first.")
    st.stop()

# Summary counts
no_helmet_count = (df["violation"] == "No Helmet").sum()
no_vest_count = (df["violation"] == "No Vest").sum()

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
        <div class='card'>
            <h3>🚨 No Helmet Violations</h3>
            <h2 style='color:black;'>{no_helmet_count}</h2>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class='card'>
            <h3>🦺 No Vest Violations</h3>
            <h2 style='color:black;'>{no_vest_count}</h2>
        </div>
    """, unsafe_allow_html=True)

# Pie Chart
fig_pie = px.pie(df, names="violation", title="Violation Breakdown")
st.plotly_chart(fig_pie, use_container_width=True)

# Line Chart
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['date'] = df['timestamp'].dt.date

df_daily = df.groupby(['date', 'violation']).size().reset_index(name='count')

fig_line = px.line(
    df_daily,
    x="date",
    y="count",
    color="violation",
    title="Daily Violation Trend",
    markers=True
)
st.plotly_chart(fig_line, use_container_width=True)

st.markdown("### 📄 Full Violation Log")
st.dataframe(df)
