import os
import datetime
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from overall_report import generate_overall_report
import json
import numpy as np

def download_json(data, filename):
    json_data = json.dumps(data, indent=4, default=str)
    st.download_button(
        label="📥 Download Data",
        data=json_data,
        file_name=filename,
        mime="application/json"
    )

_HERE = os.path.dirname(os.path.abspath(__file__))

# ================================================================
# =====================  SHARED UTILITIES  =======================
# ================================================================

COLORS = ["#7FB3D5", "#76D7C4", "#F7DC6F", "#BB8FCE", "#F1948A", "#85C1E9"]

PLOTLY_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, sans-serif"),
    margin=dict(l=40, r=20, t=40, b=40),
)

def _fig_bar(data, x, y, title, xlab, ylab, color=None, orientation='v', text=None, height=350):
    """Quick vertical/horizontal bar chart with Plotly."""
    if color is None:
        color = COLORS[0]
    if orientation == 'v':
        fig = px.bar(data, x=x, y=y, text=text or y,
                     color_discrete_sequence=[color])
        fig.update_traces(texttemplate='%{text:.2s}', textposition='outside', textfont_size=9)
    else:
        fig = px.bar(data, x=y, y=x, text=text or y, orientation='h',
                     color_discrete_sequence=[color])
        fig.update_traces(texttemplate='%{text:.1f}', textposition='outside', textfont_size=9)
    fig.update_layout(**PLOTLY_LAYOUT, title=title, xaxis_title=xlab, yaxis_title=ylab, height=height, showlegend=False)
    return fig


# ================================================================
# ====================  DATA LOADERS  ===========================
# ================================================================

@st.cache_data
def load_main_data():
    """Load final_datasett.csv — new dataset for all charts except trend."""
    df = pd.read_csv(os.path.join(_HERE, "Data", "final_merged_datasett.csv"))
    df.columns = df.columns.str.strip()

    # Vectorised date conversion (replaces slow row-by-row apply+lambda)
    date_vals = pd.to_numeric(df['Date'], errors='coerce')
    origin = pd.Timestamp('1899-12-30')
    df['perf_date'] = origin + pd.to_timedelta(date_vals, unit='D')

    for c in ['AHT', 'score', 'Calls_Answered', 'Occupancy %', 'Productive_Hours', 'AGE']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df['AHT_min'] = df['AHT'] / 60

    df['age_group'] = pd.cut(
        df['AGE'],
        bins=[18, 25, 35, 45, 55, 70],
        labels=["18-25", "25-35", "35-45", "45-55", "55+"]
    )
    return df


@st.cache_data
def load_coaching_data():
    """Load final_merged_dataset.csv for Before/After Coaching trend."""
    df = pd.read_csv(os.path.join(_HERE, "Data", "final_merged_dataset.csv"))
    df.columns = df.columns.str.strip()

    df['score']        = pd.to_numeric(df['score'],        errors='coerce')
    df['AHT']          = pd.to_numeric(df['AHT'],          errors='coerce')
    df['Occupancy %']  = pd.to_numeric(df['Occupancy %'],  errors='coerce')

    df['perf_date']    = pd.to_datetime(df['date'],        errors='coerce')
    df['coached_date'] = pd.to_datetime(df['DATECOACHED'], errors='coerce')

    return df


# ================================================================
# ====================  DASHBOARD 1  ============================
# ================================================================


def show_dashboard1():
    df = load_main_data()
    perf_df = df[df['AHT'].notna() & (df['AHT'] > 0)].copy()

    st.title("📊 Overall Dashboard")
    st.markdown("### 🔍 Filters")

    c1, c2, c3, c4 = st.columns(4)
    city    = c1.selectbox("City", ["All"] + sorted(df["CITY_NAME"].dropna().unique()))
    state   = c2.selectbox("State", ["All"] + sorted(df["STATE_NAME"].dropna().unique()))
    account = c3.selectbox("Account",    ["All"] + sorted(df["Account_merged"].dropna().unique()),  key="d1_account")
    job     = c4.selectbox("Job Family", ["All"] + sorted(df["JOB_FAMILY"].dropna().unique()),      key="d1_job")

    fdf = df.copy()
    if city != "All":    fdf = fdf[fdf["CITY_NAME"] == city]
    if state != "All":   fdf = fdf[fdf["STATE_NAME"] == state]
    if account != "All": fdf = fdf[fdf["Account_merged"] == account]
    if job != "All":     fdf = fdf[fdf["JOB_FAMILY"] == job]

    # KPIs
    st.markdown("### 📊 Overview")
    k1, k2, k3 = st.columns(3)
    k1.metric("Employees",      fdf["employee_id"].nunique())
    k2.metric("Avg Score",      round(fdf["score"].mean(), 2))
    k3.metric("Avg AHT (min)",  round(fdf["AHT_min"].mean(), 2))

    st.markdown("## Insights")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Performance vs Efficiency")
        fig = px.scatter(fdf.sample(min(5000, len(fdf)), random_state=42) if len(fdf) > 5000 else fdf,
                         x="AHT_min", y="score", opacity=0.4,
                         color_discrete_sequence=["#a8dadc"])
        fig.add_hline(y=fdf["score"].mean(), line_dash="dash", line_color="gray", opacity=0.6)
        fig.add_vline(x=fdf["AHT_min"].mean(), line_dash="dash", line_color="gray", opacity=0.6)
        fig.update_layout(**PLOTLY_LAYOUT, title="Performance vs Efficiency",
                          xaxis_title="AHT (min)", yaxis_title="Score", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Workload Pressure Analysis")
        tmp = fdf.copy()
        tmp["pressure"] = tmp["Calls_Answered"] / (tmp["score"] + 1)
        data = tmp.groupby("CITY_NAME")["pressure"].mean().reset_index()
        data = data.sort_values("pressure", ascending=False).head(5)
        fig = px.bar(data, x="CITY_NAME", y="pressure", text=data["pressure"].round(1),
                     color_discrete_sequence=["#ffd6a5"])
        fig.update_traces(textposition='outside', textfont_size=9)
        fig.update_layout(**PLOTLY_LAYOUT, title="Workload Pressure by City",
                          xaxis_title="City", yaxis_title="Pressure Index", showlegend=False)
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### Balanced Workforce Analysis")
        balanced   = ((fdf["score"] > 80) & (fdf["AHT_min"] < 10)).sum()
        imbalanced = len(fdf) - balanced
        fig = px.pie(values=[balanced, imbalanced], names=["Balanced", "Imbalanced"],
                     color_discrete_sequence=["#cdeac0", "#ffadad"], hole=0.35)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(**PLOTLY_LAYOUT, title="Balanced Workforce", height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown("### Account Efficiency Analysis")
        data = fdf.groupby("Account_merged")[["score", "AHT_min"]].mean().reset_index()
        fig = px.scatter(data, x="AHT_min", y="score", text="Account_merged",
                         color_discrete_sequence=["#e4c1f9"], size_max=12)
        fig.update_traces(textposition='top center', textfont_size=8)
        fig.update_layout(**PLOTLY_LAYOUT, title="Account Efficiency",
                          xaxis_title="AHT (min)", yaxis_title="Score", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    col5, _ = st.columns(2)
    with col5:
        st.markdown("### Job Stability Analysis")
        data = fdf.groupby("JOB_FAMILY")["score"].std().reset_index()
        fig = px.bar(data, x="JOB_FAMILY", y="score", text=data["score"].round(2),
                     color_discrete_sequence=["#ffadad"])
        fig.update_traces(textposition='outside', textfont_size=9)
        fig.update_layout(**PLOTLY_LAYOUT, title="Score Variability by Job Family",
                          xaxis_title="Job Family", yaxis_title="Std Dev", showlegend=False)
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    # ===================== TOP PERFORMERS =====================
    st.markdown("## 🏆 Top Performers")

    metric_details = {
        "Score": {"col": "score", "ascending": False, "ylabel": "Average Score"},
        "AHT (min)": {"col": "AHT_min", "ascending": True, "ylabel": "Average AHT (minutes)"},
        "Calls Answered": {"col": "Calls_Answered", "ascending": False, "ylabel": "Average Calls Answered"},
        "Occupancy %": {"col": "Occupancy %", "ascending": False, "ylabel": "Average Occupancy %"},
        "Productive Hours": {"col": "Productive_Hours", "ascending": False, "ylabel": "Average Productive Hours"}
    }

    grouping_details = [
        {"col": "employee_id", "display_name": "Employee ID", "title_suffix": "Employees", "color": "#a8dadc"},
        {"col": "CITY_NAME", "display_name": "City Name", "title_suffix": "Cities", "color": "#cdeac0"},
        {"col": "STATE_NAME", "display_name": "State Name", "title_suffix": "States", "color": "#ffd6a5"}
    ]

    metric_label = st.selectbox("Select Metric", list(metric_details.keys()))
    metric_info = metric_details[metric_label]
    metric_col = metric_info["col"]
    ascending_order = metric_info["ascending"]
    metric_ylabel = metric_info["ylabel"]

    existing_grouping_cols = [g['col'] for g in grouping_details if g['col'] in fdf.columns]
    columns_to_check = existing_grouping_cols + [metric_col]

    df_full = load_main_data()
    temp_df = df_full[columns_to_check].dropna()
    if metric_col == "AHT_min":
        temp_df = temp_df[temp_df["AHT_min"] > 0]

    # PRECOMPUTE GROUPBY ONCE
    grouped_data = {}
    for g in grouping_details:
        col = g["col"]
        if col in temp_df.columns:
            grouped_data[col] = temp_df.groupby(col)[metric_col].mean().reset_index()

    colA, colB, colC = st.columns(3)

    for col_widget, group_info in zip([colA, colB, colC], grouping_details):
        grp_col = group_info["col"]
        grp_display_name = group_info["display_name"]
        title_suffix = group_info["title_suffix"]
        color = group_info["color"]

        with col_widget:
            if grp_col not in temp_df.columns:
                st.warning(f"{grp_display_name} column not found")
                continue

            data = grouped_data[grp_col]
            data = data.sort_values(metric_col, ascending=ascending_order).head(5)

            if not data.empty:
                fig = px.bar(data, x=grp_col, y=metric_col,
                             text=data[metric_col].round(2),
                             color_discrete_sequence=[color])
                fig.update_traces(textposition='outside', textfont_size=8)
                fig.update_layout(**PLOTLY_LAYOUT,
                                  title=f"Top 5 {title_suffix} by {metric_label}",
                                  xaxis_title=grp_display_name,
                                  yaxis_title=metric_ylabel, showlegend=False)
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No data for {title_suffix}")


# ================================================================
# ====================  DASHBOARD 2  ============================
# ================================================================

def show_dashboard2():
    df = load_main_data()
    st.title("📊 Overall Dashboard")

    demo_df = df[df['GENDER'].notna()].copy()
    perf_df = df[df['AHT'].notna() & (df['AHT'] > 0)].copy()

    # Filters
    st.subheader("🔍 Filters")
    col1, col2, col3 = st.columns(3)
    sel_gender = col1.selectbox("Gender",         ["All"] + sorted(demo_df["GENDER"].dropna().unique()),                       key="d2_gender")
    sel_age    = col2.selectbox("Age Group",      ["All"] + sorted(demo_df["age_group"].dropna().astype(str).unique()),        key="d2_age")
    sel_status = col3.selectbox("Current Status", ["All"] + sorted(demo_df["CURRENT_STATUS"].dropna().unique()),               key="d2_status")

    fdemo = demo_df.copy()
    fperf = perf_df.copy()
    if sel_gender != "All":
        fdemo = fdemo[fdemo["GENDER"]                    == sel_gender]
        fperf = fperf[fperf["GENDER"]                    == sel_gender]
    if sel_age != "All":
        fdemo = fdemo[fdemo["age_group"].astype(str)     == sel_age]
        fperf = fperf[fperf["age_group"].astype(str)     == sel_age]
    if sel_status != "All":
        fdemo = fdemo[fdemo["CURRENT_STATUS"]            == sel_status]
        fperf = fperf[fperf["CURRENT_STATUS"]            == sel_status]

    # Metrics
    st.subheader("📊 Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Employees",   fdemo["employee_id"].nunique())
    col2.metric("Avg Quality Score", round(fdemo["score"].mean(), 2))
    col3.metric("Avg AHT (min)",     round(fperf["AHT_min"].mean(), 2))
    st.markdown("---")

    # EDA
    st.markdown("## 📊 Basic Exploratory Data Analysis")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Age Distribution")
        fig = px.histogram(fdemo, x="AGE", nbins=15, color_discrete_sequence=["#7FB3D5"])
        fig.update_layout(**PLOTLY_LAYOUT, title="Age Distribution",
                          xaxis_title="Age", yaxis_title="Count", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Gender Distribution")
        data = fdemo["GENDER"].value_counts().reset_index()
        data.columns = ["GENDER", "count"]
        fig = px.bar(data, x="GENDER", y="count", text="count",
                     color="GENDER", color_discrete_sequence=COLORS)
        fig.update_traces(textposition='outside', textfont_size=10)
        fig.update_layout(**PLOTLY_LAYOUT, title="Gender Distribution",
                          xaxis_title="Gender", yaxis_title="Count", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Worker Type Distribution")
        data = fdemo["WORKER_TYPE"].value_counts().reset_index()
        data.columns = ["WORKER_TYPE", "count"]
        fig = px.bar(data, x="WORKER_TYPE", y="count", text="count",
                     color_discrete_sequence=COLORS)
        fig.update_traces(textposition='outside', textfont_size=10)
        fig.update_layout(**PLOTLY_LAYOUT, title="Worker Type Distribution",
                          xaxis_title="Worker Type", yaxis_title="Count", showlegend=False)
        fig.update_xaxes(tickangle=30)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Job Family Distribution")
        data = fdemo["JOB_FAMILY"].value_counts().head(10).reset_index()
        data.columns = ["JOB_FAMILY", "count"]
        fig = px.bar(data, x="JOB_FAMILY", y="count", text="count",
                     color_discrete_sequence=COLORS)
        fig.update_traces(textposition='outside', textfont_size=9)
        fig.update_layout(**PLOTLY_LAYOUT, title="Top Job Families",
                          xaxis_title="Job Family", yaxis_title="Count", showlegend=False)
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Military Service Distribution")
        data = fdemo["MILITARY_SERVICE"].value_counts().reset_index()
        data.columns = ["MILITARY_SERVICE", "count"]
        fig = px.bar(data, x="MILITARY_SERVICE", y="count", text="count",
                     color_discrete_sequence=COLORS)
        fig.update_traces(textposition='outside', textfont_size=10)
        fig.update_layout(**PLOTLY_LAYOUT, title="Military Service",
                          xaxis_title="Service (Y/N)", yaxis_title="Count", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Business Site Distribution")
        data = fdemo["BUSINESS_SITE_LOCATION"].value_counts().head(10).reset_index()
        data.columns = ["BUSINESS_SITE_LOCATION", "count"]
        fig = px.bar(data, x="BUSINESS_SITE_LOCATION", y="count", text="count",
                     color_discrete_sequence=COLORS)
        fig.update_traces(textposition='outside', textfont_size=9)
        fig.update_layout(**PLOTLY_LAYOUT, title="Top Business Locations",
                          xaxis_title="Location", yaxis_title="Count", showlegend=False)
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        data = fdemo.groupby("age_group")["score"].mean().reset_index()
        fig = px.bar(data, x="age_group", y="score", text=data["score"].round(2),
                     color_discrete_sequence=COLORS)
        fig.update_traces(textposition='outside', textfont_size=10)
        fig.update_layout(**PLOTLY_LAYOUT, title="Age Group vs Quality Score",
                          xaxis_title="Age Group", yaxis_title="Avg Score", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        data = fperf.groupby("age_group")["AHT_min"].mean().reset_index()
        fig = px.bar(data, x="age_group", y="AHT_min", text=data["AHT_min"].round(1),
                     color_discrete_sequence=COLORS)
        fig.update_traces(textposition='outside', textfont_size=10)
        fig.update_layout(**PLOTLY_LAYOUT, title="Age Group vs Avg AHT",
                          xaxis_title="Age Group", yaxis_title="Avg AHT (min)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        data = fperf.groupby("GENDER")["AHT_min"].mean().reset_index()
        fig = px.bar(data, x="GENDER", y="AHT_min", text=data["AHT_min"].round(1),
                     color_discrete_sequence=COLORS)
        fig.update_traces(textposition='outside', textfont_size=10)
        fig.update_layout(**PLOTLY_LAYOUT, title="Gender vs Avg AHT",
                          xaxis_title="Gender", yaxis_title="Avg AHT (min)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        data = fdemo.groupby("MARITAL_STATUS")["score"].mean().reset_index()
        fig = px.bar(data, x="MARITAL_STATUS", y="score", text=data["score"].round(2),
                     color_discrete_sequence=COLORS)
        fig.update_traces(textposition='outside', textfont_size=10)
        fig.update_layout(**PLOTLY_LAYOUT, title="Marital Status vs Quality Score",
                          xaxis_title="Marital Status", yaxis_title="Avg Score", showlegend=False)
        fig.update_xaxes(tickangle=30)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        data = fperf.groupby("JOB_FAMILY")["AHT_min"].mean().sort_values().reset_index()
        fig = px.bar(data, x="AHT_min", y="JOB_FAMILY", orientation='h',
                     text=data["AHT_min"].round(1), color_discrete_sequence=["#7FB3D5"])
        fig.update_traces(textposition='outside', textfont_size=9)
        fig.update_layout(**PLOTLY_LAYOUT, title="Job Family vs AHT (min)",
                          xaxis_title="AHT (min)", yaxis_title="Job Family", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        data = fperf.groupby("WORKER_TYPE")["AHT_min"].mean().reset_index()
        fig = px.bar(data, x="WORKER_TYPE", y="AHT_min", text=data["AHT_min"].round(1),
                     color_discrete_sequence=COLORS)
        fig.update_traces(textposition='outside', textfont_size=10)
        fig.update_layout(**PLOTLY_LAYOUT, title="Worker Type vs AHT",
                          xaxis_title="Worker Type", yaxis_title="AHT (min)", showlegend=False)
        fig.update_xaxes(tickangle=30)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        sample = fperf.sample(min(5000, len(fperf)), random_state=42) if len(fperf) > 5000 else fperf
        fig = px.scatter(sample, x="AHT_min", y="score", opacity=0.2,
                         color_discrete_sequence=["#5DADE2"])
        fig.update_traces(marker_size=4)
        fig.update_layout(**PLOTLY_LAYOUT, title="Performance vs Quality",
                          xaxis_title="AHT (min)", yaxis_title="Quality Score", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        data = fperf["MILITARY_SERVICE"].value_counts().reset_index()
        data.columns = ["MILITARY_SERVICE", "count"]
        fig = px.bar(data, x="MILITARY_SERVICE", y="count", text="count",
                     color_discrete_sequence=COLORS)
        fig.update_traces(textposition='outside', textfont_size=10)
        fig.update_layout(**PLOTLY_LAYOUT, title="Military Service Distribution",
                          xaxis_title="Service", yaxis_title="Count", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col1, _ = st.columns(2)
    with col1:
        data = fperf["BUSINESS_SITE_LOCATION"].value_counts().head(10).reset_index()
        data.columns = ["BUSINESS_SITE_LOCATION", "count"]
        fig = px.bar(data, x="BUSINESS_SITE_LOCATION", y="count", text="count",
                     color_discrete_sequence=COLORS)
        fig.update_traces(textposition='outside', textfont_size=9)
        fig.update_layout(**PLOTLY_LAYOUT, title="Business Site Distribution",
                          xaxis_title="Location", yaxis_title="Count", showlegend=False)
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🏆 Best vs Worst Job Family (by AHT)")
    job_perf = fperf.groupby("JOB_FAMILY")["AHT_min"].mean().sort_values()
    c1, c2 = st.columns(2)
    with c1:
        st.write("✅ Best Performing (Lowest AHT)")
        st.dataframe(job_perf.head(3).round(2))
    with c2:
        st.write("❌ Worst Performing (Highest AHT)")
        st.dataframe(job_perf.tail(3).round(2))

    st.markdown("---")

    # ================================================================
    # ============  BEFORE vs AFTER COACHING TREND  ==================
    # ================================================================

    st.markdown("## 📈 Before vs After Coaching — QA Score Trend")

    try:
        trend_df = load_coaching_data()
    except Exception as e:
        st.error(f"Could not load coaching dataset: {e}")
        return

    required_cols = {'ID', 'perf_date', 'score', 'coached_date'}
    if not required_cols.issubset(trend_df.columns):
        st.error(f"Coaching dataset missing columns: {required_cols - set(trend_df.columns)}")
        return

    daily_df = (
        trend_df
        .groupby(['ID', 'perf_date'], as_index=False)
        .agg(score=('score', 'mean'))
    )

    coaching_dates = (
        trend_df[trend_df['coached_date'].notna()]
        .groupby('ID')['coached_date']
        .min()
        .reset_index()
        .rename(columns={'coached_date': 'coaching_date'})
    )

    # Vectorised valid employee detection
    merged = daily_df.merge(coaching_dates, on='ID', how='inner')
    has_before = merged[merged['perf_date'] < merged['coaching_date']].groupby('ID').size()
    has_after  = merged[merged['perf_date'] >= merged['coaching_date']].groupby('ID').size()
    valid_emps = sorted(set(has_before.index) & set(has_after.index))

    if len(valid_emps) == 0:
        st.warning("No employees found with QA score data both before and after their coaching date.")
        return

    st.markdown("### 🔍 Select Employee to View Trend")
    col_drop, col_info = st.columns([2, 3])

    with col_drop:
        selected_emp = st.selectbox(
            "Employee ID",
            options=sorted(valid_emps),
            key="trend_emp_select",
            help=f"{len(valid_emps)} employees have data before and after coaching"
        )

    coaching_date = pd.Timestamp(
        coaching_dates.loc[coaching_dates['ID'] == selected_emp, 'coaching_date'].values[0]
    )

    emp_daily = daily_df[daily_df['ID'] == selected_emp].sort_values('perf_date')

    with col_info:
        st.info(f"**Coaching Date:** {coaching_date.strftime('%d %b %Y')}  ·  "
                f"**Data Points:** {len(emp_daily)}")

    before = emp_daily[emp_daily['perf_date'] < coaching_date]
    after  = emp_daily[emp_daily['perf_date'] >= coaching_date]

    # Plotly chart for before/after trend
    fig = go.Figure()

    if not before.empty:
        fig.add_trace(go.Scatter(
            x=before['perf_date'], y=before['score'],
            mode='lines+markers+text', name="Before Coaching",
            line=dict(color="#E74C3C", width=2),
            marker=dict(size=7),
            text=before['score'].round(1), textposition='top center', textfont=dict(size=8)
        ))

    if not after.empty:
        fig.add_trace(go.Scatter(
            x=after['perf_date'], y=after['score'],
            mode='lines+markers+text', name="After Coaching",
            line=dict(color="#27AE60", width=2),
            marker=dict(size=7),
            text=after['score'].round(1), textposition='top center', textfont=dict(size=8)
        ))

    fig.add_vline(x=coaching_date.timestamp() * 1000, line_dash="dash", line_color="orange", line_width=2,
                  annotation_text="Coaching Date", annotation_position="top right")

    fig.update_layout(
        template="plotly_white",
        title=f"Employee {selected_emp} — QA Score Before vs After Coaching",
        xaxis_title="Date", yaxis_title="QA Score",
        legend=dict(orientation="h", y=1.12),
        height=450,
        font=dict(family="Inter, sans-serif"),
        margin=dict(l=40, r=20, t=60, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Summary metrics
    if not before.empty and not after.empty:
        avg_before = before['score'].mean()
        avg_after  = after['score'].mean()
        delta      = avg_after - avg_before
        m1, m2, m3 = st.columns(3)
        m1.metric("Avg Before", f"{avg_before:.2f}")
        m2.metric("Avg After",  f"{avg_after:.2f}")
        m3.metric("Change",     f"{delta:+.2f}", delta=f"{delta:+.2f}")


# ================================================================
# ==========================  MAIN  ==============================
# ================================================================

def show():

    tab1, tab2 = st.tabs(["📊 Dashboard 1", "📊 Dashboard 2"])

    with tab1:
        show_dashboard1()

    with tab2:
        show_dashboard2()

        st.markdown("## 📥 Download Overall Report")

        if st.button("Generate Overall Report"):

            df = load_main_data()

            report = generate_overall_report(df)

            json_data = json.dumps(report, indent=4)

            st.download_button(
            "📥 Download JSON",
            json_data,
            "overall_report.json",
            "application/json"
        )