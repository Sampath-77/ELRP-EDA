import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import performance
import os
import numpy as np

import quality
import overall

BASE_DIR = os.path.dirname(__file__)

st.set_page_config(page_title="Employee Dashboard", layout="wide")
sns.set_style("whitegrid")

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(BASE_DIR, "Data", "demographics.csv"))

def add_labels(ax):
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', label_type='edge', fontsize=9)


section = st.sidebar.radio(
    "Select Section",
    ["Employee Demography", "Performance", "Quality & Coaching", "Overall"]
)
if section != "Performance":
    st.markdown(
        "<h1 style='text-align: center;'>📊 Employee Analytics Dashboard</h1>",
        unsafe_allow_html=True
    )


# ================= DEMOGRAPHY =================

@st.cache_data
def filter_data(df, year, quarter, month, marital, location, city, state, status):
    filtered_df = df.copy()

    if year != "All":
        filtered_df = filtered_df[filtered_df["YEAR"] == year]
    if quarter != "All":
        filtered_df = filtered_df[filtered_df["QUARTER"] == quarter]
    if month != "All":
        filtered_df = filtered_df[filtered_df["MONTH"] == month]
    if marital != "All":
        filtered_df = filtered_df[filtered_df["MARITAL_STATUS"] == marital]
    if location != "All":
        filtered_df = filtered_df[filtered_df["BUSINESS_SITE_LOCATION"] == location]
    if city != "All":
        filtered_df = filtered_df[filtered_df["CITY_NAME"] == city]
    if state != "All":
        filtered_df = filtered_df[filtered_df["STATE_NAME"] == state]
    if status != "All":
        filtered_df = filtered_df[filtered_df["CURRENT_STATUS"] == status]

    return filtered_df
if section == "Employee Demography":
    df = load_data()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        year = st.selectbox("Year", ["All"] + sorted(df["YEAR"].dropna().unique()))
    with col2:
        quarter = st.selectbox("Quarter", ["All"] + sorted(df["QUARTER"].dropna().unique()))
    with col3:
        month = st.selectbox("Month", ["All"] + sorted(df["MONTH"].dropna().unique()))
    with col4:
        marital = st.selectbox("Marital Status", ["All"] + sorted(df["MARITAL_STATUS"].dropna().unique()))

    col5, col6, col7, col8 = st.columns(4)

    with col5:
        location = st.selectbox("Business Location", ["All"] + sorted(df["BUSINESS_SITE_LOCATION"].dropna().unique()))
    with col6:
        city = st.selectbox("City", ["All"] + sorted(df["CITY_NAME"].dropna().unique()))
    with col7:
        state = st.selectbox("State", ["All"] + sorted(df["STATE_NAME"].dropna().unique()))
    with col8:
        status = st.selectbox("Current Status", ["All"] + sorted(df["CURRENT_STATUS"].dropna().unique()))

    filtered_df = filter_data(df, year, quarter, month, marital, location, city, state, status)
 

    # 🔥 Precompute all counts once
    state_counts = filtered_df["STATE_NAME"].value_counts()
    city_counts = filtered_df["CITY_NAME"].value_counts()
    worker_counts = filtered_df["WORKER_TYPE"].value_counts()
    geo_counts = filtered_df["GEOZONE_CODE"].value_counts()
    military_counts = filtered_df["MILITARY_SERVICE"].value_counts()
    site_counts = filtered_df["BUSINESS_SITE_LOCATION"].value_counts()
    year_counts = filtered_df["YEAR"].value_counts().sort_index()
    quarter_counts = filtered_df["QUARTER"].value_counts().sort_index()
    

    if filtered_df.empty:
        st.warning("⚠️ No data available for selected filters")

    else:
        st.subheader("📊 Employee Insights Dashboard")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Marital Status vs Current Status")
            fig1, ax1 = plt.subplots()
            sns.countplot(data=filtered_df, x="MARITAL_STATUS", hue="CURRENT_STATUS", palette="pastel", ax=ax1)
            add_labels(ax1)
            st.pyplot(fig1)
            plt.close(fig1)

        with col2:
            st.markdown("### Gender vs Current Status")
            fig2, ax2 = plt.subplots()
            sns.countplot(data=filtered_df, x="GENDER", hue="CURRENT_STATUS", palette="muted", ax=ax2)
            add_labels(ax2)
            st.pyplot(fig2)
            plt.close(fig2)

        filtered_df["AGE_GROUP"] = pd.cut(
            filtered_df["AGE"],
            bins=[20, 30, 40, 50, 60],
            labels=["20-30", "30-40", "40-50", "50-60"]
        )

        col3, col4 = st.columns(2)

# ✅ LEFT GRAPH (percentage)
        with col3:
            st.markdown("### Age vs Current Status")

            data = (
                filtered_df.groupby(["AGE_GROUP", "CURRENT_STATUS"])
                .size()
                .reset_index(name="count")
            )

            data["percentage"] = data.groupby("AGE_GROUP")["count"].transform(lambda x: x / x.sum() * 100)

            fig3, ax3 = plt.subplots(figsize=(6,4))

            sns.barplot(
                data=data,
                x="AGE_GROUP",
                y="percentage",
                hue="CURRENT_STATUS",
                palette="Blues",
                ax=ax3
            )

            ax3.set_xlabel("Age Group")
            ax3.set_ylabel("Percentage (%)")

            for container in ax3.containers:
                ax3.bar_label(container, fmt="%.1f%%")

            plt.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)


       # ✅ RIGHT GRAPH (percentage)
        with col4:
            st.markdown("### Termination DV vs Current Status")

            order = ["0-30", "31-60", "61-90", "91-120", "120+"]
            fig4, ax4 = plt.subplots(figsize=(6,4))

            sns.countplot(
                data=filtered_df,
                x="TERMINATION_DV",
                hue="CURRENT_STATUS",
                order=order,
                palette="pastel",
                ax=ax4
            )

            add_labels(ax4)
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close(fig4)
        # EXTRA CHARTS
        col1, col2 = st.columns(2)

    with col1:
            st.markdown("### Job Family Distribution")
            fig, ax = plt.subplots(figsize=(5,4))
            sns.countplot(data=filtered_df, x="JOB_FAMILY", palette="muted", ax=ax)
            
            ax.set_xlabel("Job Family")
            ax.set_ylabel("Count")
            
            # ✅ Add values on bars
            for container in ax.containers:
                ax.bar_label(container)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


    with col2:
            st.markdown("### Management Level Distribution")
            fig, ax = plt.subplots(figsize=(5,4))
            sns.countplot(data=filtered_df, x="MANAGEMENT_LEVEL", palette="pastel", ax=ax)
            
            ax.set_xlabel("Management Level")
            ax.set_ylabel("Count")
            
            # ✅ Add values on bars
            for container in ax.containers:
                ax.bar_label(container)
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        # ================= EXTRA INSIGHT GRAPHS =================

    col5, col6 = st.columns(2)

    # 🔹 Graph 1: Attrition Rate by Job Family
    with col5:
        st.markdown("### Attrition Rate by Job Family")

        attrition = (
            filtered_df.groupby("JOB_FAMILY")["CURRENT_STATUS"]
            .value_counts(normalize=True)
            .mul(100)
            .rename("percentage")
            .reset_index()
        )

        attrition = attrition[attrition["CURRENT_STATUS"] == "TERMINATED"]

        fig, ax = plt.subplots()
        sns.barplot(data=attrition, x="JOB_FAMILY", y="percentage", palette="Blues", ax=ax)

        ax.set_xlabel("Job Family")
        ax.set_ylabel("Attrition %")

        for i, v in enumerate(attrition["percentage"]):
            ax.text(i, v, f"{v:.1f}%", ha='center')

        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close(fig)


    # 🔹 Graph 2: Age Distribution
    with col6:
        st.markdown("### Age Distribution")

        fig, ax = plt.subplots()
        sns.histplot(filtered_df["AGE"], bins=20, kde=True, ax=ax)

        ax.set_xlabel("Age")
        ax.set_ylabel("Count")
        for patch in ax.patches:
            height = patch.get_height()
            if height > 0:
                ax.text(
                    patch.get_x() + patch.get_width() / 2,
                    height,
                    int(height),
                    ha='center',
                    va='bottom',
                    fontsize=8
                )

        st.pyplot(fig)
        plt.close(fig)

        # ================= ADDITIONAL DEMOGRAPHY GRAPHS =================

    col7, col8 = st.columns(2)

    with col7:
            st.markdown("### State Distribution")
            fig, ax = plt.subplots(figsize=(6,4))
            
            sns.barplot(x=state_counts.index, y=state_counts.values, palette="Pastel1", ax=ax)
            ax.set_xlabel("State")
            ax.set_ylabel("Count")
            plt.xticks(rotation=45)
            for container in ax.containers:
                ax.bar_label(container)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with col8:
            st.markdown("### City Distribution")
            fig, ax = plt.subplots(figsize=(6,4))
            
            sns.barplot(x=city_counts.index, y=city_counts.values, palette="Pastel2", ax=ax)
            ax.set_xlabel("City")
            ax.set_ylabel("Count")
            plt.xticks(rotation=45)
            for container in ax.containers:
                ax.bar_label(container)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    col9, col10 = st.columns(2)

    with col9:
            st.markdown("### Worker Type Distribution")
            fig, ax = plt.subplots(figsize=(6,4))
            
            sns.barplot(x=worker_counts.index, y=worker_counts.values, palette="Set2", ax=ax)
            ax.set_xlabel("Worker Type")
            ax.set_ylabel("Count")
            plt.xticks(rotation=45)
            for container in ax.containers:
                ax.bar_label(container)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with col10:
            st.markdown("### Geo Zone Distribution")
            fig, ax = plt.subplots(figsize=(6,4))
            
            sns.barplot(x=geo_counts.index, y=geo_counts.values, palette="Set3", ax=ax)
            ax.set_xlabel("Geo Zone")
            ax.set_ylabel("Count")
            plt.xticks(rotation=45)
            for container in ax.containers:
                ax.bar_label(container)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    col11, col12 = st.columns(2)

    with col11:
            st.markdown("### Military Service Distribution")
            fig, ax = plt.subplots(figsize=(6,4))
            
            sns.barplot(x=military_counts.index, y=military_counts.values, palette="Pastel1", ax=ax)
            ax.set_xlabel("Military Service")
            ax.set_ylabel("Count")
            for container in ax.containers:
                ax.bar_label(container)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with col12:
            st.markdown("### Business Site Location Distribution")
            fig, ax = plt.subplots(figsize=(6,4))
            
            sns.barplot(x=site_counts.index, y=site_counts.values, palette="Set3", ax=ax)
            ax.set_xlabel("Business Site Location")
            ax.set_ylabel("Count")
            plt.xticks(rotation=45)
            for container in ax.containers:
                ax.bar_label(container)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # Time-based trends
    st.subheader("📈 Time-based Trends")

    col13, col14 = st.columns(2)

    with col13:
            st.markdown("### Employee Count by Year")
            fig, ax = plt.subplots(figsize=(6,4))
            
            sns.lineplot(x=year_counts.index, y=year_counts.values, marker='o', ax=ax)
            ax.set_xlabel("Year")
            ax.set_ylabel("Employee Count")
            for x, y in zip(year_counts.index, year_counts.values):
                ax.text(x, y, str(y), ha='center', va='bottom')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with col14:
            st.markdown("### Employee Count by Quarter")
            fig, ax = plt.subplots(figsize=(6,4))
            
            sns.barplot(x=quarter_counts.index, y=quarter_counts.values, palette="Blues", ax=ax)
            ax.set_xlabel("Quarter")
            ax.set_ylabel("Employee Count")
            for container in ax.containers:
                ax.bar_label(container)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # Age and Gender combined insights
    col15, col16 = st.columns(2)

    with col15:
            st.markdown("### Age Distribution by Gender")
            fig, ax = plt.subplots(figsize=(6,4))
            sns.histplot(data=filtered_df, x="AGE", hue="GENDER", multiple="stack", bins=20, ax=ax)
            ax.set_xlabel("Age")
            ax.set_ylabel("Count")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with col16:
            st.markdown("### Marital Status by Gender")
            fig, ax = plt.subplots(figsize=(6,4))
            marital_gender = pd.crosstab(filtered_df["MARITAL_STATUS"], filtered_df["GENDER"])
            marital_gender.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
            ax.set_xlabel("Marital Status")
            ax.set_ylabel("Count")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)



    # ================= DOWNLOAD DEMOGRAPHY REPORT =================
    import json

    st.markdown("## 📥 Download Demography Report")

    if st.button("Generate & Download Demography Report"):

            report = {}

            # 🔹 Save Filters
            report["filters"] = {
                "Year": year,
                "Quarter": quarter,
                "Month": month,
                "Marital Status": marital,
                "Business Location": location,
                "City": city,
                "State": state,
                "Current Status": status
            }

            charts = []

            # 🔹 Marital vs Current Status
            marital_data = (
                filtered_df.groupby(["MARITAL_STATUS", "CURRENT_STATUS"])
                .size()
                .reset_index(name="count")
                .to_dict(orient="records")
            )
            charts.append({
                "title": "Marital Status vs Current Status",
                "data": marital_data
            })

            # 🔹 Gender vs Current Status
            gender_data = (
                filtered_df.groupby(["GENDER", "CURRENT_STATUS"])
                .size()
                .reset_index(name="count")
                .to_dict(orient="records")
            )
            charts.append({
                "title": "Gender vs Current Status",
                "data": gender_data
            })

            # 🔹 Age vs Current Status
            age_data = (
                filtered_df.groupby(["AGE_GROUP", "CURRENT_STATUS"])
                .size()
                .reset_index(name="count")
                .to_dict(orient="records")
            )
            charts.append({
                "title": "Age vs Current Status",
                "data": age_data
            })

            # 🔹 Job Family
            charts.append({
                "title": "Job Family Distribution",
                "data": filtered_df["JOB_FAMILY"].value_counts().to_dict()
            })

            # 🔹 Management Level
            charts.append({
                "title": "Management Level Distribution",
                "data": filtered_df["MANAGEMENT_LEVEL"].value_counts().to_dict()
            })

            # 🔹 Attrition Rate
            attrition = (
                filtered_df.groupby("JOB_FAMILY")["CURRENT_STATUS"]
                .value_counts(normalize=True)
                .mul(100)
                .rename("percentage")
                .reset_index()
            )
            attrition = attrition[attrition["CURRENT_STATUS"] == "TERMINATED"]

            charts.append({
                "title": "Attrition Rate by Job Family",
                "data": attrition.to_dict(orient="records")
            })

            # 🔹 Age Distribution
            charts.append({
                "title": "Age Distribution",
                "data": filtered_df["AGE"].value_counts().to_dict()
            })

            # 🔹 State
            charts.append({
                "title": "State Distribution",
                "data": filtered_df["STATE_NAME"].value_counts().to_dict()
            })

            # 🔹 City
            charts.append({
                "title": "City Distribution",
                "data": filtered_df["CITY_NAME"].value_counts().to_dict()
            })

            # 🔹 Worker Type
            charts.append({
                "title": "Worker Type Distribution",
                "data": filtered_df["WORKER_TYPE"].value_counts().to_dict()
            })

            # 🔹 Geo Zone
            charts.append({
                "title": "Geo Zone Distribution",
                "data": filtered_df["GEOZONE_CODE"].value_counts().to_dict()
            })

            # 🔹 Military
            charts.append({
                "title": "Military Service Distribution",
                "data": filtered_df["MILITARY_SERVICE"].value_counts().to_dict()
            })

            # 🔹 Business Site Location
            charts.append({
                "title": "Business Site Location Distribution",
                "data": filtered_df["BUSINESS_SITE_LOCATION"].value_counts().to_dict()
            })

            # 🔹 Year Trend
            charts.append({
                "title": "Employee Count by Year",
                "data": filtered_df["YEAR"].value_counts().sort_index().to_dict()
            })

            # 🔹 Quarter Trend
            charts.append({
                "title": "Employee Count by Quarter",
                "data": filtered_df["QUARTER"].value_counts().sort_index().to_dict()
            })

            # 🔹 Final attach
            report["charts"] = charts

            json_data = json.dumps(report, indent=4)

            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name="demography_report.json",
                mime="application/json"
            )


elif section == "Performance":
    performance.show()
    
elif section == "Quality & Coaching":
    quality.show()

##elif section == "Coaching":
    ##coaching.show()

# ================= OVERALL =================
elif section == "Overall":
    overall.show()    