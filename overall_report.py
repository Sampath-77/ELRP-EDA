import itertools
import pandas as pd
from datetime import datetime

def generate_overall_report(df):

    final_report = []

    df = df.copy()

    # ✅ Safe column creation
    if "AHT" in df.columns:
        df["AHT_min"] = pd.to_numeric(df["AHT"], errors="coerce") / 60
    else:
        df["AHT_min"] = None

    if "AGE" in df.columns:
        df["AGE"] = pd.to_numeric(df["AGE"], errors="coerce")
        df["age_group"] = pd.cut(
            df["AGE"],
            bins=[18, 25, 35, 45, 55, 70],
            labels=["18-25", "25-35", "35-45", "45-55", "55+"]
        )

    # ✅ Filters
    cities = df["CITY_NAME"].dropna().unique() if "CITY_NAME" in df.columns else []
    jobs = df["JOB_FAMILY"].dropna().unique() if "JOB_FAMILY" in df.columns else []
    statuses = df["CURRENT_STATUS"].dropna().unique() if "CURRENT_STATUS" in df.columns else []

    combinations = itertools.product(cities, jobs, statuses)

    for city, job, status in combinations:

        temp_df = df.copy()

        temp_df = temp_df[temp_df["CITY_NAME"] == city]
        temp_df = temp_df[temp_df["JOB_FAMILY"] == job]
        temp_df = temp_df[temp_df["CURRENT_STATUS"] == status]

        if temp_df.empty:
            continue

        # ---------------- CHART DATA ---------------- #

        perf_eff = temp_df[["AHT_min", "score"]].dropna().to_dict("records") \
            if {"AHT_min", "score"}.issubset(temp_df.columns) else []

        # Workload
        if {"Calls_Answered", "score"}.issubset(temp_df.columns):
            temp_df["pressure"] = temp_df["Calls_Answered"] / (temp_df["score"].fillna(0) + 1)
            workload = (
                temp_df.groupby("CITY_NAME")["pressure"]
                .mean()
                .reset_index()
                .to_dict("records")
            )
        else:
            workload = []

        # Balanced
        if {"score", "AHT_min"}.issubset(temp_df.columns):
            balanced = int(((temp_df["score"] > 80) & (temp_df["AHT_min"] < 10)).sum())
            imbalanced = int(len(temp_df) - balanced)
        else:
            balanced, imbalanced = 0, len(temp_df)

        # Account
        if {"Account_merged", "score", "AHT_min"}.issubset(temp_df.columns):
            account_eff = (
                temp_df.groupby("Account_merged")[["score", "AHT_min"]]
                .mean()
                .reset_index()
                .to_dict("records")
            )
        else:
            account_eff = []

        # Job stability
        if "JOB_FAMILY" in temp_df.columns and "score" in temp_df.columns:
            job_stability = (
                temp_df.groupby("JOB_FAMILY")["score"]
                .std()
                .fillna(0)
                .reset_index()
                .to_dict("records")
            )
        else:
            job_stability = []

        gender_data = temp_df["GENDER"].value_counts().to_dict() \
            if "GENDER" in temp_df.columns else {}

        age_data = temp_df["AGE"].dropna().tolist() \
            if "AGE" in temp_df.columns else []

        worker_data = temp_df["WORKER_TYPE"].value_counts().to_dict() \
            if "WORKER_TYPE" in temp_df.columns else {}

        # ---------------- TOP PERFORMERS ---------------- #

        metrics = {
            "Score": "score",
            "AHT (min)": "AHT_min",
            "Calls Answered": "Calls_Answered",
            "Occupancy %": "Occupancy %",
            "Productive Hours": "Productive_Hours"
        }

        top_performers = {}

        for metric_name, metric_col in metrics.items():

            if metric_col not in temp_df.columns:
                continue

            top_performers[metric_name] = {
                "employees": (
                    temp_df.groupby("employee_id")[metric_col]
                    .mean()
                    .reset_index()
                    .sort_values(metric_col, ascending=False)
                    .head(5)
                    .to_dict("records")
                ) if "employee_id" in temp_df.columns else [],

                "cities": (
                    temp_df.groupby("CITY_NAME")[metric_col]
                    .mean()
                    .reset_index()
                    .sort_values(metric_col, ascending=False)
                    .head(5)
                    .to_dict("records")
                ),

                "states": (
                    temp_df.groupby("STATE_NAME")[metric_col]
                    .mean()
                    .reset_index()
                    .sort_values(metric_col, ascending=False)
                    .head(5)
                    .to_dict("records")
                ) if "STATE_NAME" in temp_df.columns else []
            }

        # ---------------- EMPLOYEE TREND ---------------- #

        employee_trends = []

        if "employee_id" in temp_df.columns and "score" in temp_df.columns:

    # 🔹 Pick top 5 employees based on score
            top_emp_ids = (
        temp_df.groupby("employee_id")["score"]
        .mean()
        .sort_values(ascending=False)
        .head(5)
        .index
    )

            for emp in top_emp_ids:

                emp_df = temp_df[temp_df["employee_id"] == emp]

        # Trend (date-based)
                if "Date" in emp_df.columns:
                    trend = (
                emp_df.groupby("Date")["score"]
                .mean()
                .reset_index()
                .to_dict("records")
            )
                else:
                    trend = []

                employee_trends.append({
            "employee_id": emp,
            "trend": trend
        })
                
        # ================= DASHBOARD 2 ================= #

        fdemo = temp_df[temp_df["GENDER"].notna()] if "GENDER" in temp_df.columns else temp_df
        fperf = temp_df[temp_df["AHT"].notna() & (temp_df["AHT"] > 0)] if "AHT" in temp_df.columns else temp_df

        # --- Distribution ---
        age_dist = fdemo["AGE"].dropna().tolist() if "AGE" in fdemo.columns else []
        gender_dist = fdemo["GENDER"].value_counts().to_dict() if "GENDER" in fdemo.columns else {}
        worker_dist = fdemo["WORKER_TYPE"].value_counts().to_dict() if "WORKER_TYPE" in fdemo.columns else {}
        job_dist = fdemo["JOB_FAMILY"].value_counts().head(10).to_dict() if "JOB_FAMILY" in fdemo.columns else {}

        # --- Attribute ---
        military_demo = fdemo["MILITARY_SERVICE"].value_counts().to_dict() if "MILITARY_SERVICE" in fdemo.columns else {}
        business_demo = fdemo["BUSINESS_SITE_LOCATION"].value_counts().head(10).to_dict() if "BUSINESS_SITE_LOCATION" in fdemo.columns else {}

        # --- Relationship ---
        age_score = fdemo.groupby("age_group")["score"].mean().reset_index().to_dict("records") \
            if {"age_group","score"}.issubset(fdemo.columns) else []

        age_aht = fperf.groupby("age_group")["AHT_min"].mean().reset_index().to_dict("records") \
            if {"age_group","AHT_min"}.issubset(fperf.columns) else []

        gender_aht = fperf.groupby("GENDER")["AHT_min"].mean().reset_index().to_dict("records") \
            if {"GENDER","AHT_min"}.issubset(fperf.columns) else []

        marital_score = fdemo.groupby("MARITAL_STATUS")["score"].mean().reset_index().to_dict("records") \
            if {"MARITAL_STATUS","score"}.issubset(fdemo.columns) else []

        # --- Performance ---
        job_aht = fperf.groupby("JOB_FAMILY")["AHT_min"].mean().reset_index().to_dict("records") \
            if {"JOB_FAMILY","AHT_min"}.issubset(fperf.columns) else []

        worker_aht = fperf.groupby("WORKER_TYPE")["AHT_min"].mean().reset_index().to_dict("records") \
            if {"WORKER_TYPE","AHT_min"}.issubset(fperf.columns) else []

        perf_quality = fperf[["AHT_min", "score"]].dropna().to_dict("records") \
            if {"AHT_min","score"}.issubset(fperf.columns) else []

        # --- Additional ---
        military_perf = fperf["MILITARY_SERVICE"].value_counts().to_dict() if "MILITARY_SERVICE" in fperf.columns else {}
        business_perf = fperf["BUSINESS_SITE_LOCATION"].value_counts().head(10).to_dict() if "BUSINESS_SITE_LOCATION" in fperf.columns else {}
        # ---------------- STORE ---------------- #

        final_report.append({
            "filters": {
                "City": city,
                "Job Family": job,
                "Status": status
            },
            "report_title": "Overall Employee Analytics Report",

            "charts": [
                {"title": "Performance vs Efficiency", "data": perf_eff, "insight": "Shows the relationship between handling time (AHT) and quality score to identify efficient performers."},
                {"title": "Workload Pressure", "data": workload, "insight": "Indicates how workload (calls answered) impacts performance (score) across different cities."},
                {"title": "Balanced Workforce", "data": {"Balanced": balanced, "Imbalanced": imbalanced}, "insight": "Displays the distribution of employees across different workload categories."},
                {"title": "Account Efficiency", "data": account_eff, "insight": "Measures the efficiency of account management across different regions."},
                {"title": "Job Stability", "data": job_stability, "insight": "Reflects the stability of job roles and employee retention."},
                {"title": "Age Distribution", "data": age_dist, "insight": "Shows the age demographics of the workforce."},
                {"title": "Gender Distribution", "data": gender_dist, "insight": "Displays the gender composition of the organization."},
                {"title": "Worker Type Distribution", "data": worker_dist, "insight": "Shows the distribution of different worker types (e.g., full-time, part-time, contractor)."},
                {"title": "Job Family Distribution", "data": job_dist, "insight": "Displays the distribution of employees across different job families."},

                {"title": "Military Service Distribution (Demographic)", "data": military_demo, "insight": "Shows the distribution of military service status among employees."},
                {"title": "Business Site Distribution (Demographic)", "data": business_demo, "insight": "Displays the geographic distribution of employees across different business sites."},

                {"title": "Age Group vs Quality Score", "data": age_score, "insight": "Compares quality scores across different age groups."},
                {"title": "Age Group vs Avg AHT", "data": age_aht, "insight": "Shows the average handling time across different age groups."},
                {"title": "Gender vs Avg AHT", "data": gender_aht, "insight": "Compares average handling times between different genders."},
                {"title": "Marital Status vs Quality Score", "data": marital_score, "insight": "Analyzes the relationship between marital status and quality scores."},

                {"title": "Job Family vs AHT (min)", "data": job_aht, "insight": "Shows the average handling time across different job families."},
                {"title": "Worker Type vs AHT", "data": worker_aht, "insight": "Compares average handling times between different worker types."},
                {"title": "Performance vs Quality", "data": perf_quality, "insight": "Displays the relationship between performance and quality scores."},

                {"title": "Military Service Distribution (Performance)", "data": military_perf, "insight": "Shows the distribution of military service status among high-performing employees."},
                {"title": "Business Site Distribution (Performance)", "data": business_perf, "insight": "Displays the geographic distribution of high-performing employees across different business sites."}
            
            ],

            "top_performers": top_performers,
            "employee_trends": employee_trends,
            "generated_at": str(datetime.now()),
            "summary": f"Total records: {len(temp_df)}"
        })

    return {
        "report_name": "Overall Employee Analytics Report",
        "total_combinations": len(final_report),
        "data": final_report
    }