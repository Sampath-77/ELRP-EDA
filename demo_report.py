import itertools
import pandas as pd

def generate_demography_report(df):

    final_report = []

    # ✅ Filters
    marital_values = df["MARITAL_STATUS"].dropna().unique()
    location_values = df["BUSINESS_SITE_LOCATION"].dropna().unique()
    city_values = df["CITY_NAME"].dropna().unique()
    status_values = df["CURRENT_STATUS"].dropna().unique()

    combinations = itertools.product(
        marital_values,
        location_values,
        city_values,
        status_values
    )

    for marital, location, city, status in combinations:

        temp_df = df.copy()

        temp_df = temp_df[temp_df["MARITAL_STATUS"] == marital]
        temp_df = temp_df[temp_df["BUSINESS_SITE_LOCATION"] == location]
        temp_df = temp_df[temp_df["CITY_NAME"] == city]
        temp_df = temp_df[temp_df["CURRENT_STATUS"] == status]

        if temp_df.empty:
            continue

        # ---------------- CHART DATA ---------------- #

        # 1. Marital vs Current Status
        marital_status_data = (
            temp_df.groupby(["MARITAL_STATUS", "CURRENT_STATUS"])
            .size()
            .reset_index(name="count")
            .to_dict(orient="records")
        )

        # 2. Gender vs Current Status
        gender_data = (
            temp_df.groupby(["GENDER", "CURRENT_STATUS"])
            .size()
            .reset_index(name="count")
            .to_dict(orient="records")
        )

        # 3. Age Group
        temp_df["AGE_GROUP"] = pd.cut(
            temp_df["AGE"],
            bins=[20, 30, 40, 50, 60],
            labels=["20-30", "30-40", "40-50", "50-60"]
        )

        age_data = (
            temp_df.groupby(["AGE_GROUP", "CURRENT_STATUS"])
            .size()
            .reset_index(name="count")
            .to_dict(orient="records")
        )

        # 4. Termination DV
        termination_data = (
            temp_df.groupby(["TERMINATION_DV", "CURRENT_STATUS"])
            .size()
            .reset_index(name="count")
            .to_dict(orient="records")
        )

        # 5. Job Family
        job_data = temp_df["JOB_FAMILY"].value_counts().to_dict()

        # 6. Management Level
        mgmt_data = temp_df["MANAGEMENT_LEVEL"].value_counts().to_dict()

        # 7. Attrition Rate
        attrition = (
            temp_df.groupby("JOB_FAMILY")["CURRENT_STATUS"]
            .value_counts(normalize=True)
            .mul(100)
            .rename("percentage")
            .reset_index()
        )
        attrition = attrition[attrition["CURRENT_STATUS"] == "TERMINATED"]

        # 8. Age Distribution
        age_dist = temp_df["AGE"].value_counts().to_dict()

        # 9. State Distribution
        state_data = temp_df["STATE_NAME"].value_counts().to_dict()

        # 10. City Distribution
        city_data = temp_df["CITY_NAME"].value_counts().to_dict()

        # 11. Worker Type
        worker_data = temp_df["WORKER_TYPE"].value_counts().to_dict()

        # 12. Geo Zone
        geo_data = temp_df["GEOZONE_CODE"].value_counts().to_dict()

        # 13. Military Service
        military_data = temp_df["MILITARY_SERVICE"].value_counts().to_dict()

        # 14. Business Location
        location_data = temp_df["BUSINESS_SITE_LOCATION"].value_counts().to_dict()

        # 15. Year Trend
        year_data = temp_df["YEAR"].value_counts().sort_index().to_dict()

        # 16. Quarter Trend
        quarter_data = temp_df["QUARTER"].value_counts().sort_index().to_dict()

        # 17. Age vs Gender
        age_gender = (
            temp_df.groupby(["AGE", "GENDER"])
            .size()
            .reset_index(name="count")
            .to_dict(orient="records")
        )

        # 18. Marital vs Gender
        marital_gender = (
            temp_df.groupby(["MARITAL_STATUS", "GENDER"])
            .size()
            .reset_index(name="count")
            .to_dict(orient="records")
        )

        # ---------------- STORE EVERYTHING ---------------- #

        final_report.append({
            "filters": {
                "Marital Status": marital,
                "Business Location": location,
                "City": city,
                "Current Status": status
            },
            "charts": [
                {"title": "Marital vs Current Status", "data": marital_status_data},
                {"title": "Gender vs Current Status", "data": gender_data},
                {"title": "Age vs Current Status", "data": age_data},
                {"title": "Termination DV vs Status", "data": termination_data},
                {"title": "Job Family Distribution", "data": job_data},
                {"title": "Management Level", "data": mgmt_data},
                {"title": "Attrition Rate", "data": attrition.to_dict(orient="records")},
                {"title": "Age Distribution", "data": age_dist},
                {"title": "State Distribution", "data": state_data},
                {"title": "City Distribution", "data": city_data},
                {"title": "Worker Type", "data": worker_data},
                {"title": "Geo Zone", "data": geo_data},
                {"title": "Military Service", "data": military_data},
                {"title": "Business Location Distribution", "data": location_data},
                {"title": "Year Trend", "data": year_data},
                {"title": "Quarter Trend", "data": quarter_data},
                {"title": "Age vs Gender", "data": age_gender},
                {"title": "Marital vs Gender", "data": marital_gender}
            ],
            "insight": f"Total employees: {len(temp_df)}"
        })

    return final_report