import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
import itertools
from datetime import datetime



def show():
    
# ─── Helpers ─────────────────────────────────────────────────────────────────
    def get_hover_col(df):
        for col in ['Agent_ID', 'employee_id', 'employee_id_x', 'AGENTID', 'agentid']:
            if col in df.columns:
                return col
        return None

    def clamp_df(df, col, lo=0, hi=100):
        return df[(df[col] >= lo) & (df[col] <= hi)].copy()

    def safe_json(obj):
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, float) and (obj != obj):
            return None
        if hasattr(obj, 'item'):
            return obj.item()
        raise TypeError(f"Not serialisable: {type(obj)}")

    def df_to_records(df: pd.DataFrame) -> list:
        return json.loads(df.to_json(orient='records', date_format='iso', default_handler=str))


    # ─── Column quality helpers ───────────────────────────────────────────────────
    # Columns to always exclude from filters / bivariate selectors (IDs, names, etc.)
    ID_LIKE_PATTERNS = ('agentid', 'agent_id', 'employee_id', 'employeeid',
                        'agent_name', 'agentname', 'name', 'id')

    def is_id_like(col_name: str) -> bool:
        """Return True if the column name looks like a unique-identifier column."""
        cn = col_name.lower().strip()
        return any(cn == p or cn.endswith('_' + p) or cn.startswith(p + '_') or cn == p
                for p in ID_LIKE_PATTERNS)

    def clean_coaching_df(df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop columns that are useless for analysis:
        • Unnamed columns (Excel artefacts)
        • Columns where the number of non-null rows is ≤ 1
            (these appear after SEVENDAYEFFECTIVENESS in the coaching sheet)
        Returns a copy with those columns removed.
        """
        keep = []
        for col in df.columns:
            # Drop unnamed columns
            if str(col).strip().lower().startswith('unnamed:'):
                continue
            # Drop columns with ≤ 1 non-null value (nearly empty)
            if df[col].count() <= 1:
                continue
            keep.append(col)
        return df[keep].copy()


    def usable_cat_cols_for_filter(df: pd.DataFrame,
                                    exclude_id_like: bool = True,
                                    exclude_patterns: tuple = ()) -> list:
        """
        Return object columns that are suitable as multiselect filters:
        - dtype == object
        - 1 < nunique < 20   (not constant, not all-unique)
        - not ID-like (if exclude_id_like)
        - not matching extra exclude_patterns
        """
        cols = []
        for c in df.columns:
            if df[c].dtype != 'object':
                continue
            nu = df[c].nunique()
            if nu <= 1 or nu >= 20:
                continue
            if exclude_id_like and is_id_like(c):
                continue
            cl = c.lower()
            if any(p in cl for p in exclude_patterns):
                continue
            cols.append(c)
        return cols


    def usable_cols_for_bivariate(df: pd.DataFrame) -> list:
        """
        Columns suitable as axes in bivariate analysis.
        Excludes unnamed, near-empty, and ID-like columns.
        """
        cols = []
        for c in df.columns:
            if str(c).strip().lower().startswith('unnamed:'):
                continue
            if df[c].count() <= 1:
                continue
            if is_id_like(c):
                continue
            cols.append(c)
        return cols

    @st.cache_data
    def load_and_prep_data(path, sheet_name=None):
        df = pd.read_excel(path, sheet_name=sheet_name) if sheet_name else pd.read_excel(path)
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    if 'date' in col.lower() or 'time' in col.lower() or df[col].astype(str).str.match(r'^\d{4}-\d{2}-\d{2}').all():
                        df[col] = pd.to_datetime(df[col])
                    else:
                        df[col] = df[col].astype(str)
                except (ValueError, TypeError):
                    df[col] = df[col].astype(str)
        return df

    @st.cache_data
    def get_sheet_names(path):
        return pd.ExcelFile(path).sheet_names

    METRIC_META = {
        "QA SCORE":                 {"label": "Score (0–100)",       "range": [0, 100]},
        "CSAT":                     {"label": "Score (0–100 %)",     "range": [0, 100]},
        "CUSTOMER EFFORT":          {"label": "Score (0–100)",       "range": [0, 100]},
        "FCR":                      {"label": "Rate (0–100 %)",      "range": [0, 100]},
        "NPS":                      {"label": "Score (-100 to 100)", "range": [-100, 100]},
        "OFFER RATE":               {"label": "Rate (0–100 %)",      "range": [0, 100]},
        "PCS":                      {"label": "Score (0–100)",       "range": [0, 100]},
        "RESPONSE RATE":            {"label": "Rate (0–100 %)",      "range": [0, 100]},
        "ACE QA SCORE":             {"label": "Score (0–100)",       "range": [0, 100]},
        "NEF ACE NON CALL QUALITY": {"label": "Score (0–100)",       "range": [0, 100]},
    }

    FILE_PATH = "Data/Quality_&_Coaching_Data_Nov_2025.xlsx"
    PERF_PATH = "Data/PH_Agent_Performance_Nov_2025 (1).csv"

    # ══════════════════════════════════════════════════════════════════════════════
    # REPORT BUILDERS
    # ══════════════════════════════════════════════════════════════════════════════

    def build_quality_json(df_q_raw, SCORE_COL, EMP_COL, METRIC_COL):
        """JSON 1 — Quality data only: distributions, trends, agent-level scores."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "description":  "Quality data export — per-metric, per-filter-combination",
            "sections": []
        }
        q_col_map  = {c.lower(): c for c in df_q_raw.columns}
        GEO_COL    = q_col_map.get('geo')
        LOB_COL    = q_col_map.get('lob')
        DATE_COL   = q_col_map.get('date')

        metric_vals = (["All Metrics Combined"] + df_q_raw[METRIC_COL].dropna().unique().tolist()) if METRIC_COL else ["All Metrics Combined"]
        geo_vals    = (["All"] + df_q_raw[GEO_COL].dropna().unique().tolist()) if GEO_COL else ["All"]
        lob_vals    = (["All"] + df_q_raw[LOB_COL].dropna().unique().tolist()) if LOB_COL else ["All"]

        for metric_sel, geo_sel, lob_sel in itertools.product(metric_vals, geo_vals, lob_vals):
            combo_label = f"metric={metric_sel} | geo={geo_sel} | lob={lob_sel}"
            section = {"filter_combination": {"metric": metric_sel, "geo": geo_sel, "lob": lob_sel}, "charts": []}

            df_q = df_q_raw.copy()
            if METRIC_COL and metric_sel != "All Metrics Combined":
                df_q = df_q[df_q[METRIC_COL] == metric_sel]
            if GEO_COL and geo_sel != "All":
                df_q = df_q[df_q[GEO_COL] == geo_sel]
            if LOB_COL and lob_sel != "All":
                df_q = df_q[df_q[LOB_COL] == lob_sel]

            meta = METRIC_META.get(metric_sel, {"label": "Score", "range": [0, 100]}) if metric_sel != "All Metrics Combined" else {"label": "Score", "range": [0, 100]}
            df_qc = clamp_df(df_q, SCORE_COL, meta["range"][0], meta["range"][1]) if metric_sel != "All Metrics Combined" else df_q.copy()

            def add(title, desc, data_df, extra=None):
                e = {"chart_title": title, "description": desc, "filter_combo": combo_label,
                    "data": df_to_records(data_df) if data_df is not None else []}
                if extra: e["annotations"] = extra
                section["charts"].append(e)

            try:
                agg = df_qc.groupby(EMP_COL)[SCORE_COL].agg(['mean', 'count', 'std']).reset_index()
                agg.columns = [EMP_COL, 'Avg_Score', 'Record_Count', 'Std_Score']
                add("Agent Average Scores", "Per-agent mean/count/std of quality scores.", agg)

                dist = df_qc[SCORE_COL].describe().reset_index()
                dist.columns = ['Statistic', 'Value']
                add("Score Distribution Summary", "Descriptive statistics for quality scores.", dist)

                if LOB_COL:
                    lob_agg = df_qc.groupby(LOB_COL)[SCORE_COL].agg(['mean', 'count']).reset_index()
                    lob_agg.columns = [LOB_COL, 'Avg_Score', 'Count']
                    add("Score by LOB", "Average quality score per Line of Business.", lob_agg)

                if GEO_COL:
                    geo_agg = df_qc.groupby(GEO_COL)[SCORE_COL].agg(['mean', 'count']).reset_index()
                    geo_agg.columns = [GEO_COL, 'Avg_Score', 'Count']
                    add("Score by Geo", "Average quality score per geography.", geo_agg)

                if DATE_COL:
                    df_qc2 = df_qc.copy()
                    df_qc2[DATE_COL] = pd.to_datetime(df_qc2[DATE_COL], errors='coerce')
                    for fl, fr in [("Daily", "D"), ("Weekly", "W-MON"), ("Monthly", "ME")]:
                        tr = df_qc2.dropna(subset=[DATE_COL]).groupby(
                            pd.Grouper(key=DATE_COL, freq=fr))[SCORE_COL].mean().reset_index()
                        tr.rename(columns={DATE_COL: 'Date', SCORE_COL: 'Avg_Score'}, inplace=True)
                        add(f"Quality Trend – {fl}", f"{fl} average quality score over time.", tr)
            except Exception as ex:
                section["error"] = str(ex)

            report["sections"].append(section)
        return report


    def build_coaching_json(df_c_raw, AGENT_COL):
        """JSON 2 — Coaching data only: session distributions, behaviors, timelines."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "description":  "Coaching data export — session distributions and behavior breakdowns",
            "sections": []
        }
        c_col_map       = {c.lower(): c for c in df_c_raw.columns}
        DATECOACHED_COL = c_col_map.get('datecoached')
        BEHAV_COL       = next((c for c in df_c_raw.columns if 'behavior' in c.lower() or 'behaviour' in c.lower()), None)
        GEO_COL_C       = next((c for c in df_c_raw.columns if c.lower() == 'geo'), None)

        section = {"filter_combination": {"note": "Coaching data — no sub-filters"}, "charts": []}

        def add(title, desc, data_df, extra=None):
            e = {"chart_title": title, "description": desc,
                "data": df_to_records(data_df) if data_df is not None else []}
            if extra: e["annotations"] = extra
            section["charts"].append(e)

        try:
            cc = df_c_raw.groupby(AGENT_COL).size().reset_index(name='Coaching_Count')
            add("Coaching Sessions per Agent", "Total coaching sessions each agent received.", cc)

            dist = cc.groupby('Coaching_Count').size().reset_index(name='Agent_Count')
            add("Distribution of Coaching Volumes", "How many agents received N sessions.", dist,
                extra={"total_agents": int(len(cc)), "total_sessions": int(cc['Coaching_Count'].sum()),
                    "avg_sessions": round(float(cc['Coaching_Count'].mean()), 2)})

            if BEHAV_COL:
                beh = df_c_raw.groupby(BEHAV_COL)[AGENT_COL].count().reset_index()
                beh.columns = [BEHAV_COL, 'Session_Count']
                beh = beh.sort_values('Session_Count', ascending=False)
                add("Coaching Sessions by Behavior", "Session count per coached behavior.", beh)

            if DATECOACHED_COL:
                df_cd = df_c_raw.copy()
                df_cd[DATECOACHED_COL] = pd.to_datetime(df_cd[DATECOACHED_COL], errors='coerce')
                for fl, fr in [("Daily", "D"), ("Weekly", "W-MON"), ("Monthly", "ME")]:
                    tr = df_cd.dropna(subset=[DATECOACHED_COL]).groupby(
                        pd.Grouper(key=DATECOACHED_COL, freq=fr))[AGENT_COL].count().reset_index()
                    tr.rename(columns={DATECOACHED_COL: 'Date', AGENT_COL: 'Session_Count'}, inplace=True)
                    add(f"Coaching Volume Trend – {fl}", f"{fl} total coaching sessions over time.", tr)

            if GEO_COL_C:
                geo = df_c_raw.groupby(GEO_COL_C)[AGENT_COL].count().reset_index()
                geo.columns = [GEO_COL_C, 'Session_Count']
                add("Coaching by Geo", "Session count per geography.", geo)

        except Exception as ex:
            section["error"] = str(ex)

        report["sections"].append(section)
        return report


    def build_cross_json(df_q_raw, df_c_raw, SCORE_COL, EMP_COL, AGENT_COL, METRIC_COL):
        """JSON 3 — Quality × Coaching cross analysis (no performance data)."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "description":  "Quality × Coaching cross-analysis — before/after, correlations, intensity buckets",
            "sections": []
        }
        q_col_map       = {c.lower(): c for c in df_q_raw.columns}
        c_col_map       = {c.lower(): c for c in df_c_raw.columns}
        GEO_COL         = q_col_map.get('geo')
        LOB_COL         = q_col_map.get('lob')
        DATECOACHED_COL = c_col_map.get('datecoached')
        DATE_COL        = q_col_map.get('date')

        metric_vals = (["All Metrics Combined"] + df_q_raw[METRIC_COL].dropna().unique().tolist()) if METRIC_COL else ["All Metrics Combined"]
        geo_vals    = (["All"] + df_q_raw[GEO_COL].dropna().unique().tolist()) if GEO_COL else ["All"]
        lob_vals    = (["All"] + df_q_raw[LOB_COL].dropna().unique().tolist()) if LOB_COL else ["All"]

        for metric_sel, geo_sel, lob_sel in itertools.product(metric_vals, geo_vals, lob_vals):
            combo_label = f"metric={metric_sel} | geo={geo_sel} | lob={lob_sel}"
            section = {"filter_combination": {"metric": metric_sel, "geo": geo_sel, "lob": lob_sel}, "charts": []}

            df_q = df_q_raw.copy()
            if METRIC_COL and metric_sel != "All Metrics Combined":
                df_q = df_q[df_q[METRIC_COL] == metric_sel]
            if GEO_COL and geo_sel != "All":
                df_q = df_q[df_q[GEO_COL] == geo_sel]
            if LOB_COL and lob_sel != "All":
                df_q = df_q[df_q[LOB_COL] == lob_sel]

            meta = METRIC_META.get(metric_sel, {"label": "Score", "range": [0, 100]}) if metric_sel != "All Metrics Combined" else {"label": "Score", "range": [0, 100]}
            df_qc = clamp_df(df_q, SCORE_COL, meta["range"][0], meta["range"][1]) if metric_sel != "All Metrics Combined" else df_q.copy()

            def add(title, desc, data_df, extra=None):
                e = {"chart_title": title, "description": desc, "filter_combo": combo_label,
                    "data": df_to_records(data_df) if data_df is not None else []}
                if extra: e["annotations"] = extra
                section["charts"].append(e)

            try:
                df_q_agg = df_qc.groupby(EMP_COL)[SCORE_COL].mean().reset_index()
                df_q_agg.rename(columns={SCORE_COL: 'Avg_Quality_Score'}, inplace=True)
                agg_dict = {AGENT_COL: 'count'}
                behav_matches = [c for c in df_c_raw.columns if 'behavior' in c.lower() or 'behaviour' in c.lower()]
                BEHAV_COL = behav_matches[0] if behav_matches else None
                if BEHAV_COL:
                    agg_dict[BEHAV_COL] = lambda x: ', '.join(x.dropna().astype(str).unique())
                df_c_agg = df_c_raw.groupby(AGENT_COL).agg(agg_dict)
                df_c_agg.rename(columns={AGENT_COL: 'Coaching_Sessions'}, inplace=True)
                df_c_agg = df_c_agg.reset_index()
                df_merged = pd.merge(df_q_agg, df_c_agg, left_on=EMP_COL, right_on=AGENT_COL, how='inner')
                add("Overview – Avg Quality Score vs Coaching Sessions",
                    "Per-agent avg quality score vs coaching session count.",
                    df_merged[[EMP_COL, 'Avg_Quality_Score', 'Coaching_Sessions']])

                # Before / After  — FIX: changed extra_text= → extra=
                if DATECOACHED_COL and DATE_COL:
                    df_cd = df_c_raw.copy()
                    df_qd = df_qc.copy()
                    df_cd[DATECOACHED_COL] = pd.to_datetime(df_cd[DATECOACHED_COL], errors='coerce')
                    df_qd[DATE_COL]        = pd.to_datetime(df_qd[DATE_COL], errors='coerce')
                    df_cf = df_cd.groupby(AGENT_COL)[DATECOACHED_COL].min().reset_index()
                    df_qt = pd.merge(df_qd, df_cf, left_on=EMP_COL, right_on=AGENT_COL, how='inner')
                    df_qt['Period'] = df_qt.apply(
                        lambda r: 'After Coaching' if pd.notnull(r[DATE_COL]) and pd.notnull(r[DATECOACHED_COL])
                                and r[DATE_COL] >= r[DATECOACHED_COL] else 'Before Coaching', axis=1)
                    ia = df_qt.groupby(['Period', EMP_COL])[SCORE_COL].mean().reset_index()
                    io = ia.groupby('Period')[SCORE_COL].mean().reset_index()
                    extra = {}
                    if len(io) == 2:
                        vb = io[io['Period'] == 'Before Coaching'][SCORE_COL].values
                        va = io[io['Period'] == 'After Coaching'][SCORE_COL].values
                        if len(vb) and len(va):
                            lift = ((va[0] - vb[0]) / vb[0] * 100) if vb[0] > 0 else 0
                            extra = {"avg_before": round(float(vb[0]), 2), "avg_after": round(float(va[0]), 2),
                                    "lift_pct": round(float(lift), 2)}
                    # FIX: was extra_text=extra (invalid kwarg) → extra=extra
                    add("Before vs After – Overall", "Avg quality score before and after first coaching.",
                        io.rename(columns={SCORE_COL: 'Avg_Score'}), extra=extra)

                    try:
                        pivot = ia.pivot(index=EMP_COL, columns='Period', values=SCORE_COL).dropna().reset_index()
                        if 'Before Coaching' in pivot.columns and 'After Coaching' in pivot.columns:
                            pivot['Improvement'] = pivot['After Coaching'] - pivot['Before Coaching']
                            pivot = pivot.merge(df_c_agg[[AGENT_COL, 'Coaching_Sessions']],
                                                left_on=EMP_COL, right_on=AGENT_COL, how='left')
                            pivot['Coaching Efficiency'] = pivot['Improvement'] / pivot['Coaching_Sessions'].replace(0, 1)
                            add("Before vs After – Agent-Level", "Per-agent improvement and coaching efficiency.", pivot)
                    except Exception:
                        pass

                    for fl, fr in [("Daily", "D"), ("Weekly", "W-MON"), ("Monthly", "ME")]:
                        try:
                            dq2 = df_qc.copy(); dq2[DATE_COL] = pd.to_datetime(dq2[DATE_COL], errors='coerce')
                            dc2 = df_c_raw.copy(); dc2[DATECOACHED_COL] = pd.to_datetime(dc2[DATECOACHED_COL], errors='coerce')
                            ct = dc2.dropna(subset=[DATECOACHED_COL]).groupby(pd.Grouper(key=DATECOACHED_COL, freq=fr))[AGENT_COL].count().reset_index()
                            ct.rename(columns={DATECOACHED_COL: 'Date', AGENT_COL: 'Coaching_Sessions'}, inplace=True)
                            qt = dq2.dropna(subset=[DATE_COL]).groupby(pd.Grouper(key=DATE_COL, freq=fr))[SCORE_COL].mean().reset_index()
                            qt.rename(columns={DATE_COL: 'Date', SCORE_COL: 'Avg_Quality_Score'}, inplace=True)
                            mt = pd.merge(ct, qt, on='Date', how='outer').sort_values('Date')
                            mt['Coaching_Sessions'] = mt['Coaching_Sessions'].fillna(0)
                            add(f"Trend – {fl}", f"{fl} coaching volume vs avg quality score.", mt)
                        except Exception:
                            pass

                for m in (df_q_raw[METRIC_COL].dropna().unique().tolist() if METRIC_COL else ["QA SCORE"]):
                    mm = METRIC_META.get(m, {"label": "Score", "range": [0, 100]})
                    df_qm = (df_q[df_q[METRIC_COL] == m].copy() if METRIC_COL else df_q.copy())
                    df_qm = clamp_df(df_qm, SCORE_COL, mm['range'][0], mm['range'][1])
                    if DATE_COL and DATECOACHED_COL:
                        df_qm[DATE_COL] = pd.to_datetime(df_qm[DATE_COL], errors='coerce')
                        dc = df_c_raw.copy(); dc[DATECOACHED_COL] = pd.to_datetime(dc[DATECOACHED_COL], errors='coerce')
                        fc = dc.groupby(AGENT_COL)[DATECOACHED_COL].min().reset_index()
                        fc.rename(columns={DATECOACHED_COL: 'First_Coaching_Date'}, inplace=True)
                        dfa = pd.merge(df_qm, fc, left_on=EMP_COL, right_on=AGENT_COL, how='inner')
                        dfa = dfa[dfa[DATE_COL] >= dfa['First_Coaching_Date']]
                        aa  = dfa.groupby(EMP_COL)[SCORE_COL].mean().reset_index(); aa.rename(columns={SCORE_COL: 'Avg_Score_After'}, inplace=True)
                        dfb = pd.merge(df_qm, fc, left_on=EMP_COL, right_on=AGENT_COL, how='inner')
                        dfb = dfb[dfb[DATE_COL] < dfb['First_Coaching_Date']]
                        ba  = dfb.groupby(EMP_COL)[SCORE_COL].mean().reset_index(); ba.rename(columns={SCORE_COL: 'Avg_Score_Before'}, inplace=True)
                        cc2 = df_c_raw.groupby(AGENT_COL).size().reset_index(name='Coaching_Count')
                        df_fin = pd.merge(cc2, aa, left_on=AGENT_COL, right_on=EMP_COL, how='inner')
                        if EMP_COL in df_fin.columns and EMP_COL != AGENT_COL: df_fin.drop(columns=[EMP_COL], inplace=True)
                        df_fin = df_fin[(df_fin['Avg_Score_After'] >= mm['range'][0]) & (df_fin['Avg_Score_After'] <= mm['range'][1])]
                        df_fin = pd.merge(df_fin, ba, left_on=AGENT_COL, right_on=EMP_COL, how='left')
                        if EMP_COL in df_fin.columns and EMP_COL != AGENT_COL: df_fin.drop(columns=[EMP_COL], inplace=True)
                        if not df_fin.empty:
                            cr = df_fin[['Coaching_Count', 'Avg_Score_After']].corr().iloc[0, 1]
                            add(f"Coaching Count vs Avg {m}", f"Pearson r={cr:.2f}", df_fin,
                                extra={"pearson_r": round(float(cr), 4), "metric": m, "metric_range": mm['range']})
                            df_fin2 = df_fin.copy()
                            df_fin2['Coaching_Bucket'] = pd.cut(df_fin2['Coaching_Count'], bins=[0,2,5,10,50],
                                labels=['Low (1-2)', 'Medium (3-5)', 'High (6-10)', 'Very High (11+)'])
                            bkt = df_fin2.groupby('Coaching_Bucket', observed=True).agg(
                                Avg_Score_After=('Avg_Score_After','mean'), Agent_Count=('Avg_Score_After','count')).reset_index()
                            bkt['Coaching_Bucket'] = bkt['Coaching_Bucket'].astype(str)
                            add(f"Coaching Intensity Buckets – {m}", "Avg score by coaching volume bucket.", bkt)
                            top15 = df_fin.sort_values('Coaching_Count', ascending=False).head(15).copy()
                            top15['Improvement'] = top15['Avg_Score_After'] - top15['Avg_Score_Before']
                            add(f"Top 15 Most-Coached – {m}", "Top 15 agents before/after scores.", top15)

                cd = df_c_raw.groupby(AGENT_COL).size().reset_index(name='Coaching_Count')
                da = cd.groupby('Coaching_Count').size().reset_index(name='Agent_Count')
                add("Distribution of Coaching Sessions", "How many agents received N coaching sessions.", da)

            except Exception as ex:
                section["error"] = str(ex)

            report["sections"].append(section)
        return report


    # ──────────────────────────────────────────────────────────────────────────────
    # Helper: build quality-sheet download JSON (for single Quality sheet tab)
    # ──────────────────────────────────────────────────────────────────────────────
    def build_quality_sheet_json(df_q_raw):
        q_col_map  = {c.lower(): c for c in df_q_raw.columns}
        SCORE_COL  = q_col_map.get('score')
        EMP_COL    = q_col_map.get('employee_id')
        METRIC_COL = q_col_map.get('metric_name')
        if not (SCORE_COL and EMP_COL):
            return None, "Required columns (score / employee_id) not found."
        return build_quality_json(df_q_raw, SCORE_COL, EMP_COL, METRIC_COL), None


    def build_coaching_sheet_json(df_c_raw):
        c_col_map = {c.lower(): c for c in df_c_raw.columns}
        AGENT_COL = c_col_map.get('agentid')
        if not AGENT_COL:
            return None, "Required column (agentid) not found."
        return build_coaching_json(df_c_raw, AGENT_COL), None


    # ══════════════════════════════════════════════════════════════════════════════
    # MAIN APP
    # ══════════════════════════════════════════════════════════════════════════════
    if os.path.exists(FILE_PATH):
        try:
            sheet_names = get_sheet_names(FILE_PATH)
            if len(sheet_names) > 1:
                st.markdown("### 🗄️ Database Perspective")
                cross_analysis_label = "Cross-Sheet Analysis (Quality vs Coaching)"
                sheet_options = list(sheet_names)
                if 'Quality Data ' in sheet_names and 'Coaching Data' in sheet_names:
                    sheet_options.append(cross_analysis_label)
                selected_sheet = st.radio("Select which dataset to analyze:", sheet_options, horizontal=True)

                if selected_sheet == cross_analysis_label:
                    st.markdown("---")
                    st.header("🤝 Quality vs Coaching Relationship")

                    df_q = load_and_prep_data(FILE_PATH, sheet_name='Quality Data ')
                    df_c = load_and_prep_data(FILE_PATH, sheet_name='Coaching Data')

                    st.sidebar.markdown("### 🎛️ Dynamic Quality Filters")
                    st.sidebar.caption("Filter Quality data dynamically:")
                    cat_cols_q = [c for c in df_q.columns if df_q[c].dtype == 'object'
                                and df_q[c].nunique() < 20 and df_q[c].nunique() > 0
                                and c.lower() != 'metric_name']
                    for col in cat_cols_q:
                        unique_vals = sorted([str(x) for x in df_q[col].dropna().unique()])
                        selected_vals = st.sidebar.multiselect(f"Q: Filter {col}", unique_vals, default=unique_vals)
                        if selected_vals:
                            df_q = df_q[df_q[col].astype(str).isin(selected_vals)]

                    st.sidebar.markdown("### 🎛️ Dynamic Coaching Filters")
                    st.sidebar.caption("Filter Coaching data dynamically:")
                    # FIX: skip columns with all-unique values (like agent IDs) to avoid unnamed/useless filters
                    cat_cols_c = [c for c in df_c.columns if df_c[c].dtype == 'object'
                                and df_c[c].nunique() < 20 and df_c[c].nunique() > 0
                                and c.lower() not in ('agentid', 'agent_id', 'employee_id')]
                    for col in cat_cols_c:
                        unique_vals = sorted([str(x) for x in df_c[col].dropna().unique()])
                        selected_vals = st.sidebar.multiselect(f"C: Filter {col}", unique_vals, default=unique_vals)
                        if selected_vals:
                            df_c = df_c[df_c[col].astype(str).isin(selected_vals)]

                    q_cols = {c.lower(): c for c in df_q.columns}
                    c_cols = {c.lower(): c for c in df_c.columns}

                    SCORE_COL  = q_cols.get('score')
                    EMP_COL    = q_cols.get('employee_id')
                    AGENT_COL  = c_cols.get('agentid')
                    METRIC_COL = q_cols.get('metric_name')

                    if SCORE_COL and EMP_COL and AGENT_COL:

                        df_q_agg = df_q.groupby(EMP_COL)[SCORE_COL].mean().reset_index()
                        df_q_agg.rename(columns={SCORE_COL: 'Avg_Quality_Score'}, inplace=True)

                        agg_dict    = {AGENT_COL: 'count'}
                        eff_matches = [c for c in df_c.columns if 'effectiveness' in c.lower()
                                    and pd.api.types.is_numeric_dtype(df_c[c])]
                        if eff_matches:
                            agg_dict[eff_matches[0]] = 'mean'
                        behav_col_matches = [c for c in df_c.columns
                                            if 'behavior' in c.lower() or 'behaviour' in c.lower()]
                        BEHAV_COL = behav_col_matches[0] if behav_col_matches else None
                        if BEHAV_COL:
                            agg_dict[BEHAV_COL] = lambda x: ', '.join(x.dropna().astype(str).unique())

                        df_c_agg = df_c.groupby(AGENT_COL).agg(agg_dict)
                        rename_map = {AGENT_COL: 'Coaching_Sessions'}
                        if eff_matches:   rename_map[eff_matches[0]] = 'Avg_7Day_Effectiveness'
                        if BEHAV_COL:     rename_map[BEHAV_COL]       = 'Behavior'
                        df_c_agg.rename(columns=rename_map, inplace=True)
                        df_c_agg = df_c_agg.reset_index()

                        df_merged = pd.merge(df_q_agg, df_c_agg, left_on=EMP_COL, right_on=AGENT_COL, how='inner')

                        if not df_merged.empty:
                            (tab_overview, tab_impact, tab_behav_geo,
                            tab_trend, tab_coaching, tab_performance, tab_report) = st.tabs([
                                "📊 Overview",
                                "⏳ Before vs After Impact",
                                "🌍 Behaviors & Locations",
                                "📈 Trend Over Time",
                                "🎯 Coaching vs Score Analysis",
                                "🚀 Performance Data Analysis",
                                "⬇️ Download Report"
                            ])

                            # ══════════════════════════════════════════════════════
                            # TAB 1 – OVERVIEW
                            # ══════════════════════════════════════════════════════
                            with tab_overview:
                                k1, k2, k3 = st.columns(3)
                                k1.metric("Total Merged Agents", f"{len(df_merged)}")
                                k2.metric("Overall Avg Quality (Coached Agents)", f"{df_merged['Avg_Quality_Score'].mean():.2f}")
                                k3.metric("Total Coaching Sessions", f"{df_merged['Coaching_Sessions'].sum()}")

                                c1, c2 = st.columns(2)
                                with c1:
                                    unique_metrics_in_view = df_q[METRIC_COL].dropna().unique().tolist() if METRIC_COL else []
                                    is_qa_only = (len(unique_metrics_in_view) == 1 and unique_metrics_in_view[0] == "QA SCORE")

                                    df_scatter = df_merged.copy()
                                    if is_qa_only:
                                        df_scatter = df_scatter[
                                            (df_scatter['Avg_Quality_Score'] >= 0) &
                                            (df_scatter['Avg_Quality_Score'] <= 100)]

                                    fig_scatter = px.scatter(
                                        df_scatter, x='Coaching_Sessions', y='Avg_Quality_Score',
                                        title="Avg Quality Score vs Coaching Sessions"
                                            + (" (0–100 clamped)" if is_qa_only else ""),
                                        hover_name=get_hover_col(df_scatter), template="plotly_white",
                                        size='Coaching_Sessions', color='Avg_Quality_Score',
                                        color_continuous_scale="Viridis")
                                    if is_qa_only:
                                        fig_scatter.update_yaxes(range=[0, 100])
                                    st.plotly_chart(fig_scatter, use_container_width=True)

                                with c2:
                                    num_cols_merged = df_merged.select_dtypes(include=['float64', 'int64']).columns.tolist()
                                    if len(num_cols_merged) > 1:
                                        df_corr = df_merged[num_cols_merged].copy()
                                        if 'Avg_Quality_Score' in df_corr.columns and is_qa_only:
                                            df_corr = df_corr[(df_corr['Avg_Quality_Score'] >= 0) & (df_corr['Avg_Quality_Score'] <= 100)]
                                        fig_corr = px.imshow(df_corr.corr(), text_auto=".2f",
                                                            aspect="auto", title="Correlation Heatmap",
                                                            color_continuous_scale="RdBu_r", template="plotly_white")
                                        st.plotly_chart(fig_corr, use_container_width=True)

                                st.markdown("### Top Coached Agents")
                                st.dataframe(df_merged.sort_values(by='Coaching_Sessions', ascending=False),
                                            use_container_width=True)

                            # ══════════════════════════════════════════════════════
                            # TAB 2 – BEFORE vs AFTER IMPACT
                            # ══════════════════════════════════════════════════════
                            with tab_impact:
                                DATECOACHED_COL = c_cols.get('datecoached')
                                DATE_COL        = q_cols.get('date')

                                if DATECOACHED_COL and DATE_COL:
                                    df_c_dates = df_c.copy()
                                    df_q_dates = df_q.copy()
                                    df_c_dates[DATECOACHED_COL] = pd.to_datetime(df_c_dates[DATECOACHED_COL], errors='coerce')
                                    df_q_dates[DATE_COL]        = pd.to_datetime(df_q_dates[DATE_COL], errors='coerce')
                                    df_q_dates = clamp_df(df_q_dates, SCORE_COL, 0, 100)

                                    df_c_first_date = df_c_dates.groupby(AGENT_COL)[DATECOACHED_COL].min().reset_index()
                                    df_q_time = pd.merge(df_q_dates, df_c_first_date,
                                                        left_on=EMP_COL, right_on=AGENT_COL, how='inner')
                                    df_q_time['Period'] = df_q_time.apply(
                                        lambda row: 'Avg Quality After Coaching'
                                        if pd.notnull(row[DATE_COL]) and pd.notnull(row[DATECOACHED_COL])
                                        and row[DATE_COL] >= row[DATECOACHED_COL]
                                        else 'Avg Quality Before Coaching', axis=1)

                                    impact_agg     = df_q_time.groupby(['Period', EMP_COL])[SCORE_COL].mean().reset_index()
                                    impact_overall = impact_agg.groupby('Period')[SCORE_COL].mean().reset_index()

                                    if not impact_overall.empty and len(impact_overall) == 2:
                                        val_before = impact_overall[impact_overall['Period'] == 'Avg Quality Before Coaching'][SCORE_COL].values[0]
                                        val_after  = impact_overall[impact_overall['Period'] == 'Avg Quality After Coaching'][SCORE_COL].values[0]
                                        lift_pct   = ((val_after - val_before) / val_before) * 100 if val_before > 0 else 0
                                        total_coached_agents    = impact_agg[EMP_COL].nunique()
                                        overall_coaching_volume = df_merged['Coaching_Sessions'].sum()

                                        st.success(
                                            f"**Correlation Summary:** Across **{total_coached_agents}** agents and "
                                            f"**{overall_coaching_volume}** coaching sessions, the average Quality score "
                                            f"shifted from **{val_before:.2f}** to **{val_after:.2f}**, representing an "
                                            f"aggregate improvement of **{lift_pct:+.2f}%**.")

                                        impact_overall['Total Coaching Sessions'] = [0, overall_coaching_volume]
                                        fig_impact = px.bar(
                                            impact_overall, x='Period', y=SCORE_COL, color='Period',
                                            title="Average Quality Score: Before vs After Coaching",
                                            hover_data=['Total Coaching Sessions'], template="plotly_white",
                                            text_auto=".2f",
                                            category_orders={"Period": ["Avg Quality Before Coaching", "Avg Quality After Coaching"]},
                                            color_discrete_map={"Avg Quality Before Coaching": "#ee3e32",
                                                                "Avg Quality After Coaching": "#32f384"})
                                        fig_impact.update_yaxes(range=[0, 100])
                                        fig_impact.update_layout(showlegend=False)
                                        st.plotly_chart(fig_impact, use_container_width=True)

                                        st.markdown("### Agent Level Breakdown")
                                        try:
                                            pivot_impact = impact_agg.pivot(index=EMP_COL, columns='Period', values=SCORE_COL).dropna()
                                            if not pivot_impact.empty:
                                                if ('Avg Quality Before Coaching' in pivot_impact.columns
                                                        and 'Avg Quality After Coaching' in pivot_impact.columns):
                                                    pivot_impact = pivot_impact[['Avg Quality Before Coaching', 'Avg Quality After Coaching']]
                                                    pivot_impact['Improvement'] = (pivot_impact['Avg Quality After Coaching']
                                                                                    - pivot_impact['Avg Quality Before Coaching'])
                                                    merge_cols = [AGENT_COL, 'Coaching_Sessions']
                                                    if 'Behavior' in df_c_agg.columns: merge_cols.append('Behavior')
                                                    pivot_impact = pivot_impact.merge(df_c_agg[merge_cols],
                                                                                    left_index=True, right_on=AGENT_COL, how='left')
                                                    LOB_COL = q_cols.get('lob')
                                                    if LOB_COL:
                                                        agent_lob = df_q[[EMP_COL, LOB_COL]].dropna().drop_duplicates(subset=[EMP_COL])
                                                        pivot_impact = pivot_impact.merge(agent_lob, left_on=AGENT_COL, right_on=EMP_COL, how='left')
                                                        pivot_impact[LOB_COL] = pivot_impact[LOB_COL].fillna('Unknown')
                                                    pivot_impact.set_index(AGENT_COL, inplace=True)
                                                    pivot_impact.index.name = 'Agent_ID'
                                                    pivot_impact['Coaching Efficiency'] = (pivot_impact['Improvement']
                                                                                            / pivot_impact['Coaching_Sessions'].replace(0, 1))
                                                    cols = ['Coaching_Sessions', 'Behavior']
                                                    if LOB_COL: cols.append(LOB_COL)
                                                    cols.extend(['Avg Quality Before Coaching', 'Avg Quality After Coaching',
                                                                'Improvement', 'Coaching Efficiency'])
                                                    pivot_impact = pivot_impact[[c for c in cols if c in pivot_impact.columns]]
                                                    pivot_impact.rename(columns={'Coaching_Sessions': 'Total Coaching Sessions'}, inplace=True)
                                                    st.dataframe(pivot_impact.style.background_gradient(subset=['Improvement'], cmap='PiYG'),
                                                                use_container_width=True)

                                                    st.markdown("### Core Impact Visualizations")
                                                    c_viz1, c_viz2 = st.columns(2)
                                                    with c_viz1:
                                                        fig_roi = px.scatter(pivot_impact.reset_index(),
                                                                            x='Avg Quality Before Coaching', y='Avg Quality After Coaching',
                                                                            size='Total Coaching Sessions', color='Improvement',
                                                                            hover_name='Agent_ID', color_continuous_scale='RdYlGn',
                                                                            template="plotly_white", title="ROI: Before vs After")
                                                        fig_roi.update_xaxes(range=[0, 100]); fig_roi.update_yaxes(range=[0, 100])
                                                        fig_roi.add_shape(type="line", x0=0, y0=0, x1=100, y1=100,
                                                                        line=dict(color="black", dash="dash"))
                                                        st.plotly_chart(fig_roi, use_container_width=True)
                                                        fig_hist = px.histogram(pivot_impact, x='Improvement', nbins=20,
                                                                                title="Distribution of Improvement Scores",
                                                                                template="plotly_white",
                                                                                color_discrete_sequence=['#32f384'])
                                                        st.plotly_chart(fig_hist, use_container_width=True)
                                                    with c_viz2:
                                                        if 'Behavior' in pivot_impact.columns:
                                                            bdf = pivot_impact.reset_index().copy()
                                                            bdf['Behavior_List'] = bdf['Behavior'].astype(str).str.split(', ')
                                                            bdf = bdf.explode('Behavior_List')
                                                            bagg = (bdf.groupby('Behavior_List')['Improvement'].mean().reset_index()
                                                                    .sort_values('Improvement', ascending=False))
                                                            bagg = bagg[(bagg['Behavior_List'] != 'nan') & (bagg['Behavior_List'] != 'None')]
                                                            fig_beh = px.bar(bagg.head(15), x='Behavior_List', y='Improvement',
                                                                            title="Top Behaviors by Avg Improvement", text_auto=".1f",
                                                                            template="plotly_white", color='Improvement',
                                                                            color_continuous_scale='RdYlGn')
                                                            fig_beh.update_layout(showlegend=False)
                                                            st.plotly_chart(fig_beh, use_container_width=True)
                                                        fig_box = px.box(pivot_impact.reset_index(),
                                                                        x='Total Coaching Sessions', y='Improvement',
                                                                        title="Improvement Spread vs Coaching Volume",
                                                                        template="plotly_white")
                                                        st.plotly_chart(fig_box, use_container_width=True)

                                                    st.markdown("---")
                                                    st.markdown("### Expanded Impact Visualizations")
                                                    c_ev1, c_ev2 = st.columns(2)
                                                    with c_ev1:
                                                        if LOB_COL and LOB_COL in pivot_impact.columns:
                                                            fig_lob = px.box(pivot_impact.reset_index(), x=LOB_COL, y='Improvement',
                                                                            color=LOB_COL, title="1. Improvement by LOB",
                                                                            template="plotly_white")
                                                            fig_lob.update_layout(showlegend=False)
                                                            st.plotly_chart(fig_lob, use_container_width=True)
                                                        fig_eff = px.histogram(pivot_impact.reset_index(), x='Coaching Efficiency',
                                                                            nbins=20, title="3. Coaching Efficiency (pts/session)",
                                                                            template="plotly_white",
                                                                            color_discrete_sequence=['#5c32f3'])
                                                        st.plotly_chart(fig_eff, use_container_width=True)
                                                        med_b = pivot_impact['Avg Quality Before Coaching'].median()
                                                        med_i = pivot_impact['Improvement'].median()
                                                        fig_seg = px.scatter(pivot_impact.reset_index(),
                                                                            x='Avg Quality Before Coaching', y='Improvement',
                                                                            hover_name='Agent_ID', color='Total Coaching Sessions',
                                                                            title="5. Agent Segmentation (Quadrant)",
                                                                            template="plotly_white")
                                                        fig_seg.add_hline(y=med_i, line_dash="dash", line_color="red",
                                                                        annotation_text="Median Improvement")
                                                        fig_seg.add_vline(x=med_b, line_dash="dash", line_color="blue",
                                                                        annotation_text="Median Prior Score")
                                                        st.plotly_chart(fig_seg, use_container_width=True)
                                                    with c_ev2:
                                                        pi_s = pivot_impact.reset_index().sort_values('Improvement', ascending=False)
                                                        tb = pd.concat([pi_s.head(5), pi_s.tail(5)]).sort_values('Improvement', ascending=True)
                                                        fig_tb = px.bar(tb, x='Improvement', y='Agent_ID', orientation='h',
                                                                        title="2. Top & Bottom Responders", text_auto=".1f",
                                                                        color='Improvement', color_continuous_scale='RdYlGn',
                                                                        template="plotly_white")
                                                        st.plotly_chart(fig_tb, use_container_width=True)
                                                        melted = pd.melt(pivot_impact.reset_index(), id_vars=['Agent_ID'],
                                                                        value_vars=['Avg Quality Before Coaching', 'Avg Quality After Coaching'],
                                                                        var_name='Period', value_name='Score')
                                                        fig_vio = px.violin(melted, x='Period', y='Score', color='Period',
                                                                            box=True, points="all",
                                                                            title="4. Before vs After Distribution",
                                                                            template="plotly_white",
                                                                            color_discrete_map={"Avg Quality Before Coaching": "#ee3e32",
                                                                                                "Avg Quality After Coaching": "#32f384"})
                                                        fig_vio.update_yaxes(range=[0, 100])
                                                        fig_vio.update_layout(showlegend=False)
                                                        st.plotly_chart(fig_vio, use_container_width=True)
                                        except Exception as e:
                                            st.error(f"Agent-level breakdown error: {e}")
                                    else:
                                        st.info("Insufficient data to build Before vs After comparison.")
                                else:
                                    st.info("Missing date columns in datasets for impact analysis.")

                            # ══════════════════════════════════════════════════════
                            # TAB 3 – BEHAVIORS & LOCATIONS
                            # ══════════════════════════════════════════════════════
                            with tab_behav_geo:
                                sc1, sc2 = st.columns(2)
                                with sc1:
                                    BV_COL = c_cols.get('behavior')
                                    if BV_COL:
                                        df_c_with_score = pd.merge(df_c, df_q_agg, left_on=AGENT_COL, right_on=EMP_COL, how='inner')
                                        behavior_agg = df_c_with_score.groupby(BV_COL).agg(
                                            {'Avg_Quality_Score': 'mean', AGENT_COL: 'count'}).reset_index()
                                        behavior_agg.rename(columns={AGENT_COL: 'Count'}, inplace=True)
                                        fig_behav = px.bar(behavior_agg.sort_values('Count', ascending=False).head(10),
                                                        x=BV_COL, y='Avg_Quality_Score', color='Count',
                                                        title="Avg Quality Score by Coached Behavior",
                                                        template="plotly_white")
                                        st.plotly_chart(fig_behav, use_container_width=True)
                                with sc2:
                                    GEO_COL = q_cols.get('geo')
                                    if GEO_COL:
                                        agent_geo = df_q[[EMP_COL, GEO_COL]].drop_duplicates(subset=[EMP_COL])
                                        df_c_geo  = pd.merge(df_c, agent_geo, left_on=AGENT_COL, right_on=EMP_COL, how='inner')
                                        geo_agg   = df_c_geo.groupby(GEO_COL)[AGENT_COL].count().reset_index()
                                        geo_agg.rename(columns={AGENT_COL: 'Coaching Sessions'}, inplace=True)
                                        fig_geo = px.pie(geo_agg, names=GEO_COL, values='Coaching Sessions',
                                                        title="Coaching Distribution by Geo",
                                                        template="plotly_white", hole=0.4)
                                        st.plotly_chart(fig_geo, use_container_width=True)

                            # ══════════════════════════════════════════════════════
                            # TAB 4 – TREND OVER TIME
                            # ══════════════════════════════════════════════════════
                            with tab_trend:
                                DATECOACHED_COL = c_cols.get('datecoached')
                                DATE_COL        = q_cols.get('date')
                                if DATECOACHED_COL and DATE_COL:
                                    st.markdown("### Combined Trend Analysis")
                                    freq_label = st.selectbox("Select Aggregation Level:",
                                                            ["Daily (D)", "Weekly (W)", "Monthly (M)"], key='trend_cross')
                                    freq_map = {"Daily (D)": "D", "Weekly (W)": "W-MON", "Monthly (M)": "ME"}
                                    freq     = freq_map[freq_label]

                                    df_c_dates = df_c.copy()
                                    df_q_dates = df_q.copy()
                                    df_c_dates[DATECOACHED_COL] = pd.to_datetime(df_c_dates[DATECOACHED_COL], errors='coerce')
                                    df_q_dates[DATE_COL]        = pd.to_datetime(df_q_dates[DATE_COL], errors='coerce')
                                    df_c_dates = df_c_dates.dropna(subset=[DATECOACHED_COL])
                                    df_q_dates = df_q_dates.dropna(subset=[DATE_COL])
                                    df_q_dates = clamp_df(df_q_dates, SCORE_COL, 0, 100)

                                    c_trend = df_c_dates.groupby(pd.Grouper(key=DATECOACHED_COL, freq=freq))[AGENT_COL].count().reset_index()
                                    c_trend.rename(columns={DATECOACHED_COL: 'Date', AGENT_COL: 'Total Coaching Sessions'}, inplace=True)
                                    q_trend = df_q_dates.groupby(pd.Grouper(key=DATE_COL, freq=freq))[SCORE_COL].mean().reset_index()
                                    q_trend.rename(columns={DATE_COL: 'Date', SCORE_COL: 'Average Quality Score'}, inplace=True)

                                    GEO_COL = q_cols.get('geo')
                                    if GEO_COL:
                                        df_q_dates['India_Status'] = df_q_dates[GEO_COL].apply(
                                            lambda x: 'Inside India' if str(x).strip().upper() == 'IND' else 'Outside India')
                                        q_trend_geo = df_q_dates.groupby(
                                            [pd.Grouper(key=DATE_COL, freq=freq), 'India_Status'])[SCORE_COL].mean().reset_index()
                                    else:
                                        q_trend_geo = pd.DataFrame()

                                    trend_merged = pd.merge(c_trend, q_trend, on='Date', how='outer').sort_values('Date')
                                    trend_merged['Total Coaching Sessions'] = trend_merged['Total Coaching Sessions'].fillna(0)

                                    if not trend_merged.empty:
                                        fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
                                        fig_trend.add_trace(go.Bar(x=trend_merged['Date'],
                                                                y=trend_merged['Total Coaching Sessions'],
                                                                name="Coaching Sessions", marker_color='#aaaaaa', opacity=0.4),
                                                            secondary_y=False)
                                        if GEO_COL and not q_trend_geo.empty:
                                            ind = q_trend_geo[q_trend_geo['India_Status'] == 'Inside India']
                                            out = q_trend_geo[q_trend_geo['India_Status'] == 'Outside India']
                                            fig_trend.add_trace(go.Scatter(x=ind[DATE_COL], y=ind[SCORE_COL],
                                                                            name="Inside India", mode='lines+markers',
                                                                            line=dict(color='#ee3e32', width=3)), secondary_y=True)
                                            fig_trend.add_trace(go.Scatter(x=out[DATE_COL], y=out[SCORE_COL],
                                                                            name="Outside India", mode='lines+markers',
                                                                            line=dict(color='#32f384', width=3)), secondary_y=True)
                                            fig_trend.add_trace(go.Scatter(x=trend_merged['Date'],
                                                                            y=trend_merged['Average Quality Score'],
                                                                            name="Global Avg", mode='lines',
                                                                            line=dict(color='#333333', width=2, dash='dash')),
                                                                secondary_y=True)
                                        else:
                                            fig_trend.add_trace(go.Scatter(x=trend_merged['Date'],
                                                                            y=trend_merged['Average Quality Score'],
                                                                            name="Avg Quality Score",
                                                                            mode='lines+markers+text',
                                                                            text=trend_merged['Average Quality Score'].round(2),
                                                                            textposition='top center',
                                                                            line=dict(color='#32f384', width=3)), secondary_y=True)
                                        fig_trend.update_layout(
                                            title=f"Coaching Volume vs Quality Score ({freq_label})",
                                            template="plotly_white",
                                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                                        fig_trend.update_yaxes(range=[0, 100], secondary_y=True)
                                        if freq == 'D':
                                            fig_trend.update_xaxes(dtick="86400000", tickformat="%d %b", tickangle=-45)
                                        elif freq == 'W-MON':
                                            fig_trend.update_xaxes(dtick="604800000", tickformat="%d %b", tickangle=-45)
                                        elif freq == 'ME':
                                            fig_trend.update_xaxes(dtick="M1", tickformat="%b %Y", tickangle=-45)
                                        st.plotly_chart(fig_trend, use_container_width=True)
                                else:
                                    st.warning("Missing date columns required for trend analysis.")

                            # ══════════════════════════════════════════════════════
                            # TAB 5 – COACHING vs SCORE ANALYSIS
                            # ══════════════════════════════════════════════════════
                            with tab_coaching:
                                st.markdown("## 🎯 Coaching vs Score Analysis")
                                DATECOACHED_COL = c_cols.get('datecoached')
                                DATE_COL        = q_cols.get('date')

                                if DATECOACHED_COL and DATE_COL:
                                    available_metrics = df_q[METRIC_COL].dropna().unique().tolist() if METRIC_COL else ["QA SCORE"]
                                    chart1_metric = st.selectbox(
                                        "Choose quality metric to analyse:",
                                        available_metrics,
                                        index=available_metrics.index("QA SCORE") if "QA SCORE" in available_metrics else 0,
                                        key='chart1_metric_select')

                                    meta   = METRIC_META.get(chart1_metric, {"label": "Score", "range": [0, 100]})
                                    df_q_m = df_q[df_q[METRIC_COL] == chart1_metric].copy() if METRIC_COL else df_q.copy()
                                    df_q_m = clamp_df(df_q_m, SCORE_COL, meta['range'][0], meta['range'][1])

                                    df_q_m[DATE_COL]         = pd.to_datetime(df_q_m[DATE_COL], errors='coerce')
                                    df_c_cp                  = df_c.copy()
                                    df_c_cp[DATECOACHED_COL] = pd.to_datetime(df_c_cp[DATECOACHED_COL], errors='coerce')

                                    df_coach_count = df_c.groupby(AGENT_COL).size().reset_index(name='Coaching_Count')
                                    first_coaching = df_c_cp.groupby(AGENT_COL)[DATECOACHED_COL].min().reset_index()
                                    first_coaching.rename(columns={DATECOACHED_COL: 'First_Coaching_Date'}, inplace=True)

                                    df_q_after = pd.merge(df_q_m, first_coaching, left_on=EMP_COL, right_on=AGENT_COL, how='inner')
                                    df_q_after = df_q_after[df_q_after[DATE_COL] >= df_q_after['First_Coaching_Date']]
                                    df_after_avg = df_q_after.groupby(EMP_COL)[SCORE_COL].mean().reset_index()
                                    df_after_avg.rename(columns={SCORE_COL: 'Avg_Score_After'}, inplace=True)

                                    df_q_before = pd.merge(df_q_m, first_coaching, left_on=EMP_COL, right_on=AGENT_COL, how='inner')
                                    df_q_before = df_q_before[df_q_before[DATE_COL] < df_q_before['First_Coaching_Date']]
                                    df_before_avg = df_q_before.groupby(EMP_COL)[SCORE_COL].mean().reset_index()
                                    df_before_avg.rename(columns={SCORE_COL: 'Avg_Score_Before'}, inplace=True)

                                    df_final = df_coach_count.copy()
                                    df_final = pd.merge(df_final, df_after_avg, left_on=AGENT_COL, right_on=EMP_COL, how='inner')
                                    if EMP_COL in df_final.columns and EMP_COL != AGENT_COL:
                                        df_final.drop(columns=[EMP_COL], inplace=True)
                                    df_final = df_final[(df_final['Avg_Score_After'] <= meta['range'][1]) &
                                                        (df_final['Avg_Score_After'] >= meta['range'][0])]
                                    df_final = pd.merge(df_final, df_before_avg, left_on=AGENT_COL, right_on=EMP_COL, how='left')
                                    if EMP_COL in df_final.columns and EMP_COL != AGENT_COL:
                                        df_final.drop(columns=[EMP_COL], inplace=True)

                                    if not df_final.empty:
                                        ck1, ck2, ck3 = st.columns(3)
                                        ck1.metric("Agents Analyzed", f"{df_final.shape[0]}")
                                        ck2.metric("Avg Post-Coaching Score", f"{df_final['Avg_Score_After'].mean():.2f}")
                                        ck3.metric("Total Coaching Sessions", f"{df_coach_count['Coaching_Count'].sum()}")
                                        st.markdown("---")

                                        st.markdown(f"### 📊 Number of Coachings vs Avg {chart1_metric} After Coaching")
                                        st.caption(f"Metric: **{meta['label']}** | Y-axis: {meta['range'][0]}–{meta['range'][1]}")
                                        fig_c1 = px.scatter(df_final, x='Coaching_Count', y='Avg_Score_After',
                                                            size='Coaching_Count', hover_name=get_hover_col(df_final),
                                                            labels={'Coaching_Count': 'Number of Coachings',
                                                                    'Avg_Score_After': f'Avg {chart1_metric} (Post-Coaching)'},
                                                            title=f"Coaching Count vs Avg {chart1_metric} After Coaching",
                                                            trendline="ols", template="plotly_white", opacity=0.6)
                                        fig_c1.update_yaxes(range=meta['range'])
                                        st.plotly_chart(fig_c1, use_container_width=True)

                                        correlation = df_final[['Coaching_Count', 'Avg_Score_After']].corr().iloc[0, 1]
                                        cc1, cc2 = st.columns([1, 3])
                                        with cc1: st.metric(f"Correlation (Coaching vs {chart1_metric})", f"{correlation:.2f}")
                                        with cc2:
                                            if correlation > 0.3:   st.success("✅ More coaching → better performance 📈")
                                            elif correlation > 0:   st.info("ℹ️ Slight positive association.")
                                            else:                   st.warning("🚨 Coaching NOT improving performance — review quality.")
                                        st.markdown("---")

                                        st.markdown("### 📊 Distribution of Coaching Sessions per Agent")
                                        dist_df = df_coach_count.groupby('Coaching_Count').size().reset_index(name='Agent_Count')
                                        dist_df.rename(columns={'Coaching_Count': 'Number of Coachings', 'Agent_Count': 'Number of Agents'}, inplace=True)
                                        fig_c2 = px.bar(dist_df, x='Number of Coachings', y='Number of Agents',
                                                        text='Number of Agents', title="Distribution of Coaching Sessions per Agent",
                                                        template="plotly_white", color_discrete_sequence=['#5c32f3'])
                                        fig_c2.update_traces(textposition='outside', textfont_size=12)
                                        fig_c2.update_layout(xaxis=dict(dtick=1))
                                        st.plotly_chart(fig_c2, use_container_width=True)
                                        st.caption(f"Total Agents: {df_coach_count.shape[0]}  |  Total Sessions: {df_coach_count['Coaching_Count'].sum()}")
                                        st.markdown("---")

                                        st.markdown("### 📊 Avg Quality Score by Coaching Intensity")
                                        df_final['Coaching_Bucket'] = pd.cut(df_final['Coaching_Count'],
                                            bins=[0,2,5,10,50], labels=['Low (1-2)', 'Medium (3-5)', 'High (6-10)', 'Very High (11+)'])
                                        bucket_analysis = df_final.groupby('Coaching_Bucket', observed=True).agg(
                                            Avg_Score_After=('Avg_Score_After','mean'),
                                            Agent_Count=('Avg_Score_After','count')).reset_index()
                                        fig_c3 = px.bar(bucket_analysis, x='Coaching_Bucket', y='Avg_Score_After',
                                                        text=bucket_analysis['Avg_Score_After'].round(2),
                                                        color='Avg_Score_After', color_continuous_scale='RdYlGn',
                                                        labels={'Coaching_Bucket': 'Coaching Intensity',
                                                                'Avg_Score_After': f'Avg {chart1_metric} (Post-Coaching)'},
                                                        title="Avg Post-Coaching Score by Coaching Intensity",
                                                        template="plotly_white", hover_data={'Agent_Count': True})
                                        fig_c3.update_traces(textposition='outside', textfont_size=13)
                                        fig_c3.update_yaxes(range=meta['range'])
                                        fig_c3.update_layout(showlegend=False, coloraxis_showscale=False)
                                        st.plotly_chart(fig_c3, use_container_width=True)
                                        if not bucket_analysis.empty:
                                            best = bucket_analysis.sort_values('Avg_Score_After', ascending=False).iloc[0]
                                            st.info(f"🏆 Best in **{best['Coaching_Bucket']}** range — avg **{best['Avg_Score_After']:.2f}**.")
                                        st.markdown("---")

                                        st.markdown("### 🔬 Deep-Dive: Top 2 Most-Coached Agents")
                                        top2     = df_coach_count.sort_values('Coaching_Count', ascending=False).head(2)
                                        top2_ids = top2[AGENT_COL].tolist()
                                        if len(top2_ids) >= 2:
                                            st.info(f"Agents **{top2_ids[0]}** ({top2.iloc[0]['Coaching_Count']} sessions) and "
                                                    f"**{top2_ids[1]}** ({top2.iloc[1]['Coaching_Count']} sessions).")
                                            df_q_top2 = df_q_m[df_q_m[EMP_COL].isin(top2_ids)].copy()
                                            fig_top2  = make_subplots(rows=1, cols=2,
                                                                    subplot_titles=[f"Agent {top2_ids[0]}", f"Agent {top2_ids[1]}"])
                                            colors = ['#5c32f3', '#ee3e32']
                                            for idx, aid in enumerate(top2_ids):
                                                aq = df_q_top2[df_q_top2[EMP_COL] == aid].sort_values(DATE_COL)
                                                ac = df_c_cp[df_c_cp[AGENT_COL] == aid][DATECOACHED_COL].dropna()
                                                fig_top2.add_trace(go.Scatter(x=aq[DATE_COL], y=aq[SCORE_COL],
                                                                            mode='lines+markers', name=f"Agent {aid}",
                                                                            line=dict(color=colors[idx], width=2)),
                                                                row=1, col=idx+1)
                                                for cd in ac:
                                                    fig_top2.add_vline(x=cd, line_dash="dot", line_color="grey",
                                                                    line_width=1, row=1, col=idx+1)
                                                n_c = top2[top2[AGENT_COL] == aid]['Coaching_Count'].values[0]
                                                fig_top2.add_annotation(text=f"Sessions: {n_c}",
                                                    xref=f"x{'' if idx==0 else idx+1}", yref="paper",
                                                    x=aq[DATE_COL].mean() if not aq.empty else 0,
                                                    y=1.05, showarrow=False, font=dict(size=11, color=colors[idx]))
                                            fig_top2.update_layout(title="Quality Score Over Time — Top 2 Most-Coached<br>"
                                                                "<sup>Dashed lines = coaching sessions</sup>",
                                                                template="plotly_white", height=420, showlegend=False)
                                            fig_top2.update_yaxes(range=meta['range'])
                                            st.plotly_chart(fig_top2, use_container_width=True)
                                            t2 = df_final[df_final[AGENT_COL].isin(top2_ids)][
                                                [AGENT_COL, 'Coaching_Count', 'Avg_Score_Before', 'Avg_Score_After']].copy()
                                            t2['Improvement'] = t2['Avg_Score_After'] - t2['Avg_Score_Before']
                                            st.dataframe(t2.rename(columns={AGENT_COL: 'Agent ID', 'Coaching_Count': 'Total Sessions',
                                                                            'Avg_Score_Before': 'Avg Before',
                                                                            'Avg_Score_After': 'Avg After'}).set_index('Agent ID'),
                                                        use_container_width=True)
                                        st.markdown("---")

                                        st.markdown("### 📊 Top 15 Most-Coached — Before vs After")
                                        df_top15 = df_final.sort_values('Coaching_Count', ascending=False).head(15).copy()
                                        df_top15 = df_top15.sort_values('Avg_Score_After', ascending=True)
                                        fig_c4 = go.Figure()
                                        fig_c4.add_trace(go.Bar(y=df_top15[AGENT_COL].astype(str), x=df_top15['Avg_Score_Before'],
                                                                name='Before', orientation='h', marker_color='rgba(238,62,50,0.25)',
                                                                text=df_top15['Avg_Score_Before'].round(2), textposition='outside'))
                                        fig_c4.add_trace(go.Bar(y=df_top15[AGENT_COL].astype(str), x=df_top15['Avg_Score_After'],
                                                                name='After', orientation='h', marker_color='rgba(50,243,132,0.5)',
                                                                text=df_top15['Avg_Score_After'].round(2), textposition='outside'))
                                        fig_c4.update_layout(barmode='group', title="Top 15 Most-Coached: Before vs After",
                                                            xaxis=dict(range=meta['range']), template="plotly_white",
                                                            legend=dict(orientation="h", y=1.02, x=1, xanchor='right'))
                                        st.plotly_chart(fig_c4, use_container_width=True)
                                        st.markdown("---")

                                        st.markdown("### 📋 Agent-Level Table")
                                        df_display = df_final[[AGENT_COL, 'Coaching_Count', 'Avg_Score_Before',
                                                            'Avg_Score_After', 'Coaching_Bucket']].copy()
                                        df_display['Improvement'] = df_display['Avg_Score_After'] - df_display['Avg_Score_Before']
                                        df_display.rename(columns={AGENT_COL: 'Agent ID', 'Coaching_Count': 'Total Sessions',
                                                                'Avg_Score_Before': 'Avg Before', 'Avg_Score_After': 'Avg After',
                                                                'Coaching_Bucket': 'Intensity'}, inplace=True)
                                        st.dataframe(df_display.sort_values('Total Sessions', ascending=False)
                                                    .style.background_gradient(subset=['Avg After'], cmap='Greens'),
                                                    use_container_width=True)
                                    else:
                                        st.warning("No data after merging coaching and quality datasets.")
                                else:
                                    st.warning("Required date columns not found.")

                            # ══════════════════════════════════════════════════════
                            # TAB 6 – PERFORMANCE DATA ANALYSIS
                            # ══════════════════════════════════════════════════════
                            with tab_performance:
                                st.markdown("## 🚀 Performance Data Analysis")

                                if not os.path.exists(PERF_PATH):
                                    st.error(f"Performance file not found: `{PERF_PATH}`.")
                                    st.stop()

                                @st.cache_data
                                def _load_perf_csv(path):
                                    _df = pd.read_csv(path)
                                    _df.columns = _df.columns.str.strip()
                                    _df['Date'] = pd.to_datetime(_df['Date'], errors='coerce')
                                    return _df

                                df_perf = _load_perf_csv(PERF_PATH)

                                perf_num_cols = ['Calls_Answered', 'AHT', 'Avg_Talk_Time', 'Avg_Hold_Time',
                                                'Avg_Wrap_Time', 'Occupancy %', 'Productive_Hours', 'Overall_Shrinkage']
                                perf_agg_cols = {c: 'mean' for c in perf_num_cols if c in df_perf.columns}
                                if 'Calls_Answered' in df_perf.columns:
                                    perf_agg_cols['Calls_Answered'] = 'sum'

                                df_perf_agg = df_perf.groupby('Agent_ID').agg(perf_agg_cols).reset_index()
                                df_perf_agg.rename(columns={'Calls_Answered': 'Total_Calls'}, inplace=True)

                                df_coach_all   = df_c.groupby(AGENT_COL).size().reset_index(name='Coaching_Sessions')
                                df_merged_perf = pd.merge(df_perf_agg, df_coach_all,
                                                        left_on='Agent_ID', right_on=AGENT_COL, how='inner')
                                if AGENT_COL != 'Agent_ID' and AGENT_COL in df_merged_perf.columns:
                                    df_merged_perf.drop(columns=[AGENT_COL], inplace=True)

                                df_q_qa = df_q[df_q[METRIC_COL] == 'QA SCORE'].copy() if METRIC_COL else df_q.copy()
                                df_q_qa = clamp_df(df_q_qa, SCORE_COL, 0, 100)
                                df_q_qa_agg = df_q_qa.groupby(EMP_COL)[SCORE_COL].mean().reset_index()
                                df_q_qa_agg.rename(columns={SCORE_COL: 'Avg_QA_Score', EMP_COL: 'Agent_ID'}, inplace=True)
                                df_merged_perf = pd.merge(df_merged_perf, df_q_qa_agg, on='Agent_ID', how='left')

                                view_mode = st.radio("📊 View Mode:",
                                                    ["Raw Values", "Z-Score Normalised"],
                                                    horizontal=True, key='perf_view_mode')

                                num_perf_cols = [c for c in ['AHT', 'Avg_Talk_Time', 'Avg_Hold_Time', 'Avg_Wrap_Time',
                                                            'Occupancy %', 'Productive_Hours', 'Overall_Shrinkage',
                                                            'Total_Calls', 'Avg_QA_Score', 'Coaching_Sessions']
                                                if c in df_merged_perf.columns]

                                if view_mode == "Z-Score Normalised":
                                    df_display_perf = df_merged_perf.copy()
                                    for col in num_perf_cols:
                                        mu  = df_display_perf[col].mean()
                                        sig = df_display_perf[col].std()
                                        df_display_perf[col] = (df_display_perf[col] - mu) / sig if sig > 0 else 0
                                    st.info("📐 Values shown as z-scores (mean=0, std=1). Positive = above average, Negative = below average.")
                                else:
                                    df_display_perf = df_merged_perf.copy()

                                st.markdown(f"**{len(df_merged_perf):,} agents** found across coaching + performance datasets.")
                                p1, p2, p3 = st.columns(3)
                                p1.metric("Merged Agents", f"{len(df_merged_perf):,}")
                                if 'Total_Calls' in df_merged_perf.columns:
                                    p2.metric("Avg Calls / Agent", f"{df_merged_perf['Total_Calls'].mean():,.0f}")
                                if 'AHT' in df_merged_perf.columns:
                                    p3.metric("Avg AHT (sec)", f"{df_merged_perf['AHT'].mean():,.1f}")
                                # FIX: removed Avg Coaching Sessions KPI (coaching ID is not numeric)
                                st.markdown("---")

                                if 'AHT' in df_display_perf.columns:
                                    st.markdown("### 📞 Coaching Sessions vs AHT")
                                    st.caption("AHT in seconds. Lower = better." if view_mode == "Raw Values" else "Z-score normalised values.")
                                    fig_p1 = px.scatter(df_display_perf, x='Coaching_Sessions', y='AHT',
                                                        trendline='ols', template='plotly_white',
                                                        hover_name=get_hover_col(df_display_perf),
                                                        title=f"Coaching Sessions vs AHT ({view_mode})",
                                                        labels={'Coaching_Sessions': 'Coaching Sessions', 'AHT': 'AHT'},
                                                        opacity=0.6, color_discrete_sequence=['#5c32f3'])
                                    st.plotly_chart(fig_p1, use_container_width=True)
                                st.markdown("---")

                                if 'Occupancy %' in df_display_perf.columns:
                                    st.markdown("### ⏱️ Coaching Sessions vs Occupancy %")
                                    st.caption("Higher occupancy = more time on calls. Ideally 75–85%." if view_mode == "Raw Values"
                                            else "Z-score normalised values.")
                                    fig_p2 = px.scatter(df_display_perf, x='Coaching_Sessions', y='Occupancy %',
                                                        trendline='ols', template='plotly_white',
                                                        hover_name=get_hover_col(df_display_perf),
                                                        title=f"Coaching Sessions vs Occupancy % ({view_mode})",
                                                        labels={'Coaching_Sessions': 'Coaching Sessions'},
                                                        opacity=0.6, color_discrete_sequence=['#32c4f3'])
                                    st.plotly_chart(fig_p2, use_container_width=True)
                                st.markdown("---")

                                st.markdown("### 📊 Performance by Coaching Intensity")
                                # FIX: use raw Coaching_Sessions from df_merged_perf for bucketing, not z-scored
                                df_display_perf['Coaching_Bucket'] = pd.cut(
                                    df_merged_perf['Coaching_Sessions'], bins=[0,2,5,10,100],
                                    labels=['Low (1-2)', 'Medium (3-5)', 'High (6-10)', 'Very High (11+)'])

                                perf_metric_select = st.selectbox(
                                    "Select performance metric:",
                                    [c for c in ['AHT','Avg_Talk_Time','Avg_Hold_Time','Avg_Wrap_Time',
                                                'Occupancy %','Productive_Hours','Overall_Shrinkage','Total_Calls']
                                    if c in df_display_perf.columns],
                                    key='perf_metric_select')

                                bucket_perf = df_display_perf.groupby('Coaching_Bucket', observed=True).agg(
                                    Avg_Metric=(perf_metric_select,'mean'),
                                    Agent_Count=(perf_metric_select,'count')).reset_index()
                                # FIX: avoid white color in bar chart — use Blues_r so values are visible
                                fig_p3 = px.bar(bucket_perf, x='Coaching_Bucket', y='Avg_Metric',
                                                text=bucket_perf['Avg_Metric'].round(2),
                                                color='Avg_Metric', color_continuous_scale='Teal',
                                                hover_data={'Agent_Count': True},
                                                title=f"Avg {perf_metric_select} by Coaching Intensity ({view_mode})",
                                                template='plotly_white')
                                fig_p3.update_traces(textposition='outside')
                                fig_p3.update_layout(showlegend=False, coloraxis_showscale=False)
                                st.plotly_chart(fig_p3, use_container_width=True)
                                st.markdown("---")

                                if 'Avg_QA_Score' in df_display_perf.columns and 'AHT' in df_display_perf.columns:
                                    st.markdown("### 🔗 QA Score vs AHT — Quality vs Efficiency")
                                    df_qa_aht = df_display_perf.dropna(subset=['Avg_QA_Score', 'AHT']).copy()
                                    # FIX: size must be non-negative; clip at 0 for z-score mode
                                    size_col_raw = df_merged_perf.loc[df_qa_aht.index, 'Coaching_Sessions'].clip(lower=0)
                                    fig_p4 = px.scatter(df_qa_aht, x='Avg_QA_Score', y='AHT',
                                                        color='Coaching_Sessions', color_continuous_scale='Viridis',
                                                        size=size_col_raw.values,
                                                        hover_name=get_hover_col(df_qa_aht),
                                                        trendline='ols', template='plotly_white',
                                                        title=f"QA Score vs AHT ({view_mode})",
                                                        labels={'Avg_QA_Score': 'Avg QA Score', 'AHT': 'AHT',
                                                                'Coaching_Sessions': 'Sessions'},
                                                        opacity=0.7)
                                    if view_mode == "Raw Values":
                                        fig_p4.update_xaxes(range=[0, 100])
                                    st.plotly_chart(fig_p4, use_container_width=True)
                                st.markdown("---")

                                # Quadrant Analysis
                                st.markdown("### 🔲 Quadrant Analysis — Performance vs Coaching")
                                quad_x = st.selectbox("X-axis metric:", num_perf_cols, key='quad_x',
                                                    index=num_perf_cols.index('Coaching_Sessions') if 'Coaching_Sessions' in num_perf_cols else 0)
                                quad_y = st.selectbox("Y-axis metric:", num_perf_cols, key='quad_y',
                                                    index=num_perf_cols.index('AHT') if 'AHT' in num_perf_cols else 1)

                                hover_c = get_hover_col(df_display_perf)
                                # Build unique column list — quad_x/quad_y may already be Coaching_Sessions
                                quad_base_cols = list(dict.fromkeys(
                                    [c for c in [hover_c, quad_x, quad_y, 'Coaching_Sessions'] if c is not None]
                                ))
                                df_quad = df_display_perf[quad_base_cols].dropna().copy()

                                # Align raw Coaching_Sessions for size (always non-negative)
                                # Use raw df_merged_perf values mapped to df_quad's index
                                raw_sessions_quad = df_merged_perf.loc[df_quad.index, 'Coaching_Sessions'].clip(lower=0)

                                med_x = float(df_quad[quad_x].median())
                                med_y = float(df_quad[quad_y].median())

                                # Use vectorised numpy comparison — avoids all Series truth-value issues
                                x_vals = df_quad[quad_x].to_numpy(dtype=float)
                                y_vals = df_quad[quad_y].to_numpy(dtype=float)
                                conditions = [
                                    (x_vals >= med_x) & (y_vals >= med_y),
                                    (x_vals >= med_x) & (y_vals <  med_y),
                                    (x_vals <  med_x) & (y_vals >= med_y),
                                    (x_vals <  med_x) & (y_vals <  med_y),
                                ]
                                choices = ["High X / High Y", "High X / Low Y", "Low X / High Y", "Low X / Low Y"]
                                df_quad['Quadrant'] = np.select(conditions, choices, default="Low X / Low Y")

                                quad_colors = {
                                    "High X / High Y": "#ee3e32",
                                    "High X / Low Y":  "#32f384",
                                    "Low X / High Y":  "#5c32f3",
                                    "Low X / Low Y":   "#f3c532"
                                }

                                fig_quad = px.scatter(df_quad, x=quad_x, y=quad_y, color='Quadrant',
                                                    hover_name=hover_c,
                                                    color_discrete_map=quad_colors,
                                                    title=f"Quadrant Analysis: {quad_x} vs {quad_y} ({view_mode})",
                                                    template="plotly_white", opacity=0.75,
                                                    size=raw_sessions_quad.values)
                                fig_quad.add_hline(y=med_y, line_dash="dash", line_color="grey",
                                                annotation_text=f"Median {quad_y}: {med_y:.2f}",
                                                annotation_position="bottom right")
                                fig_quad.add_vline(x=med_x, line_dash="dash", line_color="grey",
                                                annotation_text=f"Median {quad_x}: {med_x:.2f}",
                                                annotation_position="top left")
                                st.plotly_chart(fig_quad, use_container_width=True)

                                if hover_c:
                                    quad_summary = df_quad.groupby('Quadrant').agg(
                                        Agent_Count=(hover_c, 'count'),
                                    ).reset_index()
                                    quad_summary[f'Avg_{quad_x}'] = df_quad.groupby('Quadrant')[quad_x].mean().values
                                    quad_summary[f'Avg_{quad_y}'] = df_quad.groupby('Quadrant')[quad_y].mean().values
                                    quad_summary['Avg_Coaching_Sessions'] = (
                                        df_quad.assign(_raw=raw_sessions_quad.values)
                                        .groupby('Quadrant')['_raw'].mean().values
                                    )
                                    st.dataframe(quad_summary.style.background_gradient(subset=['Agent_Count'], cmap='Blues'),
                                                use_container_width=True)
                                st.markdown("---")

                                st.markdown("### 🧮 Correlation Heatmap — Performance vs Coaching")
                                heat_cols = [c for c in ['Coaching_Sessions','Avg_QA_Score','Total_Calls',
                                                        'AHT','Avg_Talk_Time','Avg_Hold_Time','Avg_Wrap_Time',
                                                        'Occupancy %','Productive_Hours','Overall_Shrinkage']
                                            if c in df_display_perf.columns]
                                if len(heat_cols) > 2:
                                    df_heat = df_display_perf[heat_cols].dropna().copy()
                                    if 'Avg_QA_Score' in df_heat.columns and view_mode == "Raw Values":
                                        df_heat = df_heat[(df_heat['Avg_QA_Score'] >= 0) & (df_heat['Avg_QA_Score'] <= 100)]
                                    corr_df = df_heat.corr()
                                    fig_heat = px.imshow(corr_df, text_auto=".2f", aspect="auto",
                                                        title=f"Pearson Correlation — Performance Metrics ({view_mode})",
                                                        color_continuous_scale='RdBu_r', template='plotly_white')
                                    st.plotly_chart(fig_heat, use_container_width=True)
                                st.markdown("---")

                                st.markdown("### 📋 Agent-Level Performance + Coaching Summary")
                                display_cols = ['Agent_ID','Coaching_Sessions','Avg_QA_Score','Total_Calls',
                                                'AHT','Occupancy %','Productive_Hours','Overall_Shrinkage']
                                display_cols = [c for c in display_cols if c in df_display_perf.columns]
                                st.dataframe(df_display_perf[display_cols].sort_values('Coaching_Sessions', ascending=False)
                                            .style.background_gradient(subset=['Coaching_Sessions'], cmap='Blues'),
                                            use_container_width=True)

                            # ══════════════════════════════════════════════════════
                            # TAB 7 – DOWNLOAD REPORT  (3 separate JSONs)
                            # ══════════════════════════════════════════════════════
                            with tab_report:
                                st.markdown("## ⬇️ Download Chart-Data Reports")
                                st.markdown(
                                    "Three separate JSON downloads are available — **Quality**, **Coaching**, and "
                                    "**Quality × Coaching cross-analysis**. Performance dataset is excluded. "
                                    "Each JSON covers every combination of metric, geo, and LOB filters."
                                )

                                df_q_raw_rep = load_and_prep_data(FILE_PATH, sheet_name='Quality Data ')
                                df_c_raw_rep = load_and_prep_data(FILE_PATH, sheet_name='Coaching Data')

                                q_cols_rep = {c.lower(): c for c in df_q_raw_rep.columns}
                                c_cols_rep = {c.lower(): c for c in df_c_raw_rep.columns}
                                SC = q_cols_rep.get('score')
                                EC = q_cols_rep.get('employee_id')
                                AC = c_cols_rep.get('agentid')
                                MC = q_cols_rep.get('metric_name')

                                if not (SC and EC and AC):
                                    st.error("Required columns (score / employee_id / agentid) not found.")
                                else:
                                    col_dl1, col_dl2, col_dl3 = st.columns(3)

                                    with col_dl1:
                                        st.markdown("### 📊 Quality Data")
                                        st.caption("Agent scores, distributions, trends — per metric/geo/LOB combo.")
                                        if st.button("🔄 Generate Quality JSON", key='gen_quality', type="primary"):
                                            with st.spinner("Building Quality report…"):
                                                rq = build_quality_json(df_q_raw_rep, SC, EC, MC)
                                            js = json.dumps(rq, default=safe_json, indent=2, ensure_ascii=False)
                                            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                                            st.success(f"✅ {len(rq['sections'])} combos, "
                                                    f"{sum(len(s['charts']) for s in rq['sections'])} charts")
                                            st.download_button("⬇️ Download Quality JSON", data=js.encode("utf-8"),
                                                            file_name=f"quality_report_{ts}.json",
                                                            mime="application/json", key='dl_quality')

                                    with col_dl2:
                                        st.markdown("### 🎓 Coaching Data")
                                        st.caption("Session distributions, behavior breakdowns, volume trends.")
                                        if st.button("🔄 Generate Coaching JSON", key='gen_coaching', type="primary"):
                                            with st.spinner("Building Coaching report…"):
                                                rc = build_coaching_json(df_c_raw_rep, AC)
                                            js = json.dumps(rc, default=safe_json, indent=2, ensure_ascii=False)
                                            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                                            st.success(f"✅ {sum(len(s['charts']) for s in rc['sections'])} chart datasets")
                                            st.download_button("⬇️ Download Coaching JSON", data=js.encode("utf-8"),
                                                            file_name=f"coaching_report_{ts}.json",
                                                            mime="application/json", key='dl_coaching')

                                    with col_dl3:
                                        st.markdown("### 🤝 Quality × Coaching")
                                        st.caption("Before/after impact, intensity buckets, correlations — all combos.")
                                        if st.button("🔄 Generate Cross-Analysis JSON", key='gen_cross', type="primary"):
                                            with st.spinner("Building cross-analysis report…"):
                                                rx = build_cross_json(df_q_raw_rep, df_c_raw_rep, SC, EC, AC, MC)
                                            js = json.dumps(rx, default=safe_json, indent=2, ensure_ascii=False)
                                            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                                            st.success(f"✅ {len(rx['sections'])} combos, "
                                                    f"{sum(len(s['charts']) for s in rx['sections'])} charts")
                                            st.download_button("⬇️ Download Cross-Analysis JSON", data=js.encode("utf-8"),
                                                            file_name=f"cross_report_{ts}.json",
                                                            mime="application/json", key='dl_cross')

                                    st.markdown("---")
                                    st.markdown("### 👁️ JSON Structure Reference")
                                    st.json({
                                        "generated_at": "<ISO timestamp>",
                                        "description": "<report description>",
                                        "sections": [{
                                            "filter_combination": {"metric": "...", "geo": "...", "lob": "..."},
                                            "charts": [{
                                                "chart_title": "<chart name>",
                                                "description": "<what the data represents>",
                                                "filter_combo": "<metric=... | geo=... | lob=...>",
                                                "data": [{"<col>": "<value>"}],
                                                "annotations": {"<key>": "<value>"}
                                            }]
                                        }]
                                    })

                        else:
                            st.warning("No overlapping agents found between datasets.")
                    else:
                        st.error("Required columns (score / employee_id / agentid) not found in datasets.")

                # ══════════════════════════════════════════════════════════════════
                # INDIVIDUAL SHEET — Quality Data
                # ══════════════════════════════════════════════════════════════════
                elif selected_sheet == 'Quality Data ' or (selected_sheet != cross_analysis_label
                                                            and selected_sheet in sheet_names
                                                            and 'quality' in selected_sheet.lower()):
                    df = load_and_prep_data(FILE_PATH, sheet_name=selected_sheet)

                    st.sidebar.markdown("### 🎛️ Dynamic Filters")
                    cat_cols = [c for c in df.columns if df[c].dtype == 'object'
                                and df[c].nunique() < 20 and df[c].nunique() > 0
                                and c.lower() not in ('employee_id',)]
                    for col in cat_cols:
                        unique_vals = sorted([str(x) for x in df[col].dropna().unique()])
                        if unique_vals:
                            selected_vals = st.sidebar.multiselect(f"Filter {col}", unique_vals, default=unique_vals)
                            if selected_vals:
                                df = df[df[col].astype(str).isin(selected_vals)]

                    q_col_map  = {c.lower(): c for c in df.columns}
                    SCORE_COL  = q_col_map.get('score')
                    EMP_COL    = q_col_map.get('employee_id')
                    METRIC_COL = q_col_map.get('metric_name')

                    # Tabs including Download Report
                    tab_ov, tab_uni, tab_bi, tab_ts, tab_dl = st.tabs([
                        "🔍 Overview & Quality", "📊 Univariate Analysis",
                        "🔗 Bivariate Analysis", "📈 Time Series", "⬇️ Download Report"])

                    with tab_ov:
                        st.markdown("### Raw Data Preview")
                        st.dataframe(df.head(100), use_container_width=True)
                        st.markdown("---")
                        col1, col2 = st.columns([1, 2])
                        total_rows   = df.shape[0]
                        total_cols   = df.shape[1]
                        missing_cells = df.isna().sum().sum()
                        total_cells  = total_rows * total_cols
                        completeness = ((total_cells - missing_cells) / total_cells) * 100 if total_cells > 0 else 0
                        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                        kpi1.metric("Total Rows", f"{total_rows:,}")
                        kpi2.metric("Total Columns", f"{total_cols:,}")
                        kpi3.metric("Completeness", f"{completeness:.1f}%")
                        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
                        if SCORE_COL and SCORE_COL in numeric_cols:
                            kpi4.metric(f"Avg {SCORE_COL}", f"{df[SCORE_COL].mean():.2f}")
                        elif numeric_cols:
                            kpi4.metric(f"Avg {numeric_cols[0]}", f"{df[numeric_cols[0]].mean():.2f}")
                        else:
                            kpi4.metric("Numeric Cols", "0")
                        with col1:
                            st.markdown("### Data Types & Missing Values")
                            info_df = pd.DataFrame({'Data Type': df.dtypes.astype(str),
                                                    'Missing (Count)': df.isna().sum(),
                                                    'Missing (%)': (df.isna().sum() / total_rows * 100).round(2)})
                            st.dataframe(info_df.style.background_gradient(subset=['Missing (%)'], cmap='Reds'), use_container_width=True)
                        with col2:
                            st.markdown("### Summary Statistics (Numeric)")
                            if numeric_cols:
                                st.dataframe(df.describe().T, use_container_width=True)
                            else:
                                st.info("No numerical columns found.")

                    with tab_uni:
                        st.markdown("### Distribution Analysis")
                        col_to_analyze = st.selectbox("Select a feature:", df.columns, key="q_univariate_select")
                        if pd.api.types.is_numeric_dtype(df[col_to_analyze]):
                            fig = px.histogram(df, x=col_to_analyze, marginal="box",
                                            title=f"Distribution of {col_to_analyze}", text_auto=True,
                                            template="plotly_white", color_discrete_sequence=['#32f384'])
                            st.plotly_chart(fig, use_container_width=True)
                        elif pd.api.types.is_datetime64_any_dtype(df[col_to_analyze]):
                            dc = df[col_to_analyze].dt.date.value_counts().reset_index()
                            dc.columns = [col_to_analyze, 'Record Count']
                            dc = dc.sort_values(col_to_analyze)
                            fig = px.line(dc, x=col_to_analyze, y='Record Count', text='Record Count',
                                        title=f"Record Frequency over {col_to_analyze}", template="plotly_white", markers=True)
                            fig.update_traces(line_color='#2ca02c', textposition='top center')
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            vc = df[col_to_analyze].value_counts().reset_index()
                            vc.columns = [col_to_analyze, 'Count']
                            c1, c2 = st.columns(2)
                            with c1:
                                fb = px.bar(vc.head(20), x=col_to_analyze, y='Count',
                                            title=f"Top 20 in {col_to_analyze}", template="plotly_white", text_auto='.2s')
                                fb.update_xaxes(categoryorder='total descending')
                                fb.update_traces(marker_color='#5c32f3')
                                st.plotly_chart(fb, use_container_width=True)
                            with c2:
                                fp = px.pie(vc.head(10), names=col_to_analyze, values='Count',
                                            title=f"Proportion (Top 10) in {col_to_analyze}",
                                            template="plotly_white", hole=0.4)
                                fp.update_traces(textposition='inside', textinfo='percent+label+value')
                                st.plotly_chart(fp, use_container_width=True)

                    with tab_bi:
                        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
                        st.markdown("### Feature Relationships")
                        c1, c2 = st.columns(2)
                        with c1: x_axis = st.selectbox("X-Axis:", df.columns, index=0, key='q_bi_x')
                        with c2:
                            yi = df.columns.get_loc(numeric_cols[0]) if numeric_cols else 0
                            y_axis = st.selectbox("Y-Axis:", df.columns, index=yi, key='q_bi_y')
                        if pd.api.types.is_numeric_dtype(df[y_axis]) and not pd.api.types.is_numeric_dtype(df[x_axis]):
                            fig = px.box(df, x=x_axis, y=y_axis, color=x_axis, title=f"{y_axis} by {x_axis}", template="plotly_white")
                            if df[x_axis].nunique() > 15: fig.update_layout(showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
                        elif not pd.api.types.is_numeric_dtype(df[x_axis]) and not pd.api.types.is_numeric_dtype(df[y_axis]):
                            fig = px.histogram(df, x=x_axis, color=y_axis, barmode='group',
                                            title=f"{x_axis} by {y_axis}", text_auto=True, template="plotly_white")
                            if df[x_axis].nunique() > 15: fig.update_layout(showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
                        elif pd.api.types.is_numeric_dtype(df[x_axis]) and pd.api.types.is_numeric_dtype(df[y_axis]):
                            lc = next((c for c in df.columns if 'id' in c.lower() or 'name' in c.lower()
                                    or 'agent' in c.lower() or 'employee' in c.lower()), None)
                            fig = px.scatter(df, x=x_axis, y=y_axis, opacity=0.6, text=lc,
                                            title=f"Scatter: {x_axis} vs {y_axis}", template="plotly_white",
                                            color_discrete_sequence=['#ee3e32'])
                            if lc: fig.update_traces(textposition='top center')
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("💡 Select one categorical (X) and one numeric (Y) feature.")
                        st.markdown("---")
                        if len(numeric_cols) > 1:
                            st.markdown("### Correlation Heatmap")
                            corr = df[numeric_cols].dropna().corr()
                            fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto",
                                                title="Pearson Correlation Between Numeric Features",
                                                color_continuous_scale="RdBu_r", template="plotly_white")
                            st.plotly_chart(fig_corr, use_container_width=True)

                    with tab_ts:
                        date_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
                        if not date_cols:
                            for c in df.columns:
                                if 'date' in c.lower() or 'time' in c.lower() or 'timestamp' in c.lower():
                                    date_cols.append(c)
                        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
                        if date_cols and numeric_cols:
                            st.markdown("### Time Series Analysis")
                            c1, c2, c3, c4 = st.columns(4)
                            with c1: ts_dc = st.selectbox("Date Column:", date_cols, key='q_ts_dc')
                            with c2: ts_vc = st.selectbox("Numeric Metric:", numeric_cols, key='q_ts_vc')
                            with c3: agg_f = st.selectbox("Aggregation:", ["mean","sum","count","min","max"], key='q_ts_agg')
                            with c4: ts_cc = st.selectbox("Color by:", ["None"] + [c for c in df.columns if c not in [ts_dc, ts_vc]], key='q_ts_cc')
                            if st.button("Generate Time Series Chart", key='q_ts_btn'):
                                try:
                                    df_ts = df.copy()
                                    df_ts[ts_dc] = pd.to_datetime(df_ts[ts_dc])
                                    df_ts['Date_Trunc'] = df_ts[ts_dc].dt.floor('D')
                                    gc = ['Date_Trunc'] + ([ts_cc] if ts_cc != "None" else [])
                                    tsg = df_ts.groupby(gc)[ts_vc].agg(agg_f).reset_index()
                                    tsg.rename(columns={ts_vc: 'Metric_Value'}, inplace=True)
                                    fk = {"data_frame": tsg, "x": 'Date_Trunc', "y": 'Metric_Value', "markers": True,
                                        "text": "Metric_Value", "title": f"Time Series: {agg_f.capitalize()} of {ts_vc}",
                                        "labels": {'Date_Trunc': 'Date', 'Metric_Value': f'{agg_f.capitalize()} {ts_vc}'},
                                        "template": "plotly_white"}
                                    if ts_cc != "None": fk["color"] = ts_cc
                                    fig_ts = px.line(**fk)
                                    fig_ts.update_traces(textposition='top center')
                                    if ts_cc == "None": fig_ts.update_traces(line_color='#00cc66', line_width=3)
                                    st.plotly_chart(fig_ts, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Could not generate time series. Error: {e}")
                        else:
                            st.info("We couldn't detect both Date/Time AND Numeric columns for time series analysis.")

                    # ── Quality Download Report tab ───────────────────────────────
                    with tab_dl:
                        st.markdown("## ⬇️ Download Quality Data Report")
                        st.markdown("Generates a JSON with chart data for every combination of metric, geo, and LOB filters.")
                        if st.button("🔄 Generate Quality JSON", key='gen_quality_sheet', type="primary"):
                            df_q_raw_dl = load_and_prep_data(FILE_PATH, sheet_name=selected_sheet)
                            report, err = build_quality_sheet_json(df_q_raw_dl)
                            if err:
                                st.error(err)
                            else:
                                js = json.dumps(report, default=safe_json, indent=2, ensure_ascii=False)
                                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                                n_charts = sum(len(s['charts']) for s in report['sections'])
                                st.success(f"✅ {len(report['sections'])} filter combinations, {n_charts} chart datasets generated.")
                                st.download_button("⬇️ Download Quality JSON", data=js.encode("utf-8"),
                                                file_name=f"quality_report_{ts}.json",
                                                mime="application/json", key='dl_quality_sheet')
                        st.markdown("---")
                        st.markdown("### 👁️ JSON Structure Reference")
                        st.json({
                            "generated_at": "<ISO timestamp>",
                            "description": "Quality data export",
                            "sections": [{
                                "filter_combination": {"metric": "...", "geo": "...", "lob": "..."},
                                "charts": [{"chart_title": "...", "description": "...", "data": [{"col": "value"}]}]
                            }]
                        })

                # ══════════════════════════════════════════════════════════════════
                # INDIVIDUAL SHEET — Coaching Data
                # ══════════════════════════════════════════════════════════════════
                elif selected_sheet == 'Coaching Data' or (selected_sheet != cross_analysis_label
                                                            and selected_sheet in sheet_names
                                                            and 'coaching' in selected_sheet.lower()):
                    df = load_and_prep_data(FILE_PATH, sheet_name=selected_sheet)
                    # Drop unnamed/near-empty columns (artefacts after SEVENDAYEFFECTIVENESS)
                    df = clean_coaching_df(df)

                    st.sidebar.markdown("### 🎛️ Dynamic Filters")
                    # Only show meaningful categorical columns — exclude IDs and unnamed
                    cat_cols = usable_cat_cols_for_filter(df, exclude_id_like=True)
                    for col in cat_cols:
                        unique_vals = sorted([str(x) for x in df[col].dropna().unique()])
                        if unique_vals:
                            selected_vals = st.sidebar.multiselect(f"Filter {col}", unique_vals, default=unique_vals)
                            if selected_vals:
                                df = df[df[col].astype(str).isin(selected_vals)]

                    tab_ov, tab_uni, tab_bi, tab_ts, tab_dl = st.tabs([
                        "🔍 Overview & Quality", "📊 Univariate Analysis",
                        "🔗 Bivariate Analysis", "📈 Time Series", "⬇️ Download Report"])

                    total_rows   = df.shape[0]
                    total_cols   = df.shape[1]
                    missing_cells = df.isna().sum().sum()
                    total_cells  = total_rows * total_cols
                    completeness = ((total_cells - missing_cells) / total_cells) * 100 if total_cells > 0 else 0
                    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

                    with tab_ov:
                        st.markdown("### Raw Data Preview")
                        st.dataframe(df.head(100), use_container_width=True)
                        st.markdown("---")
                        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                        kpi1.metric("Total Rows", f"{total_rows:,}")
                        kpi2.metric("Total Columns", f"{total_cols:,}")
                        kpi3.metric("Completeness", f"{completeness:.1f}%")
                        if numeric_cols:
                            kpi4.metric(f"Avg {numeric_cols[0]}", f"{df[numeric_cols[0]].mean():.2f}")
                        else:
                            kpi4.metric("Numeric Cols", "0")
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.markdown("### Data Types & Missing Values")
                            info_df = pd.DataFrame({'Data Type': df.dtypes.astype(str),
                                                    'Missing (Count)': df.isna().sum(),
                                                    'Missing (%)': (df.isna().sum() / total_rows * 100).round(2)})
                            st.dataframe(info_df.style.background_gradient(subset=['Missing (%)'], cmap='Reds'), use_container_width=True)
                        with col2:
                            st.markdown("### Summary Statistics (Numeric)")
                            if numeric_cols:
                                st.dataframe(df.describe().T, use_container_width=True)
                            else:
                                st.info("No numerical columns found.")

                    with tab_uni:
                        st.markdown("### Distribution Analysis")
                        col_to_analyze = st.selectbox("Select a feature:", df.columns, key="c_univariate_select")
                        if pd.api.types.is_numeric_dtype(df[col_to_analyze]):
                            fig = px.histogram(df, x=col_to_analyze, marginal="box",
                                            title=f"Distribution of {col_to_analyze}", text_auto=True,
                                            template="plotly_white", color_discrete_sequence=['#32f384'])
                            st.plotly_chart(fig, use_container_width=True)
                        elif pd.api.types.is_datetime64_any_dtype(df[col_to_analyze]):
                            dc = df[col_to_analyze].dt.date.value_counts().reset_index()
                            dc.columns = [col_to_analyze, 'Record Count']
                            dc = dc.sort_values(col_to_analyze)
                            fig = px.line(dc, x=col_to_analyze, y='Record Count', text='Record Count',
                                        title=f"Record Frequency over {col_to_analyze}", template="plotly_white", markers=True)
                            fig.update_traces(line_color='#2ca02c', textposition='top center')
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            vc = df[col_to_analyze].value_counts().reset_index()
                            vc.columns = [col_to_analyze, 'Count']
                            c1, c2 = st.columns(2)
                            with c1:
                                fb = px.bar(vc.head(20), x=col_to_analyze, y='Count',
                                            title=f"Top 20 in {col_to_analyze}", template="plotly_white", text_auto='.2s')
                                fb.update_xaxes(categoryorder='total descending')
                                fb.update_traces(marker_color='#5c32f3')
                                st.plotly_chart(fb, use_container_width=True)
                            with c2:
                                fp = px.pie(vc.head(10), names=col_to_analyze, values='Count',
                                            title=f"Proportion (Top 10) in {col_to_analyze}",
                                            template="plotly_white", hole=0.4)
                                fp.update_traces(textposition='inside', textinfo='percent+label+value')
                                st.plotly_chart(fp, use_container_width=True)

                    with tab_bi:
                        # Exclude unnamed, near-empty, and ID-like columns from dropdowns
                        bi_cols_c = usable_cols_for_bivariate(df)
                        bi_num_cols_c = [c for c in bi_cols_c if pd.api.types.is_numeric_dtype(df[c])]
                        st.markdown("### Feature Relationships")
                        if not bi_cols_c:
                            st.info("No suitable columns found for bivariate analysis.")
                        else:
                            c1, c2 = st.columns(2)
                            with c1:
                                x_axis = st.selectbox("X-Axis:", bi_cols_c, index=0, key='c_bi_x')
                            with c2:
                                default_y_idx = next((i for i, c in enumerate(bi_cols_c) if pd.api.types.is_numeric_dtype(df[c])), 0)
                                y_axis = st.selectbox("Y-Axis:", bi_cols_c, index=default_y_idx, key='c_bi_y')
                            if pd.api.types.is_numeric_dtype(df[y_axis]) and not pd.api.types.is_numeric_dtype(df[x_axis]):
                                fig = px.box(df, x=x_axis, y=y_axis, color=x_axis, title=f"{y_axis} by {x_axis}", template="plotly_white")
                                if df[x_axis].nunique() > 15: fig.update_layout(showlegend=False)
                                st.plotly_chart(fig, use_container_width=True)
                            elif not pd.api.types.is_numeric_dtype(df[x_axis]) and not pd.api.types.is_numeric_dtype(df[y_axis]):
                                fig = px.histogram(df, x=x_axis, color=y_axis, barmode='group',
                                                title=f"{x_axis} by {y_axis}", text_auto=True, template="plotly_white")
                                if df[x_axis].nunique() > 15: fig.update_layout(showlegend=False)
                                st.plotly_chart(fig, use_container_width=True)
                            elif pd.api.types.is_numeric_dtype(df[x_axis]) and pd.api.types.is_numeric_dtype(df[y_axis]):
                                fig = px.scatter(df, x=x_axis, y=y_axis, opacity=0.6,
                                                title=f"Scatter: {x_axis} vs {y_axis}", template="plotly_white",
                                                color_discrete_sequence=['#ee3e32'])
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("💡 Select one categorical (X) and one numeric (Y) feature.")
                            st.markdown("---")
                            if len(bi_num_cols_c) > 1:
                                st.markdown("### Correlation Heatmap")
                                corr = df[bi_num_cols_c].dropna().corr()
                                fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto",
                                                    title="Pearson Correlation Between Numeric Features",
                                                    color_continuous_scale="RdBu_r", template="plotly_white")
                                st.plotly_chart(fig_corr, use_container_width=True)

                    with tab_ts:
                        date_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
                        if not date_cols:
                            for c in df.columns:
                                if 'date' in c.lower() or 'time' in c.lower() or 'timestamp' in c.lower():
                                    date_cols.append(c)
                        if date_cols and numeric_cols:
                            st.markdown("### Time Series Analysis")
                            c1, c2, c3, c4 = st.columns(4)
                            with c1: ts_dc = st.selectbox("Date Column:", date_cols, key='c_ts_dc')
                            with c2: ts_vc = st.selectbox("Numeric Metric:", numeric_cols, key='c_ts_vc')
                            with c3: agg_f = st.selectbox("Aggregation:", ["mean","sum","count","min","max"], key='c_ts_agg')
                            clean_ts_cols_c = usable_cols_for_bivariate(df)
                            with c4: ts_cc = st.selectbox("Color by:", ["None"] + [c for c in clean_ts_cols_c if c not in [ts_dc, ts_vc]], key='c_ts_cc')
                            if st.button("Generate Time Series Chart", key='c_ts_btn'):
                                try:
                                    df_ts = df.copy()
                                    df_ts[ts_dc] = pd.to_datetime(df_ts[ts_dc])
                                    df_ts['Date_Trunc'] = df_ts[ts_dc].dt.floor('D')
                                    gc = ['Date_Trunc'] + ([ts_cc] if ts_cc != "None" else [])
                                    tsg = df_ts.groupby(gc)[ts_vc].agg(agg_f).reset_index()
                                    tsg.rename(columns={ts_vc: 'Metric_Value'}, inplace=True)
                                    fk = {"data_frame": tsg, "x": 'Date_Trunc', "y": 'Metric_Value', "markers": True,
                                        "text": "Metric_Value", "title": f"Time Series: {agg_f.capitalize()} of {ts_vc}",
                                        "labels": {'Date_Trunc': 'Date', 'Metric_Value': f'{agg_f.capitalize()} {ts_vc}'},
                                        "template": "plotly_white"}
                                    if ts_cc != "None": fk["color"] = ts_cc
                                    fig_ts = px.line(**fk)
                                    fig_ts.update_traces(textposition='top center')
                                    if ts_cc == "None": fig_ts.update_traces(line_color='#00cc66', line_width=3)
                                    st.plotly_chart(fig_ts, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Could not generate time series. Error: {e}")
                        else:
                            st.info("We couldn't detect both Date/Time AND Numeric columns for time series analysis.")

                    # ── Coaching Download Report tab ──────────────────────────────
                    with tab_dl:
                        st.markdown("## ⬇️ Download Coaching Data Report")
                        st.markdown("Generates a JSON with all coaching chart data including session distributions, behaviors, and trends.")
                        if st.button("🔄 Generate Coaching JSON", key='gen_coaching_sheet', type="primary"):
                            df_c_raw_dl = load_and_prep_data(FILE_PATH, sheet_name=selected_sheet)
                            report, err = build_coaching_sheet_json(df_c_raw_dl)
                            if err:
                                st.error(err)
                            else:
                                js = json.dumps(report, default=safe_json, indent=2, ensure_ascii=False)
                                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                                n_charts = sum(len(s['charts']) for s in report['sections'])
                                st.success(f"✅ {n_charts} chart datasets generated.")
                                st.download_button("⬇️ Download Coaching JSON", data=js.encode("utf-8"),
                                                file_name=f"coaching_report_{ts}.json",
                                                mime="application/json", key='dl_coaching_sheet')
                        st.markdown("---")
                        st.markdown("### 👁️ JSON Structure Reference")
                        st.json({
                            "generated_at": "<ISO timestamp>",
                            "description": "Coaching data export",
                            "sections": [{
                                "filter_combination": {"note": "Coaching data"},
                                "charts": [{"chart_title": "...", "description": "...", "data": [{"col": "value"}]}]
                            }]
                        })

                # ══════════════════════════════════════════════════════════════════
                # GENERIC SINGLE-SHEET EDA (any other sheet)
                # ══════════════════════════════════════════════════════════════════
                else:
                    df = load_and_prep_data(FILE_PATH, sheet_name=selected_sheet)

                    st.sidebar.markdown("### 🎛️ Dynamic Filters")
                    cat_cols = [c for c in df.columns if df[c].dtype == 'object'
                                and df[c].nunique() < 20 and df[c].nunique() > 0]
                    for col in cat_cols:
                        unique_vals = sorted([str(x) for x in df[col].dropna().unique()])
                        if unique_vals:
                            selected_vals = st.sidebar.multiselect(f"Filter {col}", unique_vals, default=unique_vals)
                            if selected_vals:
                                df = df[df[col].astype(str).isin(selected_vals)]

                    st.subheader("Key Performance Indicators (KPIs)")
                    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                    total_rows    = df.shape[0]
                    total_cols_n  = df.shape[1]
                    missing_cells = df.isna().sum().sum()
                    total_cells   = total_rows * total_cols_n
                    completeness  = ((total_cells - missing_cells) / total_cells) * 100 if total_cells > 0 else 0
                    kpi1.metric("Total Rows", f"{total_rows:,}")
                    kpi2.metric("Total Columns", f"{total_cols_n:,}")
                    kpi3.metric("Completeness", f"{completeness:.1f}%")
                    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
                    if 'score' in [c.lower() for c in df.columns] and numeric_cols:
                        scg = [c for c in df.columns if c.lower() == 'score' and c in numeric_cols][0]
                        kpi4.metric(f"Avg {scg}", f"{df[scg].mean():.2f}")
                    elif numeric_cols:
                        kpi4.metric(f"Avg {numeric_cols[0]}", f"{df[numeric_cols[0]].mean():.2f}")
                    else:
                        kpi4.metric("Numeric Cols", "0")

                    st.markdown("---")
                    tab_ov, tab_uni, tab_bi, tab_ts = st.tabs([
                        "🔍 Overview & Quality", "📊 Univariate Analysis",
                        "🔗 Bivariate Analysis", "📈 Time Series"])

                    with tab_ov:
                        st.markdown("### Raw Data Preview")
                        st.dataframe(df.head(100), use_container_width=True)
                        st.markdown("---")
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.markdown("### Data Types & Missing Values")
                            info_df = pd.DataFrame({'Data Type': df.dtypes.astype(str),
                                                    'Missing (Count)': df.isna().sum(),
                                                    'Missing (%)': (df.isna().sum() / total_rows * 100).round(2)})
                            st.dataframe(info_df.style.background_gradient(subset=['Missing (%)'], cmap='Reds'), use_container_width=True)
                        with col2:
                            st.markdown("### Summary Statistics (Numeric)")
                            if numeric_cols:
                                st.dataframe(df.describe().T, use_container_width=True)
                            else:
                                st.info("No numerical columns found.")

                    with tab_uni:
                        st.markdown("### Distribution Analysis")
                        col_to_analyze = st.selectbox("Select a feature:", df.columns, key="univariate_select")
                        if pd.api.types.is_numeric_dtype(df[col_to_analyze]):
                            fig = px.histogram(df, x=col_to_analyze, marginal="box",
                                            title=f"Distribution of {col_to_analyze}", text_auto=True,
                                            template="plotly_white", color_discrete_sequence=['#32f384'])
                            st.plotly_chart(fig, use_container_width=True)
                        elif pd.api.types.is_datetime64_any_dtype(df[col_to_analyze]):
                            dc = df[col_to_analyze].dt.date.value_counts().reset_index()
                            dc.columns = [col_to_analyze, 'Record Count']
                            dc = dc.sort_values(col_to_analyze)
                            fig = px.line(dc, x=col_to_analyze, y='Record Count', text='Record Count',
                                        title=f"Record Frequency over {col_to_analyze}", template="plotly_white", markers=True)
                            fig.update_traces(line_color='#2ca02c', textposition='top center')
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            vc = df[col_to_analyze].value_counts().reset_index()
                            vc.columns = [col_to_analyze, 'Count']
                            c1, c2 = st.columns(2)
                            with c1:
                                fb = px.bar(vc.head(20), x=col_to_analyze, y='Count',
                                            title=f"Top 20 in {col_to_analyze}", template="plotly_white", text_auto='.2s')
                                fb.update_xaxes(categoryorder='total descending')
                                fb.update_traces(marker_color='#5c32f3')
                                st.plotly_chart(fb, use_container_width=True)
                            with c2:
                                fp = px.pie(vc.head(10), names=col_to_analyze, values='Count',
                                            title=f"Proportion (Top 10) in {col_to_analyze}",
                                            template="plotly_white", hole=0.4)
                                fp.update_traces(textposition='inside', textinfo='percent+label+value')
                                st.plotly_chart(fp, use_container_width=True)

                    with tab_bi:
                        st.markdown("### Feature Relationships")
                        c1, c2 = st.columns(2)
                        with c1: x_axis = st.selectbox("X-Axis:", df.columns, index=0)
                        with c2:
                            yi = df.columns.get_loc(numeric_cols[0]) if numeric_cols else 0
                            y_axis = st.selectbox("Y-Axis:", df.columns, index=yi)
                        if pd.api.types.is_numeric_dtype(df[y_axis]) and not pd.api.types.is_numeric_dtype(df[x_axis]):
                            fig = px.box(df, x=x_axis, y=y_axis, color=x_axis, title=f"{y_axis} by {x_axis}", template="plotly_white")
                            if df[x_axis].nunique() > 15: fig.update_layout(showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
                        elif not pd.api.types.is_numeric_dtype(df[x_axis]) and not pd.api.types.is_numeric_dtype(df[y_axis]):
                            fig = px.histogram(df, x=x_axis, color=y_axis, barmode='group',
                                            title=f"{x_axis} by {y_axis}", text_auto=True, template="plotly_white")
                            if df[x_axis].nunique() > 15: fig.update_layout(showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
                        elif pd.api.types.is_numeric_dtype(df[x_axis]) and pd.api.types.is_numeric_dtype(df[y_axis]):
                            lc = next((c for c in df.columns if 'id' in c.lower() or 'name' in c.lower()
                                    or 'agent' in c.lower() or 'employee' in c.lower()), None)
                            fig = px.scatter(df, x=x_axis, y=y_axis, opacity=0.6, text=lc,
                                            title=f"Scatter: {x_axis} vs {y_axis}", template="plotly_white",
                                            color_discrete_sequence=['#ee3e32'])
                            if lc: fig.update_traces(textposition='top center')
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("💡 Select one categorical (X) and one numeric (Y) feature.")
                        st.markdown("---")
                        if len(numeric_cols) > 1:
                            st.markdown("### Correlation Heatmap")
                            corr = df[numeric_cols].dropna().corr()
                            fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto",
                                                title="Pearson Correlation Between Numeric Features",
                                                color_continuous_scale="RdBu_r", template="plotly_white")
                            st.plotly_chart(fig_corr, use_container_width=True)

                    with tab_ts:
                        date_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
                        if not date_cols:
                            for c in df.columns:
                                if 'date' in c.lower() or 'time' in c.lower() or 'timestamp' in c.lower():
                                    date_cols.append(c)
                        if date_cols and numeric_cols:
                            st.markdown("### Time Series Analysis")
                            c1, c2, c3, c4 = st.columns(4)
                            with c1: ts_dc = st.selectbox("Date Column:", date_cols)
                            with c2: ts_vc = st.selectbox("Numeric Metric:", numeric_cols)
                            with c3: agg_f = st.selectbox("Aggregation:", ["mean","sum","count","min","max"])
                            with c4: ts_cc = st.selectbox("Color by:", ["None"] + [c for c in df.columns if c not in [ts_dc, ts_vc]])
                            if st.button("Generate Time Series Chart"):
                                try:
                                    df_ts = df.copy()
                                    df_ts[ts_dc] = pd.to_datetime(df_ts[ts_dc])
                                    df_ts['Date_Trunc'] = df_ts[ts_dc].dt.floor('D')
                                    gc = ['Date_Trunc'] + ([ts_cc] if ts_cc != "None" else [])
                                    tsg = df_ts.groupby(gc)[ts_vc].agg(agg_f).reset_index()
                                    tsg.rename(columns={ts_vc: 'Metric_Value'}, inplace=True)
                                    fk = {"data_frame": tsg, "x": 'Date_Trunc', "y": 'Metric_Value', "markers": True,
                                        "text": "Metric_Value", "title": f"Time Series: {agg_f.capitalize()} of {ts_vc}",
                                        "labels": {'Date_Trunc': 'Date', 'Metric_Value': f'{agg_f.capitalize()} {ts_vc}'},
                                        "template": "plotly_white"}
                                    if ts_cc != "None": fk["color"] = ts_cc
                                    fig_ts = px.line(**fk)
                                    fig_ts.update_traces(textposition='top center')
                                    if ts_cc == "None": fig_ts.update_traces(line_color='#00cc66', line_width=3)
                                    st.plotly_chart(fig_ts, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Could not generate time series. Error: {e}")
                        else:
                            st.info("We couldn't detect both Date/Time AND Numeric columns for time series analysis.")
            else:
                df = load_and_prep_data(FILE_PATH)

                st.sidebar.markdown("### 🎛️ Dynamic Filters")
                cat_cols = [c for c in df.columns if df[c].dtype == 'object'
                            and df[c].nunique() < 20 and df[c].nunique() > 0]
                for col in cat_cols:
                    unique_vals = sorted([str(x) for x in df[col].dropna().unique()])
                    if unique_vals:
                        selected_vals = st.sidebar.multiselect(f"Filter {col}", unique_vals, default=unique_vals)
                        if selected_vals:
                            df = df[df[col].astype(str).isin(selected_vals)]

                st.subheader("Key Performance Indicators (KPIs)")
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                total_rows    = df.shape[0]
                total_cols    = df.shape[1]
                missing_cells = df.isna().sum().sum()
                total_cells   = total_rows * total_cols
                completeness  = ((total_cells - missing_cells) / total_cells) * 100 if total_cells > 0 else 0
                kpi1.metric("Total Rows", f"{total_rows:,}")
                kpi2.metric("Total Columns", f"{total_cols:,}")
                kpi3.metric("Completeness", f"{completeness:.1f}%")
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
                if 'score' in [c.lower() for c in df.columns] and numeric_cols:
                    scg = [c for c in df.columns if c.lower() == 'score' and c in numeric_cols][0]
                    kpi4.metric(f"Avg {scg}", f"{df[scg].mean():.2f}")
                elif numeric_cols:
                    kpi4.metric(f"Avg {numeric_cols[0]}", f"{df[numeric_cols[0]].mean():.2f}")
                else:
                    kpi4.metric("Numeric Cols", "0")

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error(f"Database file not found at {FILE_PATH}.")