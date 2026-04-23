"""
clustering_tabs.py  —  AttritionIQ  |  Performance Clustering Tabs
Drop-in module for the existing Streamlit app.

Usage in your main app:
    from clustering_tabs import render_performance_clustering_tab, render_merged_clustering_tab

    with tab_perf_cluster:
        render_performance_clustering_tab()
    with tab_merged_cluster:
        render_merged_clustering_tab()

FIXES APPLIED (v2):
    Bug 1 — _clean_merged: agent ID column detection now checks agent_id first
    Bug 2 — render_merged_clustering_tab: score/quality features default OFF
    Bug 3 — _remap_to_quadrant: collision guard prevents 2+ clusters mapping to same quadrant
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

# Primary paths — parquet files are tracked on GitHub (cloud-safe)
# Falls back to CSV for local development automatically
PERF_PATH   = "Data/PH_Agent_Performance_Nov_2025 (1).csv"   # tracked on GitHub
MERGED_PATH = "Data/final_merged_datasett.parquet"            # tracked on GitHub

# ──────────────────────────────────────────────────────────────────────────────
# CLUSTER DEFINITIONS  (for the 2-axis quadrant approach)
# ──────────────────────────────────────────────────────────────────────────────
CLUSTER_META = {
    0: {"label": "⭐ Star Performer",     "color": "#22c55e", "desc": "High occupancy & fast calls. Protect their schedules and involve them in training others."},
    1: {"label": "⏳ Busy but Slow",      "color": "#f59e0b", "desc": "High occupancy but slow calls. Coaching on hold/wrap time efficiency helps."},
    2: {"label": "😴 Underutilized",      "color": "#60a5fa", "desc": "Low occupancy but fast when working. Consider whether call routing sends enough volume."},
    3: {"label": "🎯 Coaching Priority",  "color": "#f43f5e", "desc": "Low occupancy AND slow calls. Structured coaching plan + closer TL monitoring needed."},
}

ALGO_OPTIONS = [
    "Median Split (matches Performance page)",
    "K-Means",
    "DBSCAN",
    "Agglomerative Hierarchical",
]

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading data…")
def _load(path: str) -> pd.DataFrame | None:
    """Load parquet / CSV / XLSX; auto-fallback parquet ↔ CSV for cloud vs local."""
    import os
    # Build a list of candidate paths to try
    candidates = [path]
    if path.endswith(".csv"):
        candidates.append(path.replace(".csv", ".parquet"))
    elif path.endswith(".parquet"):
        candidates.append(path.replace(".parquet", ".csv"))

    for p in candidates:
        if not os.path.exists(p):
            continue
        try:
            if p.endswith(".parquet"):
                return pd.read_parquet(p)
            elif p.endswith(".xlsx") or p.endswith(".xls"):
                return pd.read_excel(p)
            else:
                return pd.read_csv(p)
        except Exception as e:
            st.warning(f"⚠️ Could not load `{p}`: {e} — trying next option…")
            continue

    st.error(
        f"❌ Could not find or load data. Tried: {candidates}. "
        "Please check the Data/ folder."
    )
    return None


def _clean_perf(df: pd.DataFrame) -> pd.DataFrame:
    """Clean perf data and aggregate to one row per agent."""
    num_cols = ["Calls_Answered", "AHT", "Avg_Talk_Time", "Avg_Hold_Time",
                "Avg_Wrap_Time", "Occupancy %", "Productive_Hours", "Idle_Hrs"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("%", "pct")
    df = df[df.get("calls_answered", pd.Series(0)) > 0].copy()
    df = df.dropna(subset=["aht", "occupancy_pct"])
    # Aggregate to agent-level (one row per agent)
    if "agent_id" in df.columns:
        agg = {c: "mean" for c in ["aht", "occupancy_pct", "avg_talk_time",
               "avg_hold_time", "avg_wrap_time", "productive_hours", "idle_hrs"]
               if c in df.columns}
        if "calls_answered" in df.columns:
            agg["calls_answered"] = "sum"
        if "account" in df.columns:
            agg["account"] = "first"
        df = df.groupby("agent_id", as_index=False).agg(agg)
    return df


def _clean_merged(df: pd.DataFrame) -> pd.DataFrame:
    """Clean merged data: de-dup columns, cast, aggregate per agent."""
    # De-duplicate column names that collide after case-folding
    seen = {}
    new_cols = []
    for c in df.columns:
        key = c.strip().lower().replace(" ", "_").replace("%", "pct")
        if key in seen:
            seen[key] += 1
            new_cols.append(f"{key}_{seen[key]}")
        else:
            seen[key] = 0
            new_cols.append(key)
    df.columns = new_cols

    num_cols = ["aht", "occupancy_pct", "calls_answered", "avg_talk_time",
                "avg_hold_time", "avg_wrap_time", "score",
                "productive_hours", "idle_hrs", "age"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[df.get("calls_answered", pd.Series(0)) > 0].copy()
    df = df.dropna(subset=["aht", "occupancy_pct"])

    # ── BUG 1 FIX: check agent_id first, then employee_id, then agent ──────────
    id_col = next(
        (c for c in ["agent_id", "employee_id", "agent"] if c in df.columns),
        None
    )
    if id_col is None:
        st.error(
            "❌ No agent ID column found in merged dataset. "
            "Expected one of: `agent_id`, `employee_id`, `agent`. "
            "Please check your CSV column names."
        )
        return df  # return as-is so caller can still display error

    agg = {c: "mean" for c in ["aht", "occupancy_pct", "avg_talk_time",
           "avg_hold_time", "avg_wrap_time", "productive_hours", "idle_hrs",
           "score", "age"] if c in df.columns}
    if "calls_answered" in df.columns:
        agg["calls_answered"] = "sum"
    for c in ["gender", "marital_status", "current_status",
              "account_merged", "account_perf", "lob"]:
        if c in df.columns:
            agg[c] = "first"

    df = df.groupby(id_col, as_index=False).agg(agg)

    # Unified account column
    if "account_perf" in df.columns:
        df["account"] = df["account_perf"]
    elif "account_merged" in df.columns:
        df["account"] = df["account_merged"]

    return df


def _assign_quadrant_labels(df: pd.DataFrame, occ_col: str, aht_col: str) -> np.ndarray:
    """
    Label-based fallback: occupancy median × AHT median → 4 quadrants.
    Returns integer cluster labels 0-3 matching CLUSTER_META.
    """
    occ_med = df[occ_col].median()
    aht_med = df[aht_col].median()
    conditions = [
        (df[occ_col] >= occ_med) & (df[aht_col] <= aht_med),   # 0 Star
        (df[occ_col] >= occ_med) & (df[aht_col] >  aht_med),   # 1 Busy+Slow
        (df[occ_col] <  occ_med) & (df[aht_col] <= aht_med),   # 2 Underutilized
        (df[occ_col] <  occ_med) & (df[aht_col] >  aht_med),   # 3 Coaching
    ]
    return np.select(conditions, [0, 1, 2, 3], default=3)


def _run_algo(X_scaled: np.ndarray, algo: str, n_clusters: int = 4,
              eps: float = 0.5, min_samples: int = 5) -> np.ndarray:
    """Run selected algorithm and return raw labels."""
    if "K-Means" in algo:
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        return km.fit_predict(X_scaled)
    elif "DBSCAN" in algo:
        db = DBSCAN(eps=eps, min_samples=min_samples)
        return db.fit_predict(X_scaled)
    else:
        ag = AgglomerativeClustering(n_clusters=n_clusters)
        return ag.fit_predict(X_scaled)


def _remap_to_quadrant(labels: np.ndarray, df_work: pd.DataFrame,
                       occ_col: str, aht_col: str) -> np.ndarray:
    """
    Re-map algorithm cluster IDs to our 4 semantic quadrants using
    per-cluster centroids of occupancy & AHT.

    BUG 3 FIX: Collision guard — if two K-Means clusters land in the same
    quadrant, the second one is assigned the next best available quadrant
    instead of overwriting the first. Falls back to rule-based assignment
    for any remaining unassigned agents.
    """
    occ_med = df_work[occ_col].median()
    aht_med = df_work[aht_col].median()

    unique = [l for l in np.unique(labels) if l != -1]
    if len(unique) == 0:
        return _assign_quadrant_labels(df_work, occ_col, aht_col)

    # Build centroid info sorted by cluster size descending
    # (larger clusters get quadrant assignment priority)
    cluster_info = []
    for uid in unique:
        mask = labels == uid
        occ_c = df_work.loc[mask, occ_col].mean()
        aht_c = df_work.loc[mask, aht_col].mean()
        size  = mask.sum()
        cluster_info.append((uid, occ_c, aht_c, size))

    cluster_info.sort(key=lambda x: -x[3])  # largest first

    mapping = {}
    used_quadrants = set()

    for uid, occ_c, aht_c, _ in cluster_info:
        # Determine the "natural" quadrant for this centroid
        if   occ_c >= occ_med and aht_c <= aht_med:
            preferred = [0, 2, 1, 3]   # Star → fallback priority
        elif occ_c >= occ_med and aht_c >  aht_med:
            preferred = [1, 0, 3, 2]   # Busy+Slow → fallback priority
        elif occ_c <  occ_med and aht_c <= aht_med:
            preferred = [2, 3, 0, 1]   # Underutilized → fallback priority
        else:
            preferred = [3, 2, 1, 0]   # Coaching → fallback priority

        assigned = None
        for q in preferred:
            if q not in used_quadrants:
                assigned = q
                used_quadrants.add(q)
                break

        if assigned is not None:
            mapping[uid] = assigned
        # else: all 4 quadrants taken (shouldn't happen with k=4, but safe)

    # Apply mapping; anything unmapped (e.g. DBSCAN noise -1) falls back
    # to the rule-based quadrant assignment per agent
    result = np.full(len(labels), -1, dtype=int)
    for uid, quad in mapping.items():
        result[labels == uid] = quad

    # Handle noise / unmapped points with rule-based fallback
    noise_mask = result == -1
    if noise_mask.any():
        fallback = _assign_quadrant_labels(df_work[noise_mask].reset_index(drop=True),
                                           occ_col, aht_col)
        result[noise_mask] = fallback

    return result


def _pca_3d(X_scaled: np.ndarray) -> tuple[np.ndarray, list[float]]:
    pca = PCA(n_components=3, random_state=42)
    X3 = pca.fit_transform(X_scaled)
    evr = [round(v * 100, 1) for v in pca.explained_variance_ratio_]
    return X3, evr


def _silhouette(X_scaled, labels):
    u = np.unique(labels)
    u = u[u != -1]
    if len(u) < 2:
        return None
    try:
        mask = labels != -1
        return round(silhouette_score(X_scaled[mask], labels[mask]), 3)
    except Exception:
        return None


def _cluster_cards(df_work, cluster_col):
    counts = df_work[cluster_col].value_counts().to_dict()
    cols = st.columns(4)
    for i, (cid, meta) in enumerate(CLUSTER_META.items()):
        cnt = counts.get(cid, 0)
        pct = round(cnt / len(df_work) * 100, 1) if len(df_work) else 0
        with cols[i]:
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, {meta['color']}22, {meta['color']}11);
                    border: 1.5px solid {meta['color']}66;
                    border-radius: 12px;
                    padding: 14px 16px;
                    margin-bottom: 6px;
                ">
                    <div style="font-size:22px; font-weight:800; color:{meta['color']};">{cnt:,}</div>
                    <div style="font-size:12px; font-weight:600; color:{meta['color']}; margin-bottom:4px;">{meta['label']}</div>
                    <div style="font-size:11px; color:#999;">{pct}% of agents</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _plot_3d(df_plot, pca_coords, cluster_col, evr, title):
    fig = go.Figure()
    for cid, meta in CLUSTER_META.items():
        mask = df_plot[cluster_col] == cid
        if mask.sum() == 0:
            continue
        hover = [
            f"Agent: {row.get('agent_id', row.get('employee_id', row.get('agent', '?')))}<br>"
            f"Occupancy: {row.get('occupancy_pct', 0):.1f}%<br>"
            f"AHT: {row.get('aht', 0):.0f}s"
            for _, row in df_plot[mask].iterrows()
        ]
        fig.add_trace(go.Scatter3d(
            x=pca_coords[mask, 0],
            y=pca_coords[mask, 1],
            z=pca_coords[mask, 2],
            mode="markers",
            name=meta["label"],
            marker=dict(
                size=4.5,
                color=meta["color"],
                opacity=0.82,
                line=dict(width=0.5, color="white"),
            ),
            text=hover,
            hovertemplate="%{text}<extra></extra>",
        ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#e2e8f0"), x=0.02),
        scene=dict(
            xaxis_title=f"PC1 ({evr[0]}%)",
            yaxis_title=f"PC2 ({evr[1]}%)",
            zaxis_title=f"PC3 ({evr[2]}%)",
            bgcolor="rgba(10,10,20,0.0)",
            xaxis=dict(gridcolor="#334155", color="#94a3b8"),
            yaxis=dict(gridcolor="#334155", color="#94a3b8"),
            zaxis=dict(gridcolor="#334155", color="#94a3b8"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="#e2e8f0"), bgcolor="rgba(0,0,0,0)"),
        height=550,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def _feature_importance_chart(df_work, cluster_col, features):
    """
    Pseudo feature-importance: per-cluster z-score deviation from global mean.
    Shows which features most differentiate each cluster.
    """
    global_mean = df_work[features].mean()
    global_std  = df_work[features].std().replace(0, 1)

    rows = []
    for cid, meta in CLUSTER_META.items():
        sub = df_work[df_work[cluster_col] == cid]
        if len(sub) == 0:
            continue
        cl_mean = sub[features].mean()
        z = (cl_mean - global_mean) / global_std
        for feat, zval in z.items():
            rows.append({"Cluster": meta["label"], "Feature": feat,
                         "Z-Score Deviation": round(zval, 3),
                         "color": meta["color"]})

    imp_df = pd.DataFrame(rows)
    if imp_df.empty:
        return None

    fig = px.bar(
        imp_df,
        x="Feature",
        y="Z-Score Deviation",
        color="Cluster",
        barmode="group",
        color_discrete_map={m["label"]: m["color"] for m in CLUSTER_META.values()},
        title="Feature Importance by Cluster  (Z-Score deviation from global mean)",
        labels={"Z-Score Deviation": "Deviation (σ)"},
        height=420,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#475569", line_width=1)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        xaxis=dict(gridcolor="#1e293b", tickangle=-30, color="#94a3b8"),
        yaxis=dict(gridcolor="#1e293b", color="#94a3b8"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        title_font=dict(size=14),
    )
    return fig


def _algo_sidebar_controls(prefix: str):
    """Render algorithm controls in a sidebar-like expander."""
    with st.expander("⚙️ Algorithm Settings", expanded=False):
        algo = st.selectbox(
            "Clustering Algorithm",
            ALGO_OPTIONS,
            index=0,          # default = Median Split (matches Performance page)
            key=f"{prefix}_algo",
        )
        n_clusters = st.slider("Number of Clusters (K-Means / Hierarchical)",
                               2, 8, 4, key=f"{prefix}_k")
        eps = st.slider("DBSCAN – ε (epsilon)", 0.1, 3.0, 0.5,
                        step=0.05, key=f"{prefix}_eps")
        min_s = st.slider("DBSCAN – min_samples", 2, 20, 5, key=f"{prefix}_mins")
    return algo, n_clusters, eps, min_s


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — PERFORMANCE DATASET CLUSTERING
# ──────────────────────────────────────────────────────────────────────────────

def render_performance_clustering_tab(perf_path: str = PERF_PATH):
    st.markdown("## 📊 Performance Dataset — Agent Clustering")
    st.caption(f"Source: `{perf_path}`")

    df_raw = _load(perf_path)
    if df_raw is None:
        return

    df = _clean_perf(df_raw)
    st.markdown(f"**{len(df):,}** unique agents after aggregation")

    # ── Controls ─────────────────────────────────────────────────────────────
    col1, col2 = st.columns([2, 1])
    with col1:
        algo, n_clusters, eps, min_s = _algo_sidebar_controls("perf")
    with col2:
        account_opts = ["All"] + sorted(df["account"].dropna().unique().tolist()) \
            if "account" in df.columns else ["All"]
        account_sel = st.selectbox("Filter by Account", account_opts, key="perf_acct")

    if account_sel != "All":
        df = df[df["account"] == account_sel]

    # ── Feature selection ─────────────────────────────────────────────────────
    PERF_FEATURES = [c for c in [
        "occupancy_pct", "aht", "avg_talk_time", "avg_hold_time",
        "avg_wrap_time", "calls_answered", "productive_hours", "idle_hrs"
    ] if c in df.columns]

    with st.expander("🔧 Feature Selection", expanded=False):
        chosen = st.multiselect(
            "Features used for clustering",
            PERF_FEATURES,
            default=["occupancy_pct", "aht", "avg_talk_time", "avg_hold_time"],
            key="perf_feats",
        )
    if len(chosen) < 2:
        st.warning("Select at least 2 features.")
        return

    df_work = df.dropna(subset=chosen).copy()

    if df_work.empty or len(df_work) < 5:
        st.warning(
            f"⚠️ Not enough valid records to cluster after filtering "
            f"(found {len(df_work)} rows — need at least 5). "
            "Try selecting a different account or adjusting the feature selection."
        )
        return

    # ── Run clustering ────────────────────────────────────────────────────────
    try:
        if "Median Split" in algo:
            # Pure median split on occupancy × AHT — matches Performance page exactly
            st.info(
                "🔗 **Median Split** assigns agents using the same occupancy/AHT boundary "
                "as the Performance page — counts will match exactly.",
                icon="ℹ️",
            )
            df_work["cluster"] = _assign_quadrant_labels(df_work, "occupancy_pct", "aht")
            # PCA still on chosen features for the 3-D plot
            chosen_num = [c for c in chosen if c in df_work.columns]
            Xp = df_work[chosen_num].fillna(df_work[chosen_num].median()).values
            X_scaled = StandardScaler().fit_transform(Xp)
            X_pca, evr = _pca_3d(X_scaled)
            pca_arr = X_pca
            sil = None   # silhouette not meaningful for rule-based split
        else:
            # Machine-learning algorithms — impute NaNs first so K-Means never crashes
            Xp = df_work[chosen].copy()
            for col in Xp.columns:
                Xp[col] = Xp[col].fillna(Xp[col].median())
            X_scaled = StandardScaler().fit_transform(Xp.values)

            raw_labels = _run_algo(X_scaled, algo, n_clusters, eps, min_s)
            df_work["cluster"] = _remap_to_quadrant(raw_labels, df_work, "occupancy_pct", "aht")
            X_pca, evr = _pca_3d(X_scaled)
            pca_arr = X_pca
            sil = _silhouette(X_scaled, raw_labels)
            st.info(
                f"⚠️ **{algo}** groups agents by similarity across all selected features. "
                "Counts will differ from the Performance page (which uses only occupancy × AHT).",
                icon="ℹ️",
            )

        # ── Summary cards ───────────────────────────────────────────────────────
        st.markdown("### Cluster Summary")
        _cluster_cards(df_work, "cluster")
        if sil is not None:
            st.caption(f"Silhouette Score: **{sil}**  (closer to 1.0 = better-separated clusters)")

        st.markdown("---")

        # ── 3-D scatter ─────────────────────────────────────────────────────────
        st.markdown("### 3D Cluster Visualization (PCA)")
        fig3d = _plot_3d(df_work, pca_arr, "cluster", evr,
                         f"Performance Clusters — {algo}")
        st.plotly_chart(fig3d, use_container_width=True)

        # ── Feature importance ───────────────────────────────────────────────────
        st.markdown("### Feature Importance by Cluster")
        fig_imp = _feature_importance_chart(df_work, "cluster", chosen)
        if fig_imp:
            st.plotly_chart(fig_imp, use_container_width=True)
        st.caption("Each bar shows how far that cluster's average deviates from the global mean "
                   "(in standard deviations). Positive = above average, Negative = below average.")

        # ── Cluster legend ───────────────────────────────────────────────────────
        st.markdown("### Cluster Definitions")
        cols = st.columns(2)
        for i, (cid, meta) in enumerate(CLUSTER_META.items()):
            with cols[i % 2]:
                st.markdown(
                    f"<div style='padding:10px 14px;border-left:4px solid {meta['color']};"
                    f"background:{meta['color']}11;border-radius:6px;margin-bottom:10px;'>"
                    f"<b style='color:{meta['color']};'>{meta['label']}</b><br>"
                    f"<span style='font-size:13px;color:#94a3b8;'>{meta['desc']}</span></div>",
                    unsafe_allow_html=True,
                )

        # ── Data table ──────────────────────────────────────────────────────────
        with st.expander("🗃️ View Clustered Data Sample", expanded=False):
            show_cols = ["agent_id", "account", "date"] + chosen + ["cluster"]
            show_cols = [c for c in show_cols if c in df_work.columns]
            df_show = df_work[show_cols].copy()
            df_show["cluster_label"] = df_show["cluster"].map(
                {k: v["label"] for k, v in CLUSTER_META.items()})
            st.dataframe(df_show.head(500), use_container_width=True)

        # ── Download ────────────────────────────────────────────────────────────
        csv = df_work.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Clustered Performance Data",
                           csv, "perf_clusters.csv", "text/csv")

    except Exception as _e:
        st.error(f"⚠️ Clustering failed: {_e}")
        st.info("Try changing the algorithm settings, reducing features, or selecting a different account.")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — MERGED DATASET CLUSTERING
# ──────────────────────────────────────────────────────────────────────────────

def render_merged_clustering_tab(merged_path: str = MERGED_PATH):
    st.markdown("## 🔗 Merged Dataset — Holistic Agent Clustering")
    st.caption(f"Source: `{merged_path}`")

    df_raw = _load(merged_path)
    if df_raw is None:
        return

    df = _clean_merged(df_raw)
    st.markdown(f"**{len(df):,}** unique agents after aggregation")

    # ── Controls ─────────────────────────────────────────────────────────────
    col1, col2 = st.columns([2, 1])
    with col1:
        algo, n_clusters, eps, min_s = _algo_sidebar_controls("merged")
    with col2:
        if "account" in df.columns:
            accts = ["All"] + sorted(df["account"].dropna().unique().tolist())
            acct_sel = st.selectbox("Filter by Account", accts, key="mrg_acct")
            if acct_sel != "All":
                df = df[df["account"] == acct_sel]

    # ── Feature sets ─────────────────────────────────────────────────────────
    PERF_F   = [c for c in ["occupancy_pct", "aht", "avg_talk_time", "avg_hold_time",
                             "avg_wrap_time", "calls_answered", "productive_hours", "idle_hrs"]
                if c in df.columns]
    QUALITY_F = [c for c in ["score", "coaching_count"] if c in df.columns]
    DEMOG_F   = [c for c in ["age"] if c in df.columns]

    with st.expander("🔧 Feature Selection", expanded=False):
        st.markdown("**Performance features**")
        sel_perf = st.multiselect("", PERF_F,
                                  default=["occupancy_pct", "aht", "avg_talk_time", "avg_hold_time"],
                                  key="mrg_pf", label_visibility="collapsed")

        # ── BUG 2 FIX: quality features default OFF to prevent score from
        #    dominating the clustering and collapsing quadrant separation ────────
        st.markdown("**Quality / Coaching features** *(off by default — enable to include quality score in clustering)*")
        sel_qual = st.multiselect("", QUALITY_F, default=[],
                                  key="mrg_qf", label_visibility="collapsed")

        st.markdown("**Demographic features**")
        sel_dem  = st.multiselect("", DEMOG_F, default=[],
                                  key="mrg_df", label_visibility="collapsed")

    chosen = sel_perf + sel_qual + sel_dem
    if len(chosen) < 2:
        st.warning("Select at least 2 features.")
        return

    df_work = df.dropna(subset=chosen).copy()

    if df_work.empty or len(df_work) < 5:
        st.warning(
            f"⚠️ Not enough valid records to cluster after filtering "
            f"(found {len(df_work)} rows — need at least 5). "
            "Try selecting a different account or adjusting the feature selection."
        )
        return

    # ── Run clustering ────────────────────────────────────────────────────────
    try:
        if "Median Split" in algo:
            st.info(
                "🔗 **Median Split** assigns agents using the same occupancy/AHT boundary "
                "as the Performance page — counts will match exactly.",
                icon="ℹ️",
            )
            df_work["cluster"] = _assign_quadrant_labels(df_work, "occupancy_pct", "aht")
            chosen_num = [c for c in chosen if c in df_work.columns]
            Xp = df_work[chosen_num].fillna(df_work[chosen_num].median()).values
            X_scaled = StandardScaler().fit_transform(Xp)
            X_pca, evr = _pca_3d(X_scaled)
            pca_arr = X_pca
            sil = None
        else:
            # Impute NaNs so K-Means never crashes
            Xp = df_work[chosen].copy()
            for col in Xp.columns:
                Xp[col] = Xp[col].fillna(Xp[col].median())
            X_scaled = StandardScaler().fit_transform(Xp.values)

            raw_labels = _run_algo(X_scaled, algo, n_clusters, eps, min_s)

            if "occupancy_pct" in chosen and "aht" in chosen:
                df_work["cluster"] = _remap_to_quadrant(raw_labels, df_work, "occupancy_pct", "aht")
            else:
                unique_ids = [l for l in np.unique(raw_labels) if l != -1]
                remap = {old: new % 4 for new, old in enumerate(sorted(unique_ids))}
                df_work["cluster"] = np.vectorize(lambda x: remap.get(x, 3))(raw_labels)

            X_pca, evr = _pca_3d(X_scaled)
            pca_arr = X_pca
            sil = _silhouette(X_scaled, raw_labels)
            st.info(
                f"⚠️ **{algo}** groups agents by similarity across all selected features. "
                "Counts will differ from the Performance page (which uses only occupancy × AHT).",
                icon="ℹ️",
            )

        # ── Summary cards ───────────────────────────────────────────────────────
        st.markdown("### Cluster Summary")
        _cluster_cards(df_work, "cluster")
        if sil is not None:
            st.caption(f"Silhouette Score: **{sil}**  (closer to 1.0 = better-separated clusters)")

        # ── Demographic breakdown ────────────────────────────────────────────────
        if any(c in df_work.columns for c in ["gender", "marital_status", "current_status"]):
            with st.expander("👥 Demographic Breakdown by Cluster", expanded=True):
                demo_tabs = st.tabs(["Gender", "Status", "Marital Status"])
                def _bar(col_name, tab):
                    with tab:
                        if col_name not in df_work.columns:
                            st.caption("Column not available.")
                            return
                        grp = (df_work.groupby(["cluster", col_name])
                               .size().reset_index(name="count"))
                        grp["cluster_label"] = grp["cluster"].map(
                            {k: v["label"] for k, v in CLUSTER_META.items()})
                        fig = px.bar(grp, x="cluster_label", y="count",
                                     color=col_name, barmode="group",
                                     title=f"{col_name.replace('_',' ').title()} by Cluster",
                                     height=340)
                        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                          plot_bgcolor="rgba(0,0,0,0)",
                                          font=dict(color="#e2e8f0"),
                                          xaxis=dict(color="#94a3b8"),
                                          yaxis=dict(color="#94a3b8"))
                        st.plotly_chart(fig, use_container_width=True)
                _bar("gender", demo_tabs[0])
                _bar("current_status", demo_tabs[1])
                _bar("marital_status", demo_tabs[2])

        st.markdown("---")

        # ── 3-D scatter ─────────────────────────────────────────────────────────
        st.markdown("### 3D Cluster Visualization (PCA)")
        fig3d = _plot_3d(df_work, pca_arr, "cluster", evr,
                         f"Merged Dataset Clusters — {algo}")
        st.plotly_chart(fig3d, use_container_width=True)

        # ── Feature importance ───────────────────────────────────────────────────
        st.markdown("### Feature Importance by Cluster")
        fig_imp = _feature_importance_chart(df_work, "cluster", chosen)
        if fig_imp:
            st.plotly_chart(fig_imp, use_container_width=True)
        st.caption("Z-score deviation from global mean. "
                   "Shows which features most define each cluster's identity.")

        # ── Per-cluster stats table ──────────────────────────────────────────────
        st.markdown("### Per-Cluster Feature Statistics")
        stat_df = df_work.groupby("cluster")[chosen].agg(["mean", "median", "std"]).round(2)
        stat_df.index = stat_df.index.map({k: v["label"] for k, v in CLUSTER_META.items()})
        st.dataframe(stat_df, use_container_width=True)

        # ── Cluster legend ───────────────────────────────────────────────────────
        st.markdown("### Cluster Definitions")
        cols = st.columns(2)
        for i, (cid, meta) in enumerate(CLUSTER_META.items()):
            with cols[i % 2]:
                st.markdown(
                    f"<div style='padding:10px 14px;border-left:4px solid {meta['color']};"
                    f"background:{meta['color']}11;border-radius:6px;margin-bottom:10px;'>"
                    f"<b style='color:{meta['color']};'>{meta['label']}</b><br>"
                    f"<span style='font-size:13px;color:#94a3b8;'>{meta['desc']}</span></div>",
                    unsafe_allow_html=True,
                )

        # ── Raw data ────────────────────────────────────────────────────────────
        with st.expander("🗃️ View Clustered Data Sample", expanded=False):
            id_cols = [c for c in ["employee_id", "agent_id", "agent", "account"] if c in df_work.columns]
            show = id_cols + chosen + ["cluster"]
            df_show = df_work[show].copy()
            df_show["cluster_label"] = df_show["cluster"].map(
                {k: v["label"] for k, v in CLUSTER_META.items()})
            st.dataframe(df_show.head(500), use_container_width=True)

        csv = df_work.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Clustered Merged Data",
                           csv, "merged_clusters.csv", "text/csv")

    except Exception as _e:
        st.error(f"⚠️ Clustering failed: {_e}")
        st.info("Try changing the algorithm settings, reducing features, or selecting a different account.")


def show():
    tab_perf_cluster, tab_merged_cluster = st.tabs([
        "🎯 Performance Clustering",
        "🔗 Merged Clustering",
    ])

    with tab_perf_cluster:
        render_performance_clustering_tab()

    with tab_merged_cluster:
        render_merged_clustering_tab()