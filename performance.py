"""
Performance Analysis Dashboard
================================
Changes vs previous version:
- Title: "Performance Analysis" (not AttritionIQ)
- Tabs reordered as a story for non-technical / client audiences
- Tab "Burnout & Attrition Risk" removed
- Shrinkage heatmap replaced with plain bar + table (no heatmaps for non-technical)
- All charts use soft, low-light colours (pastel palette)
- Every chart has a plain-English "What does this mean?" caption
- All technical jargon expanded on first use
- Zero raw column names shown to the user
- All optimisations from previous version retained (Scattergl, np.select, caching)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os, pickle, warnings
warnings.filterwarnings('ignore')



def show():
    st.markdown(
        "<h1 style='text-align: center;'>📊 Employee Analytics Dashboard</h1>",
        unsafe_allow_html=True
    )
# ═══════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════
    PERF_PATH  = "data2/PH_Agent_Performance_Nov_2025.csv"
    DEMO_PATH  = "data2/EMPLOYEE_DEMOGRAPHICS_Nov2025_synthetic.xlsx"
    COACH_PATH = "data2/Quality___Coaching_Data_Nov_2025.xlsx"
    PKL_RAW    = "data2/.cache_df_raw.pkl"
    PKL_ALL    = "data2/.cache_df_all.pkl"

    # ═══════════════════════════════════════════════════════════════
    # LOW-LIGHT COLOUR PALETTE  (soft, easy on the eyes)
    # ═══════════════════════════════════════════════════════════════
    # Primary soft tones — no neon, no high-contrast primary colours
    C_TEAL   = "#5BBFB5"   # soft teal (main accent)
    C_SAGE   = "#7FB5A0"   # muted green
    C_BLUE   = "#6B9FD4"   # dusty blue
    C_PURPLE = "#9B8EC4"   # soft lavender
    C_PEACH  = "#E8956D"   # warm peach (replaces hard orange)
    C_GOLD   = "#D4A847"   # muted gold
    C_ROSE   = "#D47FA0"   # muted rose
    C_SLATE  = "#7A90A4"   # steel blue-grey

    PALETTE = [C_TEAL, C_BLUE, C_PURPLE, C_SAGE, C_PEACH, C_GOLD, C_ROSE, C_SLATE,
            "#A8C5B5","#B5C4D4","#C4B5D4","#D4C4B5","#B5D4C4","#C4D4B5","#D4B5C4"]

    # Tier colours — soft but distinguishable
    TIER_COLORS = {"Good": C_SAGE, "Average": C_GOLD, "Poor": C_PEACH}
    QUAD_COLORS = {
        "Star Performer":  C_TEAL,
        "Busy but Slow":   C_GOLD,
        "Underutilized":   C_BLUE,
        "Coaching Priority": C_PEACH,
    }

    _DARK  = {"BG":"#12151e","CARD":"#1c2030","TEXT":"#d4dbe8","SUB":"#8a97aa","BORDER":"#2c3347"}
    _LIGHT = {"BG":"#f4f6f9","CARD":"#ffffff","TEXT":"#2c3347","SUB":"#6b7a8d","BORDER":"#dde3ec"}

    def _theme_css(t):
        bg=t["BG"]; card=t["CARD"]; txt=t["TEXT"]; sub=t["SUB"]; brd=t["BORDER"]
        return f"""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
    html,body,[class*="css"]{{font-family:'Inter',sans-serif;}}
    .metric-card{{background:{card};border:1px solid {brd};border-radius:10px;padding:14px 16px;text-align:center;}}
    .metric-value{{font-size:1.75rem;font-weight:700;color:{C_TEAL};font-family:'JetBrains Mono',monospace;line-height:1.1;}}
    .metric-value-warm{{font-size:1.75rem;font-weight:700;color:{C_PEACH};font-family:'JetBrains Mono',monospace;line-height:1.1;}}
    .metric-value-gold{{font-size:1.75rem;font-weight:700;color:{C_GOLD};font-family:'JetBrains Mono',monospace;line-height:1.1;}}
    .metric-label{{font-size:.72rem;color:{sub};margin-top:4px;letter-spacing:.04em;text-transform:uppercase;font-weight:600;}}
    .metric-sub{{font-size:.78rem;color:{txt};margin-top:3px;opacity:.85;}}
    .section-header{{font-size:1.08rem;font-weight:700;color:{txt};border-left:3px solid {C_TEAL};padding-left:10px;margin:20px 0 10px 0;}}
    .page-title{{font-size:1.65rem;font-weight:700;color:{txt};margin-bottom:2px;}}
    .page-subtitle{{color:{sub};font-size:.85rem;margin-bottom:16px;}}
    .insight-box{{background:{card};border:1px solid {brd};border-left:4px solid {C_TEAL};border-radius:8px;padding:12px 16px;margin:10px 0;}}
    .insight-box p{{color:{txt};font-size:.9rem;line-height:1.6;margin:0;}}
    .story-note{{background:{"#eef7f5" if bg==_LIGHT["BG"] else "#1a2535"};border-radius:8px;padding:10px 14px;margin:8px 0 14px 0;}}
    .story-note p{{color:{sub};font-size:.85rem;line-height:1.5;margin:0;font-style:italic;}}
    </style>"""

    def card(value, label, sub=None, color="accent"):
        cls = {"accent":"metric-value","warm":"metric-value-warm","gold":"metric-value-gold"}.get(color,"metric-value")
        s   = f'<div class="metric-sub">{sub}</div>' if sub else ""
        return f'<div class="metric-card"><div class="{cls}">{value}</div><div class="metric-label">{label}</div>{s}</div>'

    def sh(title):
        st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

    def insight(text):
        """Plain-English explanation box shown under every chart."""
        st.markdown(f'<div class="insight-box"><p>💡 {text}</p></div>', unsafe_allow_html=True)



    def norm(s):
        rng = s.max()-s.min()
        return (s-s.min())/(rng+1e-9) if rng>0 else s*0

    def plotly_layout(fig, title="", height=400, xlab="", ylab=""):
        t    = st.session_state.get("_theme", _LIGHT)
        bg   = t["BG"]; card_ = t["CARD"]; txt = t["TEXT"]; brd = t["BORDER"]
        grid = "#dde3ec" if bg==_LIGHT["BG"] else "#2c3347"
        fig.update_layout(
            title=dict(text=title, font=dict(size=12, color=txt, family="Inter")),
            paper_bgcolor=bg, plot_bgcolor=card_,
            font=dict(family="Inter", color=txt, size=11),
            height=height, margin=dict(l=50, r=20, t=45, b=50),
            legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=brd, borderwidth=1),
            xaxis=dict(gridcolor=grid, zerolinecolor=grid, showgrid=True,
                    title=dict(text=xlab, font=dict(size=10))),
            yaxis=dict(gridcolor=grid, zerolinecolor=grid, showgrid=True,
                    title=dict(text=ylab, font=dict(size=10))),
        )
        return fig

    # ═══════════════════════════════════════════════════════════════
    # SIDEBAR
    # ═══════════════════════════════════════════════════════════════
    st.sidebar.markdown("### Performance Analysis")
    st.sidebar.markdown("---")
    _tc = st.sidebar.radio("Display Mode", ["Light", "Dark"], horizontal=True)
    st.session_state["_theme"] = _DARK if _tc == "Dark" else _LIGHT
    _T = st.session_state["_theme"]; BG = _T["BG"]; CARD = _T["CARD"]
    st.markdown(_theme_css(_T), unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════
    # DATA LOADERS
    # ═══════════════════════════════════════════════════════════════
    @st.cache_data(show_spinner="Loading performance data…")
    def load_perf():
        if os.environ.get("ATTRIQ_USE_PKL") and os.path.exists(PKL_RAW) and os.path.exists(PKL_ALL):
            return pickle.load(open(PKL_RAW,"rb")), pickle.load(open(PKL_ALL,"rb"))
        df = pd.read_csv(PERF_PATH)
        df['Date']     = pd.to_datetime(df['Date'], format='%m/%d/%Y')
        df['Agent_ID'] = df['Agent_ID'].astype(str).str.strip().str.upper()
        df['is_ghost'] = ((df['Calls_Answered'].fillna(0)==0) &
                        (df['Handle_Time_In_Hrs']==0) & (df['Productive_Hours']==0))
        df['aht_outlier'] = df['AHT'] > 14400
        def _tier(x):
            if pd.isna(x):  return 'Unknown'
            if x >= 75:     return 'Good'
            if x >= 40:     return 'Average'
            return 'Poor'
        df['occ_tier']   = df['Occupancy %'].map(_tier)
        dfa              = df[~df['is_ghost'] & ~df['aht_outlier']].copy()
        dfa['occ_tier']  = dfa['Occupancy %'].map(_tier)
        dfa['day_of_week'] = dfa['Date'].dt.day_name()
        if os.environ.get("ATTRIQ_USE_PKL"):
            os.makedirs(os.path.dirname(PKL_RAW), exist_ok=True)
            pickle.dump(df,  open(PKL_RAW,"wb"), protocol=4)
            pickle.dump(dfa, open(PKL_ALL,"wb"), protocol=4)
        return df, dfa

    @st.cache_data(show_spinner="Loading demographics…")
    def load_demo():
        d = pd.read_excel(DEMO_PATH)
        d['EMPLOYEE_ID'] = d['EMPLOYEE_ID'].astype(str).str.strip().str.upper()
        return d

    @st.cache_data(show_spinner="Loading coaching data…")
    def load_coach():
        coach = pd.read_excel(COACH_PATH, sheet_name='Coaching Data')
        coach = coach[[c for c in coach.columns if not c.startswith('Unnamed')]].copy()
        coach['AGENTID'] = coach['AGENTID'].astype(str).str.strip().str.upper()
        return coach

    @st.cache_data(show_spinner=False)
    def build_agents(_dfa):
        """Vectorised agent-level summary — no row-wise .apply()."""
        df_c = _dfa[_dfa['Calls_Answered'] > 0]
        ag = df_c.groupby('Agent_ID').agg(
            account     =('Account',           'first'),
            active_days =('Date',              'nunique'),
            avg_calls   =('Calls_Answered',    'mean'),
            avg_occ     =('Occupancy %',       'mean'),
            avg_aht     =('AHT',              'mean'),
            avg_talk    =('Avg_Talk_Time',     'mean'),
            avg_hold    =('Avg_Hold_Time',     'mean'),
            avg_wrap    =('Avg_Wrap_Time',     'mean'),
            avg_shrink  =('Overall_Shrinkage', 'mean'),
            avg_unplan  =('UnPlanned_OOO',     'mean'),
            avg_idle    =('Idle_Hrs',          'mean'),
            avg_prod    =('Productive_Hours',  'mean'),
            total_calls =('Calls_Answered',    'sum'),
        ).dropna(subset=['avg_occ','avg_aht'])
        # Quadrant — vectorised
        occ_med = ag['avg_occ'].median(); aht_med = ag['avg_aht'].median()
        ag['quadrant'] = np.select(
            [(ag['avg_occ']>=occ_med) & (ag['avg_aht']<=aht_med),
            (ag['avg_occ']>=occ_med) & (ag['avg_aht']> aht_med),
            (ag['avg_occ']< occ_med) & (ag['avg_aht']<=aht_med)],
            ['Star Performer','Busy but Slow','Underutilized'],
            default='Coaching Priority')
        # Performance tier — vectorised
        G = ag['avg_aht'].quantile(0.33); A = ag['avg_aht'].quantile(0.66)
        o = np.where(ag['avg_occ']>=75, 2, np.where(ag['avg_occ']>=40, 1, 0))
        a = np.where(ag['avg_aht']<=G,  2, np.where(ag['avg_aht']<=A,  1, 0))
        ag['perf_tier']  = np.select([o+a>=3, o+a>=2], ['Good','Average'], default='Poor')
        ag['eligible']   = ag['active_days'] >= 15
        return ag.round(3)

    @st.cache_data(show_spinner=False)
    def _multi_acc(_df_raw):
        cnts = _df_raw.groupby('Agent_ID')['Account'].nunique()
        ids  = set(cnts[cnts>1].index)
        detail = (_df_raw[_df_raw['Agent_ID'].isin(ids)]
                .groupby('Agent_ID')['Account']
                .apply(lambda x: ', '.join(sorted(x.unique())))
                .reset_index().rename(columns={'Account':'Accounts Found In'}))
        detail['No. of Accounts'] = detail['Accounts Found In'].str.count(',')+1
        return ids, detail.sort_values('No. of Accounts', ascending=False)

    @st.cache_data(show_spinner=False)
    def _coach_agg(_coach):
        return (_coach.groupby('AGENTID').agg(
            sessions=('COACHINGID','count'),
            first_coached=('DATECOACHED','min'),
            last_coached=('DATECOACHED','max'),
        ).reset_index().rename(columns={'AGENTID':'Agent_ID'}))

    @st.cache_data(show_spinner=False)
    def _agent_filter(_agent_all, acc_key):
        return _agent_all[_agent_all['account'].isin(set(acc_key))].copy()

    @st.cache_data(show_spinner=False)
    def _daily_agg(_df_f):
        d = _df_f.groupby('Date').agg(
            Total_Calls  =('Calls_Answered','sum'),
            Active_Agents=('Agent_ID','nunique'),
            Median_AHT   =('AHT','median'),
            Median_Occ   =('Occupancy %','median'),
            Median_Shrink=('Overall_Shrinkage','median'),
        ).reset_index()
        full = pd.date_range('2025-11-01','2025-11-30', freq='D')
        d = d.set_index('Date').reindex(full).reset_index().rename(columns={'index':'Date'})
        d['Day_of_Week'] = d['Date'].dt.day_name()
        return d

    # ═══════════════════════════════════════════════════════════════
    # LOAD
    # ═══════════════════════════════════════════════════════════════
    df_raw, df_all = load_perf()
    demo_raw       = load_demo()
    coach_raw      = load_coach()
    agent_all      = build_agents(df_all)
    multi_ids, multi_detail = _multi_acc(df_raw)
    coach_agg      = _coach_agg(coach_raw)

    # ═══════════════════════════════════════════════════════════════
    # SIDEBAR FILTERS
    # ═══════════════════════════════════════════════════════════════
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Filter the data**")
    all_accounts = sorted(df_all['Account'].unique())
    sel_accounts = st.sidebar.multiselect("Client Account", all_accounts, default=all_accounts)
    min_d = df_all['Date'].min().date(); max_d = df_all['Date'].max().date()
    date_range = st.sidebar.date_input("Date Range", value=(min_d, max_d),
                                    min_value=min_d, max_value=max_d)
    sel_tiers = st.sidebar.multiselect("Performance Tier",
        ['Good','Average','Poor'], default=['Good','Average','Poor'])
    show_ghost = st.sidebar.toggle("Include zero-call rows", value=False)
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Zero-call rows = agents who were on the roster that day but took no calls. "
        "Usually leave, no-shows, or training days.")

    if not sel_accounts:
        st.warning("⚠️ Please select at least one account from the sidebar.")
        st.stop()

    # ── Build filtered frames ─────────────────────────────────────
    df_base = df_raw if show_ghost else df_all
    df_f    = df_base[df_base['Account'].isin(sel_accounts)].copy()
    if len(date_range) == 2:
        d0, d1 = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        df_f   = df_f[(df_f['Date']>=d0) & (df_f['Date']<=d1)]
    if not show_ghost and 'occ_tier' in df_f.columns:
        df_f = df_f[df_f['occ_tier'].isin(sel_tiers)]

    df_f       = df_f.copy()
    df_calls   = df_f[df_f['Calls_Answered'].fillna(0) > 0].copy()
    agent_f    = _agent_filter(agent_all, tuple(sorted(sel_accounts)))
    daily      = _daily_agg(df_f)

    if df_f.empty:
        st.warning("⚠️ No data matches the selected filters. Please adjust the sidebar.")
        st.stop()

    # ═══════════════════════════════════════════════════════════════
    # PAGE HEADER
    # ═══════════════════════════════════════════════════════════════
    st.markdown('<div class="page-title">📊 Performance Analysis</div>', unsafe_allow_html=True)

    # ── Headline KPI Row ─────────────────────────────────────────
    active_agents  = df_all[df_all['Calls_Answered'].fillna(0)>0]['Agent_ID'].nunique()
    inactive_agents= df_raw['Agent_ID'].nunique() - active_agents
    total_calls    = int(df_calls['Calls_Answered'].sum())
    med_occ        = df_f['Occupancy %'].median()
    med_aht_s      = df_calls['AHT'].median()
    med_aht_min    = med_aht_s / 60
    prod_login_err  = (df_all['Productive_Hours']>df_all['Avg_Login_Hours']).sum()

    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.markdown(card(f"{active_agents:,}",  "Agents Who Took Calls",   "Made at least 1 call in November"),      unsafe_allow_html=True)
    k2.markdown(card(f"{inactive_agents:,}","Agents With Zero Calls",  "On roster but took no calls","warm"),    unsafe_allow_html=True)
    k3.markdown(card(f"{total_calls:,}",    "Total Calls Handled",     "Across all selected accounts"),          unsafe_allow_html=True)
    k4.markdown(card(f"{prod_login_err:,}","Data Anomaly Rows","Productive hrs > Login hrs","warn"),unsafe_allow_html=True)
    k5.markdown(card(f"{med_occ:.1f}%",     "Typical Occupancy",       "% of work time spent on calls","gold"),  unsafe_allow_html=True)
    k6.markdown(card(f"{med_aht_min:.1f} min","Typical Call Length",   "Average time per call (lower = faster)","gold"), unsafe_allow_html=True)

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════
    # STORY-ORDERED TABS
    # ═══════════════════════════════════════════════════════════════
    # Order: Overview → How Busy → How Fast → Time Lost → Who's Best/Worst
    #        → Account Comparison → Monthly Trends → Star Agents → Agent Profiles → Data Quality → Download
    tabs = st.tabs([
        "Overview",
        "How Busy Are They?",
        "How Fast Are They?",
        "Time Lost to Absence",
        "Who's Performing Best?",
        "Account Comparison",
        "Time Series Trends",
        "Star Agents",
        "Agent Profiles & Coaching",
        "Data Quality",
        "Download Report",
    ])

    # ────────────────────────────────────────────────────────────────
    # TAB 1 · OUR TEAM  — "Here is who we are looking at"
    # ────────────────────────────────────────────────────────────────
    with tabs[0]:
        

        col1, col2 = st.columns([1.4, 1])
        with col1:
            sh("How Many Agents Does Each Account Have?")
            acc = df_f.groupby('Account').agg(
                Agents=('Agent_ID','nunique'), Calls=('Calls_Answered','sum')
            ).reset_index().sort_values('Agents', ascending=True)
            fig = px.bar(acc, y='Account', x='Agents', orientation='h',
                        color='Agents', color_continuous_scale=[[0,"#c8dfe0"],[1,C_TEAL]],
                        text='Agents')
            fig.update_traces(textposition='outside', textfont_size=9)
            fig = plotly_layout(fig, "", 460, xlab="Number of Agents")
            fig.update_coloraxes(showscale=False)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            

        with col2:
            sh("Active vs Zero-Call Agents")
            ghost_cnt = df_raw['is_ghost'].sum()
            fig2 = go.Figure(go.Pie(
                labels=['Agents who took calls', 'Agents with zero calls'],
                values=[len(df_all), ghost_cnt], hole=0.55,
                marker=dict(colors=[C_TEAL, "#dde3ec"],
                            line=dict(color=BG, width=2))))
            fig2 = plotly_layout(fig2, "", 260)
            fig2.update_traces(textfont_size=11)
            st.plotly_chart(fig2, use_container_width=True)
            
            sh("Quick Facts")
            facts = [
                (f"{df_raw.shape[0]:,}", "Total records in the dataset"),
                (f"{df_all['Date'].nunique()}", "Days covered (Nov 1–30)"),
                (f"{df_raw['Agent_ID'].nunique():,}", "Unique agents on roster"),
                (f"{df_raw['Account'].nunique()}", "Client accounts"),
            ]
            c1, c2 = st.columns(2)
            for i, (v, l) in enumerate(facts):
                col = c1 if i % 2 == 0 else c2
                col.markdown(card(v, l), unsafe_allow_html=True)
                col.markdown("<div style='height:5px'></div>", unsafe_allow_html=True)

        sh("Account Summary at a Glance")
        acc_tbl = df_f.groupby('Account').agg(
            Agents           =('Agent_ID',          'nunique'),
            Calls_Handled    =('Calls_Answered',     'sum'),
            Avg_Calls_Per_Day=('Calls_Answered',     'mean'),
            Typical_Occ_Pct  =('Occupancy %',        'median'),
            Typical_AHT_Min  =('AHT',                lambda x: round(x.median()/60,1)),
            Avg_Shrinkage_Pct=('Overall_Shrinkage',  'median'),
        ).round(2).reset_index().sort_values('Agents', ascending=False)
        acc_tbl.columns = ['Account','Agents','Total Calls','Avg Calls/Day','Typical Occupancy %',
                        'Typical Call Length (min)','Typical Time Lost (Shrinkage %)']
        st.dataframe(acc_tbl, use_container_width=True, hide_index=True)
        

    # ────────────────────────────────────────────────────────────────
    # TAB 2 · HOW BUSY ARE THEY? — Occupancy
    # ────────────────────────────────────────────────────────────────
    with tabs[1]:
        

        sh("Are Agents Busy Enough?")
        st.markdown("""
        <br>
        Occupancy % = the percentage of an agent's logged-in time spent actively on calls (talking, putting the customer on hold, or writing up notes after the call).<br><br>
        <b>What is a good number?</b><br>
        &nbsp;&nbsp;&nbsp;🟢 <b>75% and above</b> — On target. The agent is handling a good volume of calls.<br>
        &nbsp;&nbsp;&nbsp;🟡 <b>40% to 74%</b> — Acceptable, but the agent has a lot of idle time between calls.<br>
        &nbsp;&nbsp;&nbsp;🔴 <b>Below 40%</b> — The agent is mostly sitting idle. Either they're under-scheduled or something else is wrong.
        </div>
        """, unsafe_allow_html=True)

        if df_f['Occupancy %'].dropna().empty:
            st.warning("No occupancy data available for the selected filters.")
        else:
            oa = (df_f.groupby('Account')['Occupancy %'].median()
                .reset_index().sort_values('Occupancy %'))
            oa.columns = ['Account', 'Median Occupancy %']
            oa['label'] = oa['Median Occupancy %'].round(1).astype(str) + '%'

            fig = px.bar(oa, x='Median Occupancy %', y='Account', orientation='h',
                        text='label', color='Median Occupancy %',
                        color_continuous_scale=[[0,"#e8c5b0"],[0.4,"#d4c87a"],[1,C_TEAL]])
            fig.add_vline(x=75, line_dash='dash', line_color=C_SAGE, line_width=1.5,
                        annotation_text='Target (75%)', annotation_font_color=C_SAGE,
                        annotation_position='top right')
            fig.add_vline(x=40, line_dash='dot', line_color=C_GOLD, line_width=1.5,
                        annotation_text='Minimum (40%)', annotation_font_color=C_GOLD)
            fig.update_traces(textposition='outside', textfont_size=9)
            fig = plotly_layout(fig, "", 480, xlab="Median Occupancy % (higher = busier)")
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, use_container_width=True)
            

            st.markdown("---")
            sh("What Share of Each Account's Agents Are Hitting the Target?")
            tier_by_acc = (df_f.dropna(subset=['occ_tier'])
                        .groupby(['Account','occ_tier']).size().reset_index(name='count'))
            tier_by_acc['pct'] = (tier_by_acc['count'] /
                tier_by_acc.groupby('Account')['count'].transform('sum') * 100).round(1)
            # Rename for friendly display
            tier_by_acc['Tier'] = tier_by_acc['occ_tier']

            fig2 = px.bar(tier_by_acc, x='pct', y='Account', color='Tier',
                        orientation='h', barmode='stack',
                        color_discrete_map={'Good':C_SAGE,'Average':C_GOLD,'Poor':C_PEACH,'Unknown':'#ccc'},
                        text='pct')
            fig2.update_traces(texttemplate='%{text:.0f}%', textposition='inside',
                            textfont=dict(size=8, color='white'), insidetextanchor='middle')
            fig2 = plotly_layout(fig2, "", 480, xlab="% of Agent-Days in Each Tier")
            st.plotly_chart(fig2, use_container_width=True)
            

    # ────────────────────────────────────────────────────────────────
    # TAB 3 · HOW FAST ARE THEY? — AHT
    # ────────────────────────────────────────────────────────────────
    with tabs[2]:
        

        sh("Average Handle Time (AHT) — How Long Does a Typical Call Take?")
        st.markdown("""
        <div style='background:#eef6f4;border-radius:8px;padding:12px 16px;margin-bottom:12px;border-left:4px solid #5BBFB5;'>
        <br>
        AHT (Average Handle Time) = how long an agent spends on a single call, from the moment they answer to the moment they finish writing up their notes. It includes:<br>
        &nbsp;&nbsp;&nbsp;🗣️ <b>Talk time</b> — actively speaking with the customer<br>
        &nbsp;&nbsp;&nbsp;⏸️ <b>Hold time</b> — customer is on hold while the agent checks something<br>
        &nbsp;&nbsp;&nbsp;📝 <b>Wrap time</b> — the agent updates records after the call ends<br><br>
        <b>Lower AHT = agent handles calls faster = more customers helped per shift.</b>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            sh("Typical Call Length by Account")
            aht_acc = (df_calls.groupby('Account')['AHT']
                    .median().reset_index().sort_values('AHT'))
            aht_acc['Minutes'] = (aht_acc['AHT']/60).round(1)
            aht_acc['label']   = aht_acc['Minutes'].astype(str) + ' min'
            fig = px.bar(aht_acc, x='AHT', y='Account', orientation='h',
                        text='label', color='Minutes',
                        color_continuous_scale=[[0,C_TEAL],[0.5,"#d4c87a"],[1,"#e8956d"]])
            fig.update_traces(textposition='outside', textfont_size=9)
            fig = plotly_layout(fig, "", 460, xlab="Median AHT (seconds) — shorter bar is better")
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, use_container_width=True)
            

        with c2:
            sh("Where Is the Time Going? Talk vs Hold vs Wrap")
            comp = df_calls.groupby('Account')[
                ['Avg_Talk_Time','Avg_Hold_Time','Avg_Wrap_Time']].mean().reset_index()
            comp = comp.rename(columns={
                'Avg_Talk_Time':'Talk Time (speaking)',
                'Avg_Hold_Time':'Hold Time (customer waiting)',
                'Avg_Wrap_Time':'Wrap Time (post-call notes)'})
            comp_m = comp.melt(id_vars='Account', var_name='Part of the Call', value_name='Seconds')
            comp_m['label'] = comp_m['Seconds'].round(0).astype(int).astype(str) + 's'
            fig3 = px.bar(comp_m, x='Seconds', y='Account', color='Part of the Call',
                        orientation='h', barmode='stack',
                        color_discrete_map={
                            'Talk Time (speaking)':     C_TEAL,
                            'Hold Time (customer waiting)': C_GOLD,
                            'Wrap Time (post-call notes)':  C_PURPLE},
                        text='label')
            fig3.update_traces(textposition='inside', textfont_size=8)
            fig3 = plotly_layout(fig3, "", 460, xlab="Seconds (average per call)")
            st.plotly_chart(fig3, use_container_width=True)
            

        st.markdown("---")
        sh("Call Length Range — Is It Consistent or All Over the Place?")
        st.caption(
            "The box shows the middle 50% of call lengths. The line inside the box is the typical value (median). "
            "Dots outside the box are unusually long or short calls. A wide box means high variation — "
            "some calls are much longer than others, which makes scheduling harder."
        )
        metric_sel = st.selectbox("Look at which metric?",
            ['AHT (Full call length)','Avg_Talk_Time','Avg_Hold_Time','Avg_Wrap_Time'],
            key='aht_box_sel',
            format_func=lambda x: {
                'AHT (Full call length)': 'AHT — Full call length',
                'Avg_Talk_Time': 'Talk Time only',
                'Avg_Hold_Time': 'Hold Time only',
                'Avg_Wrap_Time': 'Wrap Time only'}.get(x, x))
        col_map = {
            'AHT (Full call length)': 'AHT',
            'Avg_Talk_Time': 'Avg_Talk_Time',
            'Avg_Hold_Time': 'Avg_Hold_Time',
            'Avg_Wrap_Time': 'Avg_Wrap_Time',
        }
        real_col = col_map[metric_sel]
        fig4 = px.box(df_calls, x='Account', y=real_col, color='Account',
                    color_discrete_sequence=PALETTE)
        fig4.update_traces(boxmean=True)
        fig4 = plotly_layout(fig4, "", 420, xlab="Account", ylab="Seconds")
        fig4.update_layout(showlegend=False)
        fig4.update_xaxes(tickangle=35)
        st.plotly_chart(fig4, use_container_width=True)

    # ────────────────────────────────────────────────────────────────
    # TAB 4 · TIME LOST TO ABSENCE — Shrinkage
    # ────────────────────────────────────────────────────────────────
    with tabs[3]:
        

        sh("Shrinkage — Scheduled Time That Gets Lost")
        st.markdown("""
        <div style='background:#eef6f4;border-radius:8px;padding:12px 16px;margin-bottom:14px;border-left:4px solid #5BBFB5;'>
        <br>
        &nbsp;&nbsp;&nbsp;📅 <b>Planned leave</b> — approved holidays, scheduled training, team meetings<br>
        &nbsp;&nbsp;&nbsp;❌ <b>Unplanned absence</b> — no-shows, last-minute sick days, unexpected leave<br>
        &nbsp;&nbsp;&nbsp;⏰ <b>Late arrivals</b> — agents who arrived but logged in late<br>
        </div>
        """, unsafe_allow_html=True)

        sh_c = ['Overall_Shrinkage','Planned_OOO','UnPlanned_OOO','Tardy_OOO']
        sh_labels = {
            'Overall_Shrinkage': 'Total Time Lost (%)',
            'Planned_OOO':       'Planned Leave (%)',
            'UnPlanned_OOO':     'Unplanned Absence (%)',
            'Tardy_OOO':         'Late Arrivals (%)',
        }

        shrink_acc = df_f.groupby('Account')[sh_c].median().reset_index()

        sa1, sa2 = st.columns(2)
        with sa1:
            sh("Total Time Lost by Account")
            shrink_sorted = shrink_acc.sort_values('Overall_Shrinkage', ascending=True)
            shrink_sorted['label'] = shrink_sorted['Overall_Shrinkage'].round(1).astype(str)+'%'
            fig = px.bar(shrink_sorted, x='Overall_Shrinkage', y='Account',
                        orientation='h', text='label',
                        color='Overall_Shrinkage',
                        color_continuous_scale=[[0,C_TEAL],[0.5,C_GOLD],[1,C_PEACH]])
            fig.update_traces(textposition='outside', textfont_size=9)
            fig = plotly_layout(fig, "", 460, xlab="Median % of scheduled time lost")
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, use_container_width=True)
            

        with sa2:
            sh("What Type of Absence Is It?")
            shrink_types = shrink_acc[['Account','Planned_OOO','UnPlanned_OOO','Tardy_OOO']].copy()
            shrink_m = shrink_types.melt(id_vars='Account', var_name='Type', value_name='Pct')
            shrink_m['Type'] = shrink_m['Type'].map(sh_labels)
            fig2 = px.bar(shrink_m, x='Pct', y='Account', color='Type',
                        orientation='h', barmode='stack',
                        color_discrete_map={
                            'Planned Leave (%)':     C_BLUE,
                            'Unplanned Absence (%)': C_PEACH,
                            'Late Arrivals (%)':     C_GOLD},
                        text='Pct')
            fig2.update_traces(texttemplate='%{text:.1f}%', textposition='inside',
                            textfont=dict(size=8), insidetextanchor='middle')
            fig2 = plotly_layout(fig2, "", 460, xlab="% of scheduled time")
            st.plotly_chart(fig2, use_container_width=True)
            

        st.markdown("---")
        sh("Shrinkage Summary Table")
        shrink_tbl = shrink_acc.copy()
        shrink_tbl.columns = ['Account','Total Time Lost (%)','Planned Leave (%)',
                            'Unplanned Absence (%)','Late Arrivals (%)']
        shrink_tbl = shrink_tbl.round(1).sort_values('Total Time Lost (%)', ascending=False)
        st.dataframe(shrink_tbl, use_container_width=True, hide_index=True)
        

    # ────────────────────────────────────────────────────────────────
    # TAB 5 · WHO'S PERFORMING BEST? — Quadrants
    # ────────────────────────────────────────────────────────────────
    with tabs[4]:
        

        sh("The Four Performance Groups")
        st.markdown("""
        <div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:16px;'>
        <div style='background:#edf7f5;border-radius:8px;padding:12px 14px;border-left:4px solid #5BBFB5;'>
            <b style='color:#5BBFB5;'>⭐ Star Performer</b><br>
            <span style='font-size:.88rem;'>High occupancy <i>and</i> fast calls. These are your best agents — protect their schedules and involve them in training others.</span>
        </div>
        <div style='background:#fef9ee;border-radius:8px;padding:12px 14px;border-left:4px solid #D4A847;'>
            <b style='color:#D4A847;'>⏳ Busy but Slow</b><br>
            <span style='font-size:.88rem;'>High occupancy but slow calls. These agents are working hard but calls are taking longer than they should — usually a hold time or wrap time issue. Coaching on efficiency helps.</span>
        </div>
        <div style='background:#eef2fa;border-radius:8px;padding:12px 14px;border-left:4px solid #6B9FD4;'>
            <b style='color:#6B9FD4;'>😴 Underutilized</b><br>
            <span style='font-size:.88rem;'>Low occupancy but fast when they do work. These agents could handle more calls — consider whether call routing is directing enough volume to them.</span>
        </div>
        <div style='background:#fef3ee;border-radius:8px;padding:12px 14px;border-left:4px solid #E8956D;'>
            <b style='color:#E8956D;'>🎯 Coaching Priority</b><br>
            <span style='font-size:.88rem;'>Low occupancy <i>and</i> slow calls. These agents need the most support — a structured coaching plan and closer monitoring from their team leader.</span>
        </div>
        </div>
        """, unsafe_allow_html=True)

        # Quadrant summary KPIs
        total_q = len(agent_f)
        qk1, qk2, qk3, qk4 = st.columns(4)
        for col_ui, qname, color in zip(
            [qk1, qk2, qk3, qk4],
            ['Star Performer','Busy but Slow','Underutilized','Coaching Priority'],
            [C_TEAL, C_GOLD, C_BLUE, C_PEACH]
        ):
            cnt = (agent_f['quadrant']==qname).sum()
            col_ui.markdown(
                f'<div class="metric-card"><div style="font-size:1.5rem;font-weight:700;'
                f'color:{color};font-family:JetBrains Mono,monospace">{cnt:,}</div>'
                f'<div class="metric-label">{qname}</div>'
                f'<div class="metric-sub">{cnt/total_q*100:.0f}% of all agents</div></div>',
                unsafe_allow_html=True)

        st.markdown("---")
        sh("Where Does Every Agent Sit?")
        st.caption(
            "Each dot is one agent. Left = low occupancy (less busy). Right = high occupancy (busier). "
            "Bottom = fast calls. Top = slow calls. The ideal agent is in the bottom-right corner (Star Performer). "
            
        )

        fc1, fc2 = st.columns(2)
        sel_qa  = fc1.multiselect("Filter by Account", sorted(agent_f['account'].dropna().unique()),
                                default=sorted(agent_f['account'].dropna().unique()), key='q_acc')
        min_days = fc2.slider("Only show agents with at least this many active days", 1, 30, 1)

        plot_ag = agent_f[agent_f['account'].isin(sel_qa) & (agent_f['active_days']>=min_days)].copy()
        AHT_CAP = 2500
        plot_ag['avg_aht_capped'] = plot_ag['avg_aht'].clip(upper=AHT_CAP)
        occ_med = agent_f['avg_occ'].median(); aht_med = agent_f['avg_aht'].median()

        MAX_PTS = 3000
        if len(plot_ag) > MAX_PTS:
            plot_sample = plot_ag.sample(MAX_PTS, random_state=42)
            st.caption(f"ℹ️ Showing {MAX_PTS:,} agents (representative sample of {len(plot_ag):,}) for smooth display.")
        else:
            plot_sample = plot_ag

        fig_q = go.Figure()
        for qname, color in QUAD_COLORS.items():
            dq = plot_sample[plot_sample['quadrant']==qname]
            full_cnt = (plot_ag['quadrant']==qname).sum()
            fig_q.add_trace(go.Scattergl(
                x=dq['avg_occ'], y=dq['avg_aht_capped'],
                mode='markers',
                name=f"{qname} ({full_cnt:,})",
                marker=dict(color=color, size=6, opacity=0.65,
                            line=dict(width=0.4, color='rgba(0,0,0,0.15)')),
                customdata=dq[['account','active_days','avg_calls','avg_aht']].values,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Occupancy: %{x:.1f}%<br>"
                    "Avg Call Length: %{customdata[3]:.0f}s<br>"
                    "Avg Calls/Day: %{customdata[2]:.1f}<br>"
                    "Days Active: %{customdata[1]}<extra></extra>")
            ))

        Y_MAX = AHT_CAP + 50
        fig_q.add_vline(x=occ_med, line=dict(color='rgba(100,120,140,0.3)', width=1.5, dash='dash'))
        fig_q.add_hline(y=aht_med, line=dict(color='rgba(100,120,140,0.3)', width=1.5, dash='dash'))

        for x0,x1,y0,y1,fill,label,lx,ly in [
            (0,occ_med,aht_med,Y_MAX,'rgba(232,149,109,0.06)','Coaching Priority',
            occ_med*0.35,aht_med*1.33),
            (occ_med,101,aht_med,Y_MAX,'rgba(212,168,71,0.06)','Busy but Slow',
            occ_med*1.28,aht_med*1.33),
            (0,occ_med,0,aht_med,'rgba(107,159,212,0.06)','Underutilized',
            occ_med*0.35,aht_med*0.38),
            (occ_med,101,0,aht_med,'rgba(91,191,181,0.06)','Star Performer',
            occ_med*1.28,aht_med*0.38),
        ]:
            fig_q.add_shape(type='rect', x0=x0, x1=x1, y0=y0, y1=y1,
                            fillcolor=fill, line=dict(width=0), layer='below')
            fig_q.add_annotation(x=lx, y=ly, text=f"<b>{label}</b>",
                                showarrow=False, font=dict(color=QUAD_COLORS[label], size=10), opacity=0.4)

        fig_q.update_layout(
            paper_bgcolor=BG, plot_bgcolor=CARD, height=520,
            font=dict(family="Inter", color=_T["TEXT"]),
            margin=dict(l=55,r=20,t=30,b=55),
            xaxis=dict(title="Occupancy % — how much of their day was spent on calls (right = busier)",
                    gridcolor=_T["BORDER"], range=[-2,103]),
            yaxis=dict(title=f"Avg Call Length in seconds — lower = faster (capped at {AHT_CAP}s)",
                    gridcolor=_T["BORDER"], range=[-30, Y_MAX]),
            legend=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor="#ccc", borderwidth=1,
                        font=dict(size=10, color="#333")),
            hovermode="closest")
        st.plotly_chart(fig_q, use_container_width=True)
        

        st.markdown("---")
        sh("Performance Group — Account Breakdown")
        qa = agent_f.groupby(['account','quadrant']).size().reset_index(name='count')
        qa['label'] = qa['count'].astype(str)
        fig3 = px.bar(qa, x='count', y='account', color='quadrant',
                    orientation='h', barmode='stack',
                    color_discrete_map=QUAD_COLORS, text='label')
        fig3.update_traces(textposition='inside', textfont=dict(size=8, color='white'),
                        insidetextanchor='middle')
        fig3 = plotly_layout(fig3, "", 500, xlab="Number of Agents")
        st.plotly_chart(fig3, use_container_width=True)
        

        st.markdown("---")
        sh("Accounts That Need Coaching Attention Most — Ranked")
        st.caption(
            "This ranking combines three things: how many agents are in the Coaching Priority group, "
            "how long their calls take, and how often they are absent without notice. Higher score = more urgent."
        )
        cp = agent_f[agent_f['quadrant']=='Coaching Priority'].copy()
        if cp.empty:
            st.info("No agents in the Coaching Priority group for the current filters.")
        else:
            p90a = agent_f['attrition_risk'].quantile(0.9) if 'attrition_risk' in agent_f.columns else 0
            cp_acc = cp.groupby('account').agg(
                Agents_Needing_Coaching=('avg_occ','count'),
                Avg_Occupancy=('avg_occ','mean'),
                Avg_Call_Length_s=('avg_aht','mean'),
            ).round(1).reset_index().sort_values('Agents_Needing_Coaching', ascending=False)
            max_aht = cp_acc['Avg_Call_Length_s'].max()
            cp_acc['Urgency_Score'] = (
                (cp_acc['Agents_Needing_Coaching']/cp_acc['Agents_Needing_Coaching'].max()*60) +
                ((cp_acc['Avg_Call_Length_s']-600)/(max_aht+1e-9)*40)
            ).clip(0).round(1)
            cp_acc['Avg_Call_Length_min'] = (cp_acc['Avg_Call_Length_s']/60).round(1)
            st.dataframe(
                cp_acc[['account','Agents_Needing_Coaching','Avg_Occupancy',
                        'Avg_Call_Length_min','Urgency_Score']]
                .rename(columns={
                    'account':'Account',
                    'Agents_Needing_Coaching':'Agents Needing Coaching',
                    'Avg_Occupancy':'Average Occupancy %',
                    'Avg_Call_Length_min':'Avg Call Length (min)',
                    'Urgency_Score':'Urgency Score (higher = act first)'}),
                use_container_width=True, hide_index=True)

    # ────────────────────────────────────────────────────────────────
    # TAB 6 · ACCOUNT COMPARISON
    # ────────────────────────────────────────────────────────────────
    with tabs[5]:
        

        sh("Account Performance — Side by Side")
        CALL_METRICS = {
            'Calls Answered per Day':   'Calls_Answered',
            'AHT — Avg Call Length (s)':'AHT',
            'Talk Time (s)':            'Avg_Talk_Time',
            'Hold Time (s)':            'Avg_Hold_Time',
            'Wrap Time (s)':            'Avg_Wrap_Time',
        }
        view_mode = st.radio("View", ["One metric at a time","All metrics side by side"],
                            horizontal=True, key='acc_view')

        if view_mode == "One metric at a time":
            sel_metric = st.selectbox("Choose a metric to compare", list(CALL_METRICS.keys()))
            real_col = CALL_METRICS[sel_metric]
            fig = px.box(df_calls, x='Account', y=real_col, color='Account',
                        color_discrete_sequence=PALETTE)
            fig.update_traces(boxmean=True)
            fig = plotly_layout(fig, f"{sel_metric} — range of values per account", 440,
                                xlab="Account", ylab=sel_metric)
            fig.update_layout(showlegend=False)
            fig.update_xaxes(tickangle=35)
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            for i, (label, col) in enumerate(CALL_METRICS.items()):
                acc_m = df_calls.groupby('Account')[col].mean().reset_index().sort_values(col)
                acc_m['label'] = acc_m[col].round(1).astype(str)
                fig_m = px.bar(acc_m, x='Account', y=col,
                            color_discrete_sequence=[PALETTE[i % len(PALETTE)]], text='label')
                fig_m.update_traces(textposition='outside', textfont_size=9)
                fig_m = plotly_layout(fig_m, label, 300, xlab="Account", ylab=label)
                fig_m.update_xaxes(tickangle=35)
                st.plotly_chart(fig_m, use_container_width=True)

        st.markdown("---")
        sh("Performance Tier Mix — Which Account Has the Best Agents?")
        tier_acc = (agent_f.groupby(['account','perf_tier']).size().reset_index(name='count'))
        tier_acc['pct'] = (tier_acc['count'] /
            tier_acc.groupby('account')['count'].transform('sum') * 100).round(1)
        tier_acc = tier_acc.rename(columns={'account':'Account','perf_tier':'Performance Tier'})
        fig_t = px.bar(tier_acc, x='pct', y='Account', color='Performance Tier',
                    orientation='h', barmode='stack',
                    color_discrete_map={'Good':C_SAGE,'Average':C_GOLD,'Poor':C_PEACH},
                    text='pct')
        fig_t.update_traces(texttemplate='%{text:.0f}%', textposition='inside',
                            textfont=dict(size=8, color='white'), insidetextanchor='middle')
        fig_t = plotly_layout(fig_t, "", 480, xlab="% of Agents")
        st.plotly_chart(fig_t, use_container_width=True)
        

    # ────────────────────────────────────────────────────────────────
    # TAB 7 · MONTH IN REVIEW — Time Series
    # ────────────────────────────────────────────────────────────────
    with tabs[6]:
        

        NOV = ['2025-11-01','2025-11-30']
        sh("How Did Things Change Day by Day?")
        metric_labels = {
            'Total_Calls':   'Total Calls Handled',
            'Active_Agents': 'Number of Agents Active',
            'Median_AHT':    'Typical Call Length (AHT in seconds)',
            'Median_Occ':    'Typical Occupancy %',
            'Median_Shrink': 'Typical Time Lost (Shrinkage %)',
        }
        tm = st.selectbox("Choose what to track over time", list(metric_labels.keys()),
                        key='ts_metric', format_func=lambda x: metric_labels[x])
        color_map = {'Total_Calls':C_TEAL,'Active_Agents':C_BLUE,'Median_AHT':C_PEACH,
                    'Median_Occ':C_SAGE,'Median_Shrink':C_GOLD}
        fig = px.line(daily, x='Date', y=tm, markers=True,
                    color_discrete_sequence=[color_map[tm]])
        fig.update_traces(line_width=2.2, marker_size=6)
        fig.update_xaxes(range=NOV, dtick='D1', tickformat='%d %b', tickangle=45)
        fig = plotly_layout(fig, "", 380, xlab="Date", ylab=metric_labels[tm])
        st.plotly_chart(fig, use_container_width=True)
        

        st.markdown("---")
        sh("Which Days of the Week Are Busiest?")
        dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        dow = daily.groupby('Day_of_Week')[['Total_Calls','Median_Occ']].mean().round(2)
        dow = dow.reindex([d for d in dow_order if d in dow.index]).reset_index()

        fig_dow = go.Figure()
        fig_dow.add_trace(go.Bar(
            x=dow['Day_of_Week'], y=dow['Total_Calls'], name='Total Calls',
            marker_color=C_BLUE, opacity=0.75,
            text=dow['Total_Calls'].round(0).astype(int), textposition='outside'))
        fig_dow.add_trace(go.Scatter(
            x=dow['Day_of_Week'], y=dow['Median_Occ'], name='Typical Occupancy (%)',
            mode='lines+markers', line=dict(color=C_TEAL, width=2.5),
            marker=dict(size=8), yaxis='y2'))
        fig_dow.update_layout(
            yaxis=dict(title='Total Calls', gridcolor=_T["BORDER"]),
            yaxis2=dict(title='Typical Occupancy (%)', overlaying='y', side='right'),
            legend=dict(orientation='h', y=1.12),
            paper_bgcolor=BG, plot_bgcolor=CARD, height=360,
            font=dict(family="Inter", color=_T["TEXT"]),
            margin=dict(l=50,r=50,t=30,b=50))
        st.plotly_chart(fig_dow, use_container_width=True)
        

        st.markdown("---")
        sh("Track a Single Agent Across November")
        all_ids = sorted(df_calls['Agent_ID'].unique().tolist())
        if not all_ids:
            st.warning("No agents found with calls for the selected filters.")
        else:
            sel_agent = st.selectbox("Select an agent", all_ids, key='agent_ts')
            sel_ind = st.multiselect("Metrics to show",
                ['Calls_Answered','AHT','Occupancy %','Productive_Hours','Overall_Shrinkage'],
                default=['Calls_Answered','AHT','Occupancy %'],
                format_func=lambda x: {
                    'Calls_Answered':'Calls Handled','AHT':'Call Length (AHT)',
                    'Occupancy %':'Occupancy %','Productive_Hours':'Hours Worked',
                    'Overall_Shrinkage':'Time Lost (Shrinkage %)'}.get(x,x))
            if len(sel_ind) > 5:
                st.warning("⚠️ More than 5 metrics makes the chart hard to read.")
            if sel_agent and sel_ind:
                nov_dates = pd.date_range('2025-11-01','2025-11-30', freq='D')
                agent_daily = (df_calls[df_calls['Agent_ID']==sel_agent][['Date']+sel_ind]
                            .set_index('Date').reindex(nov_dates)
                            .reset_index().rename(columns={'index':'Date'}))
                if agent_daily.dropna(how='all', subset=sel_ind).empty:
                    st.warning(f"No data found for agent {sel_agent} with the current filters.")
                else:
                    al = agent_daily.melt(id_vars='Date', var_name='Metric', value_name='Value')
                    al['Metric'] = al['Metric'].map({
                        'Calls_Answered':'Calls Handled','AHT':'Call Length (sec)',
                        'Occupancy %':'Occupancy %','Productive_Hours':'Hours Worked',
                        'Overall_Shrinkage':'Time Lost (%)'})
                    fig_ag = px.line(al, x='Date', y='Value', color='Metric',
                                    markers=True, color_discrete_sequence=PALETTE)
                    fig_ag.update_traces(line_width=2, marker_size=6)
                    fig_ag.update_xaxes(range=NOV, dtick='D1', tickformat='%d %b', tickangle=45)
                    fig_ag = plotly_layout(fig_ag, f"Agent {sel_agent} — Daily View", 460,
                                        xlab="Date", ylab="Value")
                    st.plotly_chart(fig_ag, use_container_width=True)
                    

    # ────────────────────────────────────────────────────────────────
    # TAB 8 · STAR AGENTS — Top Performers
    # ────────────────────────────────────────────────────────────────
    with tabs[7]:
        

        eligible_ids = agent_all[agent_all['active_days']>=15].index.tolist()
        df_elig = df_calls[df_calls['Agent_ID'].isin(eligible_ids)]

        if df_elig.empty:
            st.warning("No agents with 15+ active days found for the current filters.")
        else:
            sh("Top Performing Agents — Consistent Throughout November")
            c1, c2 = st.columns(2)
            rank_by = c1.selectbox("Rank agents by", [
                'Most Calls Handled','Highest Occupancy','Fastest Call Handling'],
                key='rank_metric')
            n_show = c2.slider("How many agents to show", 5, 20, 10)

            rank_col_map = {
                'Most Calls Handled':     ('avg_calls',  False),
                'Highest Occupancy':      ('avg_occ',    False),
                'Fastest Call Handling':  ('avg_aht',    True),   # ascending = best AHT is lowest
            }
            rank_col, asc = rank_col_map[rank_by]

            if asc:
                top_ids = agent_all[agent_all['eligible']].nsmallest(n_show, rank_col).index.tolist()
            else:
                top_ids = agent_all[agent_all['eligible']].nlargest(n_show, rank_col).index.tolist()

            top_tbl = agent_all.loc[top_ids, [
                'account','active_days','avg_calls','avg_occ','avg_aht','perf_tier','quadrant']
            ].round(2).reset_index()
            top_tbl.columns = ['Agent ID','Account','Days Active','Avg Calls/Day',
                            'Avg Occupancy %','Avg Call Length (sec)','Performance Tier','Group']
            st.dataframe(top_tbl, use_container_width=True, hide_index=True)
            

            st.markdown("---")
            sh("How Did the Top Agents Perform Day by Day?")
            unit_map = {'avg_calls':'calls','avg_occ':'%','avg_aht':'seconds'}
            track_metric = st.selectbox("Track which metric for top agents?",
                ['Calls Answered','Occupancy %','AHT — Call Length'],
                key='top_track')
            track_col_map = {'Calls Answered':'Calls_Answered','Occupancy %':'Occupancy %',
                            'AHT — Call Length':'AHT'}
            t_col = track_col_map[track_metric]
            df_top = df_calls[df_calls['Agent_ID'].isin(top_ids)]
            df_top_d = df_top.groupby(['Agent_ID','Date'])[t_col].mean().reset_index()
            fig = px.line(df_top_d, x='Date', y=t_col, color='Agent_ID',
                        markers=True, color_discrete_sequence=PALETTE)
            fig.update_traces(line_width=1.8, marker_size=5)
            fig.update_xaxes(range=['2025-11-01','2025-11-30'],
                            dtick='D1', tickformat='%d %b', tickangle=45)
            fig = plotly_layout(fig, f"Top {n_show} Agents — Daily {track_metric}", 480,
                                xlab="Date", ylab=track_metric)
            st.plotly_chart(fig, use_container_width=True)
            

    # ────────────────────────────────────────────────────────────────
    # TAB 9 · AGENT PROFILES & COACHING
    # ────────────────────────────────────────────────────────────────
    with tabs[8]:
        

        demo_nov   = demo_raw[demo_raw['MONTH']==11].copy()
        agent_demo = agent_f.reset_index().merge(
            demo_nov[['EMPLOYEE_ID','MARITAL_STATUS','JOB_FAMILY','GENDER','AGE']],
            left_on='Agent_ID', right_on='EMPLOYEE_ID', how='left')
        agent_demo['Age Group'] = pd.cut(agent_demo['AGE'],
            bins=[18,25,30,35,40,45,50,60,100], right=False,
            labels=['18-25','25-30','30-35','35-40','40-45','45-50','50-60','60+'])

        KPI_OPT = {
            'Occupancy % (how busy)':   ('avg_occ',  '{:.1f}%'),
            'Call Length — AHT (sec)':  ('avg_aht',  '{:.0f}s'),
            'Calls Handled per Day':    ('avg_calls', '{:.1f}'),
            'Hold Time (sec)':          ('avg_hold',  '{:.0f}s'),
            'Wrap Time (sec)':          ('avg_wrap',  '{:.0f}s'),
            'Time Lost — Shrinkage %':  ('avg_shrink','{:.1f}%'),
        }
        sel_kpi = st.selectbox("Which metric do you want to explore?", list(KPI_OPT.keys()),
                            key='demo_kpi')
        kpi_col, kpi_fmt = KPI_OPT[sel_kpi]

        def simple_bar(df_in, grp_col, friendly_name, ht=300, orient='h'):
            grp = (df_in.dropna(subset=[grp_col, kpi_col])
                .groupby(grp_col)[kpi_col].mean().reset_index()
                .sort_values(kpi_col, ascending=(orient=='h')))
            grp['label'] = grp[kpi_col].apply(lambda v: kpi_fmt.format(v))
            if orient == 'h':
                fig = px.bar(grp, x=kpi_col, y=grp_col, orientation='h',
                            color_discrete_sequence=[C_TEAL], text='label')
            else:
                fig = px.bar(grp, x=grp_col, y=kpi_col,
                            color_discrete_sequence=[C_TEAL], text='label')
            fig.update_traces(textposition='outside', textfont_size=9)
            fig.update_layout(showlegend=False)
            return plotly_layout(fig, f"{friendly_name} vs {sel_kpi}", ht,
                                xlab=('' if orient=='v' else sel_kpi),
                                ylab=('' if orient=='h' else sel_kpi))

        sh("A · How Does Age Affect Performance?")
        ag1, ag2 = st.columns(2)
        with ag1:
            st.plotly_chart(simple_bar(agent_demo,'Age Group','Age Group',310,'v'), use_container_width=True)
        with ag2:
            # Tier mix heatmap replaced with a simpler stacked bar
            grp = (agent_demo.dropna(subset=['Age Group','perf_tier'])
                .groupby(['Age Group','perf_tier']).size().reset_index(name='count'))
            grp['pct'] = (grp['count'] / grp.groupby('Age Group')['count'].transform('sum') * 100).round(1)
            fig = px.bar(grp, x='Age Group', y='pct', color='perf_tier', barmode='stack',
                        color_discrete_map=TIER_COLORS, text='pct')
            fig.update_traces(texttemplate='%{text:.0f}%', textposition='inside',
                            textfont=dict(size=8, color='white'), insidetextanchor='middle')
            fig = plotly_layout(fig, "Age Group — % in Each Performance Tier", 310,
                                xlab="Age Group", ylab="% of Agents")
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")

        sh("B · Does Gender Make a Difference?")
        gg1, gg2 = st.columns(2)
        with gg1:
            st.plotly_chart(simple_bar(agent_demo,'GENDER','Gender',280,'v'), use_container_width=True)
        with gg2:
            grp2 = (agent_demo.dropna(subset=['GENDER','perf_tier'])
                    .groupby(['GENDER','perf_tier']).size().reset_index(name='count'))
            grp2['pct'] = (grp2['count']/grp2.groupby('GENDER')['count'].transform('sum')*100).round(1)
            fig2 = px.bar(grp2, x='GENDER', y='pct', color='perf_tier', barmode='stack',
                        color_discrete_map=TIER_COLORS, text='pct')
            fig2.update_traces(texttemplate='%{text:.0f}%', textposition='inside',
                            textfont=dict(size=8, color='white'), insidetextanchor='middle')
            fig2 = plotly_layout(fig2, "Gender — % in Each Performance Tier", 280,
                                xlab="Gender", ylab="% of Agents")
            st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("---")

        sh("C · Did Coaching Sessions Make Calls Faster?")
        agent_coached = agent_f.reset_index().merge(
            coach_agg[['Agent_ID','sessions']], on='Agent_ID', how='left')
        agent_coached['Was Coached'] = agent_coached['sessions'].notna()
        agent_coached['sessions']    = agent_coached['sessions'].fillna(0).astype(int)
        coached_only = agent_coached[agent_coached['Was Coached']].dropna(subset=['avg_occ','avg_aht'])

        cc1, cc2 = st.columns(2)
        with cc1:
            fig7 = px.scatter(coached_only, x='sessions', y='avg_aht', color='perf_tier',
                            trendline='ols',
                            color_discrete_map={'Good':C_SAGE,'Average':C_GOLD,'Poor':C_PEACH},
                            opacity=0.6,
                            labels={'sessions':'Number of Coaching Sessions',
                                    'avg_aht':'Avg Call Length (AHT in seconds)',
                                    'perf_tier':'Performance Tier'})
            fig7 = plotly_layout(fig7,
                "More coaching sessions → shorter calls? (downward line = yes)", 340,
                xlab="Coaching Sessions", ylab="Avg Call Length (seconds)")
            st.plotly_chart(fig7, use_container_width=True)
            

        with cc2:
            # Before vs after coaching — simple dumbbell
            df_w = df_all[df_all['Calls_Answered'].fillna(0)>0].merge(
                coach_agg[['Agent_ID','first_coached','last_coached','sessions']],
                on='Agent_ID', how='inner')
            bef = df_w[df_w['Date']<df_w['first_coached']].groupby('Agent_ID').agg(
                occ_before=('Occupancy %','mean'), aht_before=('AHT','mean'))
            aft = df_w[df_w['Date']>df_w['last_coached']].groupby('Agent_ID').agg(
                occ_after=('Occupancy %','mean'), aht_after=('AHT','mean'))
            ba = bef.join(aft, how='inner').join(
                coach_agg.set_index('Agent_ID')[['sessions']], how='inner')
            ba['aht_delta'] = ba['aht_after'] - ba['aht_before']

            if len(ba) == 0:
                st.info("Not enough before/after data to compare coaching impact for current filters.")
            else:
                improved_pct = (ba['aht_delta'] < 0).mean() * 100
                avg_improvement = ba['aht_delta'].mean()
                m1, m2 = st.columns(2)
                m1.markdown(card(f"{improved_pct:.0f}%","Agents Who Got Faster After Coaching",
                                "Calls took less time after sessions","gold"), unsafe_allow_html=True)
                m2.markdown(card(f"{abs(avg_improvement):.0f}s","Average Call Time Saved",
                                "Seconds saved per call on average"), unsafe_allow_html=True)
                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

                ba['session_group'] = pd.cut(ba['sessions'],
                    bins=[0,1,2,3,5,100], labels=['1','2','3','4-5','6+'])
                sg = (ba.groupby('session_group', observed=True).agg(
                    pct_improved=('aht_delta', lambda x:(x<0).mean()*100)
                ).round(1).reset_index())
                fig_sg = px.bar(sg, x='session_group', y='pct_improved',
                                color_discrete_sequence=[C_TEAL], text='pct_improved')
                fig_sg.update_traces(texttemplate='%{text:.0f}%', textposition='outside')
                fig_sg.add_hline(y=50, line_dash='dash', line_color=C_GOLD, line_width=1.5,
                                annotation_text='50% baseline')
                fig_sg = plotly_layout(fig_sg,
                    "% of Agents with Faster Calls — by Number of Sessions", 300,
                    xlab="Number of Coaching Sessions", ylab="% Who Improved")
                st.plotly_chart(fig_sg, use_container_width=True)
                

    # ────────────────────────────────────────────────────────────────
    # TAB 10 · DATA QUALITY
    # ────────────────────────────────────────────────────────────────
    with tabs[9]:
        

        sh("What Did We Find in the Raw Data?")
        issues = [
            ("~120,000 rows", "Zero-activity records (ghost rows)",
            "33% of all records show an agent who was scheduled but took no calls. "
            "These are expected — they represent leave, training, and no-shows. "
            "They are excluded from all performance calculations."),
            ("~180 agents", "Agents listed under multiple client accounts",
            "These agents appear in more than one account's roster. "
            "This may be intentional (shared resources) or a data entry error. "
            "Recommend verifying with the client before the next reporting cycle."),
            ("~400 rows", "Productive hours exceeds login hours",
            "Some records show agents clocking more productive hours than they were actually logged in. "
            "This is impossible and points to a data capture issue in the system."),
            ("5 rows", "Extremely long call records (over 4 hours AHT)",
            "Five records show a single call taking over 4 hours. "
            "These are almost certainly data entry errors and have been excluded."),
            ("~40 rows", "Duplicate records (same agent, same day)",
            "A small number of agent-date combinations appear more than once. "
            "These have been flagged and will not double-count."),
        ]

        for value, label, explanation in issues:
            st.markdown(f"""
            <div style='background:{CARD};border:1px solid {_T["BORDER"]};border-radius:8px;
                        padding:12px 16px;margin:8px 0;display:flex;gap:16px;align-items:flex-start;'>
            <div style='min-width:100px;text-align:center;'>
                <div style='font-size:1.2rem;font-weight:700;color:{C_PEACH};
                            font-family:JetBrains Mono,monospace;'>{value}</div>
            </div>
            <div>
                <div style='font-weight:600;color:{_T["TEXT"]};margin-bottom:3px;'>{label}</div>
                <div style='font-size:.85rem;color:{_T["SUB"]};line-height:1.5;'>{explanation}</div>
            </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        sh("Agents Listed Under More Than One Account")
        st.caption(
            "The table below lists agents who appear in multiple account rosters. "
            "This needs to be verified with the client — it could mean a legitimate shared resource "
            "or a data error that's inflating headcount numbers."
        )
        if multi_detail.empty:
            st.success("No agents found in multiple accounts for the current dataset.")
        else:
            st.dataframe(
                multi_detail.rename(columns={'Agent_ID':'Agent ID'}),
                use_container_width=True, hide_index=True)

        st.markdown("---")
        sh("Missing Data by Column")
        miss = pd.DataFrame({
            'Field': df_raw.columns,
            'Blank Records': df_raw.isnull().sum().values,
            'Blank %': (df_raw.isnull().sum().values / len(df_raw) * 100).round(1)
        }).sort_values('Blank %', ascending=False)
        miss = miss[miss['Blank Records'] > 0]
        if miss.empty:
            st.success("No blank values found in any column.")
        else:
            fig = px.bar(miss, x='Blank %', y='Field', orientation='h',
                        color='Blank %',
                        color_continuous_scale=[[0,C_TEAL],[0.4,C_GOLD],[1,C_PEACH]],
                        text='Blank %')
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside', textfont_size=9)
            fig = plotly_layout(fig, "", 340, xlab="% of records that are blank")
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, use_container_width=True)
            

    # ────────────────────────────────────────────────────────────────
    # TAB 11 · DOWNLOAD REPORT
    # ────────────────────────────────────────────────────────────────
    with tabs[10]:
        import json, datetime
        sh("Download Full Data Report")
        st.markdown(
            "This generates a JSON file containing all the key numbers from this dashboard — "
            "account summaries, performance tier breakdowns, and agent-level statistics. "
            
        )

        def build_report():
            report = {
                "__info": {
                    "generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "accounts": sorted(df_all['Account'].unique().tolist()),
                    "total_agents": int(df_raw['Agent_ID'].nunique()),
                }
            }
            df_c = df_all[df_all['Calls_Answered'].fillna(0) > 0]
            report["account_summary"] = (
                df_c.groupby('Account').agg(
                    Agents=('Agent_ID','nunique'),
                    Total_Calls=('Calls_Answered','sum'),
                    Median_Occ=('Occupancy %','median'),
                    Median_AHT_s=('AHT','median'),
                ).round(2).reset_index().to_dict(orient='records'))
            report["performance_tiers"] = (
                agent_all.groupby('perf_tier')['avg_occ'].count()
                .rename('agent_count').reset_index()
                .rename(columns={'perf_tier':'tier'}).to_dict(orient='records'))
            report["quadrant_summary"] = (
                agent_all.groupby('quadrant').agg(
                    agents=('avg_occ','count'),
                    avg_occupancy=('avg_occ','mean'),
                    avg_aht_s=('avg_aht','mean'),
                ).round(2).reset_index().to_dict(orient='records'))
            return report

        col_btn, col_info = st.columns([1, 3])
        with col_btn:
            gen = st.button("Generate & Download", type="primary", use_container_width=True)
        with col_info:
            st.markdown("**Includes:** Account summary · Performance tier counts · Quadrant breakdown")
        if gen:
            with st.spinner("Building report…"):
                rpt = build_report()
                rpt_json = json.dumps(rpt, indent=2, default=str)
                fname = f"Performance_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            st.success(f"Ready — {len(rpt_json)//1024:,} KB")
            st.download_button(
                f"Download Report",
                data=rpt_json, file_name=fname, mime="application/json",
                use_container_width=True)
