import os
import streamlit as st
import pandas as pd
import plotly.express as px
from plotly import graph_objects as go
from st_aggrid import AgGrid, GridOptionsBuilder,GridUpdateMode,DataReturnMode
from sklearn.cluster import KMeans
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────────
# Page config & CSS
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Leads Dashboard", layout="wide")
st.markdown("""
<style>
.cards-container { display:flex; gap:16px; margin-bottom:16px; overflow-x:auto; }
.card { flex:0 0 auto; padding:16px; border-radius:12px;
       background-color:#000; color:#fff;
       box-shadow:0 4px 12px rgba(0,0,0,0.3); width:150px; text-align:center; }
.card-icon { font-size:32px; margin-bottom:8px; }
.card-value{ font-size:24px; font-weight:600; }
.card-title{ font-size:12px; color:#ddd; }
/* mobile stacking */
@media (max-width:600px) {
  .cards-container { flex-direction:column !important; gap:8px !important; }
  .card { width:100% !important; }
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Logging utilities
# ──────────────────────────────────────────────────────────────────────────────
def ensure_file(path, cols):
    if not os.path.exists(path):
        pd.DataFrame(columns=cols).to_csv(path, index=False)

ensure_file("user_logs.csv",  ["Timestamp","User","Action","Details"])
ensure_file("audit_logs.csv", ["Timestamp","User","Action","Enquiry No","Field","Old Value","New Value","Details"])

def log_event(user, action, details=""):
    pd.DataFrame([[datetime.now(), user, action, details]],
                 columns=["Timestamp","User","Action","Details"])\
      .to_csv("user_logs.csv", mode="a", header=False, index=False)

def log_audit(user, action, enq, field, old, new, details=""):
    pd.DataFrame([[datetime.now(), user, action, enq, field, old, new, details]],
                 columns=["Timestamp","User","Action","Enquiry No","Field","Old Value","New Value","Details"])\
      .to_csv("audit_logs.csv", mode="a", header=False, index=False)

# ──────────────────────────────────────────────────────────────────────────────
# Load & preprocess data (preserve historical 'Uploaded by')
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    tmp = pd.read_csv("leads.csv", nrows=0)
    leads = pd.read_csv(
        "leads.csv",
        dtype={"KVA": float},
        parse_dates=["Enquiry Date","Planned Followup Date","Enquiry Closure Date"],
        dayfirst=True, keep_default_na=False
    ).loc[:, ~tmp.columns.duplicated()]

    # Coerce dates
    leads["Enquiry Date"] = pd.to_datetime(leads["Enquiry Date"], errors="coerce")

    # Drop unused columns
    drop_cols = ["Corporate Name","Tehsil","Pincode","PAN NO.",
                 "Events","Finance Required","Finance Company"]
    leads.drop(columns=[c for c in drop_cols if c in leads.columns], inplace=True)

    # Ensure questionnaire columns
    for i in range(1,6):
        col = f"Question{i}"
        if col not in leads.columns:
            leads[col] = ""

    # Ensure other fields
    defaults = {
        "Remarks": "", "No of Follow-ups": 0,
        "Next Action": "", "Planned Followup Date": pd.NaT
    }
    for col, default in defaults.items():
        if col not in leads.columns:
            leads[col] = default

    # Preserve any existing 'Uploaded by'; if missing, init blank
    if "Uploaded by" not in leads.columns:
        leads["Uploaded by"] = ""

    users = pd.read_csv("users.csv")
    return leads, users

leads_df, users_df = load_data()

# ──────────────────────────────────────────────────────────────────────────────
# Filter helper
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data
def get_filtered(df, sel, kva, sd, ed):
    d = df.copy()
    for c, vals in sel.items():
        d = d[d[c].isin(vals)]
    d = d[(d["KVA"]>=kva[0]) & (d["KVA"]<=kva[1])]
    d = d[(d["Enquiry Date"].dt.date>=sd) & (d["Enquiry Date"].dt.date<=ed)]
    return d
# ──────────────────────────────────────────────────────────────────────────────
# Session & Login
# ──────────────────────────────────────────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    def _on_enter(): st.session_state["do_login"] = True
    c1,c2,c3 = st.columns([1,2,1])
    with c2:
        st.title("Please log in")
        u = st.text_input("Username", key="login_user")
        p = st.text_input("Password", type="password", key="login_pass", on_change=_on_enter)
        if st.button("Login") or st.session_state.pop("do_login", False):
            m = users_df[(users_df["Username"]==u)&(users_df["Password"]==p)]
            if not m.empty:
                st.session_state.update({
                    "logged_in": True,
                    "user": u,
                    "role": m.iloc[0]["Role"]
                })
                log_event(u, "Login")
                st.rerun()
            else:
                st.error("Invalid credentials")
    st.stop()

current_user = st.session_state["user"]
role         = st.session_state["role"]

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar Filters
# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
# Sidebar Filters  (now hierarchical: State → City → Dealer → Employee → Segment)
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    if st.button("Logout"):
        st.session_state.clear()
        st.rerun()

    with st.expander("Filters", expanded=True):
        st.header("Filter Leads")

        # 1) State
        all_states = sorted(leads_df["State"].dropna().unique())
        state_sel = st.selectbox("State", ["All"] + all_states, key="f_state")
        if state_sel == "All":
            df_state = leads_df
        else:
            df_state = leads_df[leads_df["State"] == state_sel]

        # 2) City (Location)
        all_cities = sorted(df_state["Location"].dropna().unique())
        city_sel = st.selectbox("City", ["All"] + all_cities, key="f_city")
        if city_sel == "All":
            df_city = df_state
        else:
            df_city = df_state[df_state["Location"] == city_sel]

        # 3) Dealer
        all_dealers = sorted(df_city["Dealer"].dropna().unique())
        dealer_sel = st.selectbox("Dealer", ["All"] + all_dealers, key="f_dealer")
        if dealer_sel == "All":
            df_dealer = df_city
        else:
            df_dealer = df_city[df_city["Dealer"] == dealer_sel]

        # 4) Employee
        all_emps = sorted(df_dealer["Employee Name"].dropna().unique())
        emp_sel = st.selectbox("Employee", ["All"] + all_emps, key="f_emp")
        if emp_sel == "All":
            df_emp = df_dealer
        else:
            df_emp = df_dealer[df_dealer["Employee Name"] == emp_sel]

        # 5) Segment
        all_segs = sorted(df_emp["Segment"].dropna().unique())
        seg_sel = st.selectbox("Segment", ["All"] + all_segs, key="f_seg")
        if seg_sel == "All":
            df_seg = df_emp
        else:
            df_seg = df_emp[df_emp["Segment"] == seg_sel]

        # Collect selections for filtering
        selected = {
            "State":       all_states if state_sel=="All"  else [state_sel],
            "Location":    sorted(leads_df["Location"].unique()) if city_sel=="All"  else [city_sel],
            "Dealer":      sorted(leads_df["Dealer"].unique())   if dealer_sel=="All" else [dealer_sel],
            "Employee Name": all_emps if emp_sel=="All"          else [emp_sel],
            "Segment":     sorted(leads_df["Segment"].unique())  if seg_sel=="All"    else [seg_sel],
        }

        # 6) KVA & Date range (unchanged)
        mn = int(leads_df["KVA"].min() or 0)
        mx = int(leads_df["KVA"].max() or mn + 1)
        kva_range = st.slider("KVA Range", mn, mx, (mn, mx), key="f_kva")

        today = datetime.today().date()
        first = today.replace(day=1)
        date_vals = st.date_input("Enquiry Date Range", (first, today), key="f_date")
        if isinstance(date_vals, (list, tuple)) and len(date_vals)==2:
            start_date, end_date = date_vals
        else:
            start_date, end_date = first, today

# ──────────────────────────────────────────────────────────────────────────────
# Apply filters + Employee scoping
# ──────────────────────────────────────────────────────────────────────────────
filtered_df = get_filtered(leads_df, selected, kva_range, start_date, end_date)
# 1) coerce both date columns to datetime64 (in case some rows slipped through)
filtered_df["Enquiry Closure Date"] = pd.to_datetime(
    filtered_df["Enquiry Closure Date"], errors="coerce"
)
filtered_df["Enquiry Date"] = pd.to_datetime(
    filtered_df["Enquiry Date"], errors="coerce"
)

# 2) get 'today' as a pandas Timestamp
today_ts = pd.Timestamp.today().normalize()

# 3) fill NaT on the closure date, and also on enquiry date just in case
close_series = filtered_df["Enquiry Closure Date"].fillna(today_ts)
enq_series   = filtered_df["Enquiry Date"].fillna(today_ts)

# 4) subtract series from series, then take the number of days
filtered_df["Lead Age (Days)"] = (close_series - enq_series).dt.days
log_event(current_user, "Filter Applied",
          f"{selected}, KVA={kva_range}, Dates={start_date}–{end_date}")
# ── compute how many days each lead has been open ─────────────────────────


# Employee sees only leads whose Employee Name first token matches OR they uploaded
if role == "Employee":
    fn = current_user.split()[0].lower()
    mask_name   = (
        filtered_df["Employee Name"]
          .str.split().str[0]
          .str.lower()
          .eq(fn)
    )
    mask_upload = filtered_df["Uploaded by"].str.lower().str.contains(fn, na=False)
    filtered_df = filtered_df[mask_name | mask_upload]

open_stages = ["Prospecting","Qualified"]
won_stages  = ["Closed-Won","Order Booked"]
lost_stages = ["Closed-Dropped","Closed-Lost"]
# ──────────────────────────────────────────────────────────────────────────────
# Tabs setup
# ──────────────────────────────────────────────────────────────────────────────
tabs = ["KPI","Charts","Top Dealers","Top Employees",
        "Upload New Lead","Lead Update","Insights"]
if role=="Admin": tabs.append("Admin")
tabs = st.tabs(tabs)

# ──────────────────────────────────────────────────────────────────────────────
# KPI Tab (edit form removed)
# ──────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("Key Performance Indicators")

    # KPI cards (unchanged)…
    total = len(filtered_df)
    open_cnt = filtered_df["Enquiry Stage"].isin(open_stages).sum()
    won_cnt = filtered_df["Enquiry Stage"].isin(won_stages).sum()
    lost_cnt = filtered_df["Enquiry Stage"].isin(lost_stages).sum()
    conv_pct = f"{won_cnt / total * 100:.1f}%" if total else "0%"
    closed_pct = f"{(won_cnt + lost_cnt) / total * 100:.1f}%" if total else "0%"

    html = '<div class="cards-container">'
    for icon, title, val in [
        ("📈", "Total Leads", total),
        ("🕒", "Open Leads", open_cnt),
        ("❌", "Lost Leads", lost_cnt),
        ("🏆", "Won Leads", won_cnt),
        ("🔄", "Conversion %", conv_pct),
        ("✅", "Closed %", closed_pct),
    ]:
        html += (
            f"<div class='card'>"
            f"<div class='card-icon'>{icon}</div>"
            f"<div class='card-value'>{val}</div>"
            f"<div class='card-title'>{title}</div>"
            "</div>"
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

    # Drill‑down filter (unchanged)…
    choice = st.radio(
        "Details for:", ["All", "Open", "Lost", "Won"],
        horizontal=True, key="kpi_drill"
    )
    if choice == "All":
        ddf = filtered_df
    elif choice == "Open":
        ddf = filtered_df[filtered_df["Enquiry Stage"].isin(open_stages)]
    elif choice == "Lost":
        ddf = filtered_df[filtered_df["Enquiry Stage"].isin(lost_stages)]
    else:
        ddf = filtered_df[filtered_df["Enquiry Stage"].isin(won_stages)]

    # Search & select a lead snapshot
    st.markdown("### Lead Details (search & select below)")
    opts = (
        ddf["Enquiry No"].astype(str)
        + " – "
        + ddf["Name"]
    ).tolist()
    chosen = st.multiselect(
        "Search & select a lead",
        opts,
        default=[],
        max_selections=1,
        key="kpi_lead_select"
    )

    if chosen:
        enq_no, _ = chosen[0].split(" – ", 1)
        row = leads_df[leads_df["Enquiry No"].astype(str) == enq_no].iloc[0]

        with st.expander(f"📋 Lead #{enq_no} Snapshot", expanded=True):
            st.markdown("**Lead Snapshot**")
            st.write(f"**Lead Age (Days):** {row.get('Lead Age (Days)', 'N/A')}")

            # Core info
            for col in (
                "Enquiry No", "Name", "Dealer", "Employee Name",
                "Enquiry Stage", "Phone Number", "Email"
            ):
                st.write(f"**{col}:** {row.get(col, '')}")

            # Questionnaire answers
            for i in range(1, 6):
                st.write(f"**Question{i}:** {row.get(f'Question{i}', '')}")

            # Follow‑up info
            pf = pd.to_datetime(row.get("Planned Followup Date"), errors="coerce")
            pf_str = pf.date().isoformat() if pd.notna(pf) else "N/A"
            st.write(f"**Planned Follow‑up Date:** {pf_str}")
            st.write(f"**No of Follow‑ups:** {row.get('No of Follow‑ups', 0)}")
            st.write(f"**Next Action:** {row.get('Next Action', '')}")

    # Summary table (unchanged)…
   # ─────────────────────────────────────────────────────────────────────────
    # Summary table (re‐ordered columns)
    # ─────────────────────────────────────────────────────────────────────────
    if not ddf.empty:
        # 1) Define your front‐of‐table preference
        pref = [
            "Name",
            "Dealer",
            "Employee Name",
            "Segment",
            # pick one of these location columns if present
            ( "Location" 
              if "Location" in ddf.columns 
              else ("Area Office" if "Area Office" in ddf.columns else "District")
            ),
            "KVA",
        ]

        # 2) Build ordered list: preferred first (if they exist), then the rest
        ordered_cols = [c for c in pref if c in ddf.columns] \
                     + [c for c in ddf.columns if c not in pref]

        ordered = ddf[ordered_cols]

        # 3) Render via AgGrid as before
        gb = GridOptionsBuilder.from_dataframe(ordered)
        gb.configure_pagination(paginationAutoPageSize=True)
        gb.configure_default_column(enableValue=True, sortable=True, filter=True)
        AgGrid(
            ordered,
            gridOptions=gb.build(),
            enable_enterprise_modules=False,
        )
    else:
        st.info("No leads to display.")

# --- Charts Tab ---
### ── REPLACE YOUR ENTIRE CHARTS TAB WITH THIS BLOCK ────────────────────────
with tabs[1]:
    st.subheader("Leads Visualisations")

    # 1️⃣  Pipeline Funnel (unchanged)
    counts = (
        filtered_df["Enquiry Stage"]
        .value_counts()
        .reindex(open_stages + won_stages + lost_stages, fill_value=0)
    )
    funnel_vals = (
        [counts[s] for s in open_stages] +
        [counts[won_stages[0]] + counts[won_stages[1]]] +
        [counts[lost_stages[0]] + counts[lost_stages[1]]]
    )
    st.plotly_chart(
        go.Figure(
            go.Funnel(y=["Prospecting","Qualified","Won","Lost"], x=funnel_vals)
        ).update_layout(title="Lead Pipeline Funnel"),
        use_container_width=True,
    )

    # 2️⃣  Helper to build Top‑10 charts
    def top10(df, group, metric):
        if metric == "Lead Age (Days)":
            agg = (
                df.groupby(group)["Lead Age (Days)"]
                .mean()
                .reset_index(name="MetricValue")
                .sort_values("MetricValue", ascending=False)
                .head(10)
            )
        else:
        # … your existing Total/Open/Closed/Conversion logic …
            stage_lists = df.groupby(group)["Enquiry Stage"].agg(list)
            def value(lst):
                if metric=="Total":   return len(lst)
                if metric=="Open":    return sum(s in open_stages for s in lst)
                if metric=="Closed":  return sum(s in won_stages+lost_stages for s in lst)
                t = len(lst); w = sum(s in won_stages for s in lst)
                return w/t*100 if t else 0
            agg = (
                stage_lists.apply(value)
                           .sort_values(ascending=False)
                           .head(10)
                           .reset_index(name="MetricValue")
            )

    # ── round for display ────────────────────────────────────────
        if metric == "Conversion":
            agg["MetricValue"] = agg["MetricValue"].round(1)
        elif metric == "Lead Age (Days)":
            agg["MetricValue"] = agg["MetricValue"].round(1).astype(int)

        bar_color = "red" if metric in ("Open","Lead Age (Days)") else "#1f77b4"
        ylab = (
            "Average Lead Age (Days)" if metric=="Lead Age (Days)"
            else ("Conversion %" if metric=="Conversion" else f"Leads {metric}")
        )
        title = f"Top 10 {group}s by {metric}"

        fig = px.bar(
            agg,
            x=group,
            y="MetricValue",
            labels={group:group, "MetricValue":ylab},
            title=title,
            color_discrete_sequence=[bar_color],
            text="MetricValue",
        )
        if metric == "Conversion":
            fig.update_yaxes(ticksuffix="%")
        return fig


    # 3️⃣  Metric selector
    metric_opt = st.selectbox(
        "Metric for all Top‑10 charts",
        ["Total", "Open", "Closed", "Conversion","Lead Age (Days)"],
        key="top10_metric",
    )

    # 4️⃣  Four Top‑10 charts
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(top10(filtered_df, "Dealer",        metric_opt), use_container_width=True)
    with col2:
        st.plotly_chart(top10(filtered_df, "Employee Name", metric_opt), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(top10(filtered_df, "State",   metric_opt), use_container_width=True)
    with col4:
        st.plotly_chart(top10(filtered_df, "Segment", metric_opt), use_container_width=True)
### ── END REPLACEMENT ────────────────────────────────────────────────────────
    # ── Trend Analysis ──────────────────────────────────────────────
    # ── Trend Analysis ──────────────────────────────────────────────
        st.subheader("Leads Trend with 7‑day MA")
    freq = st.radio(
    "Aggregation",
    ["Daily", "Weekly", "Monthly"],
    horizontal=True,
    key="trend_freq",
)
    rule = {"Daily":"D", "Weekly":"W", "Monthly":"M"}[freq]

# Resample, count, compute MA, and reset index so 'Enquiry Date' is a column
    ts = (
    filtered_df
      .set_index("Enquiry Date")
      .resample(rule)
      .size()
      .rename("count")
      .to_frame()
      .reset_index()
)
    ts["7‑day MA"] = ts["count"].rolling(window=7, min_periods=1).mean()

# Now plot with 'Enquiry Date' as the x‑column
    fig = px.line(
    ts,
    x="Enquiry Date",
    y=["count","7‑day MA"],
    labels={"value":"Leads","Enquiry Date":"Date"},
    title=f"Leads per {freq} with 7‑day Moving Average",
)
    st.plotly_chart(fig, use_container_width=True)



# --- Top Dealers & Top Employees ---
def top5(df, by):
    agg = df.groupby(by).agg(
        Total_Leads=("Enquiry No","count"),
        Won_Leads=("Enquiry Stage", lambda x: x.isin(won_stages).sum()),
        Lost_Leads=("Enquiry Stage", lambda x: x.isin(lost_stages).sum())
    )
    agg["Conv %"] = (agg["Won_Leads"]/agg["Total_Leads"]*100).round(1)
    return agg.sort_values("Won_Leads", ascending=False).head(5).reset_index()

with tabs[2]:
    st.subheader("Top 5 Dealers")
    st.table(top5(filtered_df, "Dealer"))
with tabs[3]:
    st.subheader("Top 5 Employees")
    st.table(top5(filtered_df, "Employee Name"))
# --- Upload New Lead ---
with tabs[4]:
    st.subheader("Upload New Lead")
    uf = st.file_uploader("Upload leads Excel (xlsx)", type="xlsx", key="upload_new")
    if uf:
        df_new = pd.read_excel(uf, engine="openpyxl")
        for i in range(1,6):
            col = f"Question{i}"
            if col not in df_new.columns:
                df_new[col] = ""
        if "Uploaded by" not in df_new.columns:
            df_new["Uploaded by"] = current_user
        if "upload_idx" not in st.session_state:
            st.session_state.upload_idx = 0
            st.session_state.new_df   = df_new
        idx, total = st.session_state.upload_idx, len(df_new)

        if idx < total:
            lead = st.session_state.new_df.iloc[idx]
            st.write(f"**Lead {idx+1}/{total}: {lead['Name']}**")
            with st.form(key=f"newlead_form_{idx}"):
                q1 = st.selectbox("Q1. Site status", ["under construction","nearly constructed","constructed","planning"], key=f"q1_{idx}")
                q2 = st.selectbox("Q2. Contact person", ["Owner","Manager","Purchase Dept","Other"], key=f"q2_{idx}")
                q3 = st.selectbox("Q3. Decision maker?", ["Yes","No"], key=f"q3_{idx}")
                q4 = st.selectbox("Q4. Orientation", ["Price","Quality"], key=f"q4_{idx}")
                q5 = st.selectbox("Q5. Who decides?", ["contact person","owner","manager","purchase head"], key=f"q5_{idx}")
                submitted = st.form_submit_button("Submit Lead")
            if submitted:
                name = lead["Name"]
                enq_no = lead["Enquiry No"]
                exists = (leads_df["Enquiry No"] == enq_no).any()
                if exists:
                    st.warning(f"Lead with Enquiry No {enq_no} already exists; skipped.")
                else:
                    for i, ans in enumerate((q1,q2,q3,q4,q5), start=1):
                        log_audit(current_user,"Create",lead["Enquiry No"], f"Question{i}","",ans,"Questionnaire")
                        st.session_state.new_df.at[idx, f"Question{i}"] = ans
                    entry = st.session_state.new_df.loc[idx].copy()
                    entry["Created By"]    = current_user
                    entry["Uploaded by"]   = current_user
                    leads_df.loc[len(leads_df)] = entry
                    leads_df.to_csv("leads.csv", index=False)
                    st.cache_data.clear()
                    log_event(current_user,"New Lead Uploaded",name)
                    st.success(f"Lead '{name}' added.")
                st.session_state.upload_idx += 1
                st.rerun()

# --- Lead Update ---
with tabs[5]:
    st.subheader("Lead Update")
    open_df = filtered_df[filtered_df["Enquiry Stage"].isin(open_stages)]
    if open_df.empty:
        st.info("No open leads.")
    else:
        search = st.text_input("Search Lead (Name or Enq No)", key="update_search")
        opts   = (open_df["Enquiry No"].astype(str)+" - "+open_df["Name"]).tolist()
        if search:
            opts = [o for o in opts if search.lower() in o.lower()]
        if not opts:
            st.warning("No leads found.")
        else:
            sel = st.selectbox("Select Lead", opts, key="update_select")
            enq = sel.split(" - ",1)[0]
            row = open_df[open_df["Enquiry No"].astype(str)==enq].iloc[0]
            idx = row.name
            with st.form("update_form"):
                stages    = list(filtered_df["Enquiry Stage"].dropna().unique())
                new_stage = st.selectbox("Enquiry Stage", stages, index=stages.index(row["Enquiry Stage"]), key="update_stage")
                new_remark= st.text_area("Remarks", value=row.get("Remarks",""), key="update_remarks")
                pf = pd.to_datetime(row.get("Planned Followup Date", None), errors="coerce")
                default_date = pf.date() if pd.notna(pf) else datetime.today().date()

                new_date = st.date_input(
                    "Next Follow‑up Date",
                     value=default_date,
                     key="update_date"
                )
                new_fu    = st.number_input("No of Follow-ups", min_value=0, value=int(row.get("No of Follow-ups",0)), key="update_fu")
                new_act   = st.text_input("Next Action", value=row.get("Next Action",""), key="update_action")
                submitted = st.form_submit_button("Save Changes")
            if submitted:
                updates = {
                    "Enquiry Stage": new_stage,
                    "Remarks": new_remark,
                    "Planned Followup Date": pd.to_datetime(new_date),
                    "No of Follow-ups": new_fu,
                    "Next Action": new_act
                }
                for field,new_val in updates.items():
                    old_val = leads_df.at[idx, field]
                    if (pd.isna(old_val) and new_val is not None) or (old_val != new_val):
                        log_audit(current_user,"Update",enq,field,old_val,new_val,"Lead changed")
                        leads_df.at[idx, field] = new_val
                if new_stage in won_stages:
                    leads_df.at[idx,"EnquiryStatus"]="Converted"
                    leads_df.at[idx,"Enquiry Closure Date"]=datetime.now()
                elif new_stage in lost_stages:
                    leads_df.at[idx,"EnquiryStatus"]="Closed"
                    leads_df.at[idx,"Enquiry Closure Date"]=datetime.now()
                leads_df.to_csv("leads.csv", index=False)
                st.cache_data.clear()
                log_event(current_user,"Lead Updated",f"{enq} -> {new_stage}")
                st.success("Lead updated.")
                st.rerun()

# --- Insights (Dealer Segmentation) ---
with tabs[6]:
    st.subheader("Dealer Segmentation (K‑Means)")
    stats = (
        filtered_df.groupby("Dealer")["Enquiry No"]
        .agg(Total_Leads="count").reset_index()
    )
    stats["Conversion %"] = (
        filtered_df.groupby("Dealer")["Enquiry Stage"]
        .apply(lambda x: x.isin(won_stages).sum()/len(x)*100).values
    )
    stats = stats[stats["Total_Leads"]>=5]
    if len(stats)>=3:
        X = stats[["Total_Leads","Conversion %"]]
        stats["Cluster"] = KMeans(n_clusters=3, random_state=0).fit_predict(X).astype(str)
        fig = px.scatter(stats, x="Total_Leads", y="Conversion %",
                         color="Cluster", hover_data=["Dealer"],
                         title="Dealer Clusters")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(stats, use_container_width=True)
    else:
        st.info("Not enough data (≥3 dealers with ≥5 leads).")

# --- Admin Panel ---
if role=="Admin":
    with tabs[-1]:
        st.subheader("Admin Panel")
        # Historical upload...
        hf = st.file_uploader("Upload Historical Leads", type=["xlsx","csv"], key="hist")
        if hf and st.button("Process Historical", key="hist_btn"):
            if hf.name.endswith(".xlsx"):
                hdf = pd.read_excel(hf, engine="openpyxl")
            else:
                hdf = pd.read_csv(hf)
            if "Uploaded by" not in hdf.columns:
                hdf["Uploaded by"] = current_user
            else:
                hdf["Uploaded by"] = hdf["Uploaded by"].fillna(current_user)
            orig = len(leads_df)
            combo = pd.concat([leads_df, hdf], ignore_index=True)
            combo.drop_duplicates(subset=["Enquiry No"], keep="first", inplace=True)
            combo.to_csv("leads.csv", index=False)
            st.success(f"{len(combo)-orig} added.")
            st.cache_data.clear()
            log_event(current_user,"Historical Upload",f"{len(combo)-orig}")
            st.rerun()

        st.markdown("---")
        # Reset Data
        confirm = st.text_input("Type DELETE to confirm reset", key="rst")
        if st.button("Reset All Data") and confirm=="DELETE":
            pd.DataFrame(columns=leads_df.columns).to_csv("leads.csv", index=False)
            st.success("Data wiped.")
            st.cache_data.clear()
            log_event(current_user,"Dashboard Reset")
            st.rerun()

        st.markdown("---")
        # Audit Logs
        audit = pd.read_csv("audit_logs.csv")
        st.download_button("Download Audit Logs", audit.to_csv(index=False).encode(), "audit_logs.csv")
        st.dataframe(audit, use_container_width=True)

        st.markdown("---")
        # User Management
        st.subheader("Users")
        st.dataframe(users_df[["Username","Role"]], use_container_width=True)
        with st.form("add_user"):
            nu = st.text_input("Username")
            np = st.text_input("Password", type="password")
            nr = st.selectbox("Role", ["Admin","Manager","Employee"])
            if st.form_submit_button("Add User"):
                if nu and np:
                    if nu in users_df["Username"].values:
                        st.warning("Exists.")
                    else:
                        users_df.loc[len(users_df)] = [nu,np,nr]
                        users_df.to_csv("users.csv", index=False)
                        log_event(current_user,"User Added",nu)
                        st.success(f"Added {nu}.")
                        st.rerun()
        to_del = st.multiselect("Delete Users", [u for u in users_df["Username"] if u!=current_user])
        if st.button("Delete Selected"):
            if to_del:
                users_df = users_df[~users_df["Username"].isin(to_del)]
                users_df.to_csv("users.csv", index=False)
                log_event(current_user,"User Deleted",",".join(to_del))
                st.success(f"Deleted {', '.join(to_del)}.")
                st.rerun()

        st.markdown("---")
        # User Activity
        ul = pd.read_csv("user_logs.csv")
        st.download_button("Download User Log", ul.to_csv(index=False).encode(), "user_logs.csv")
        st.dataframe(ul, use_container_width=True)
