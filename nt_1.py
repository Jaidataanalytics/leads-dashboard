import os
import streamlit as st
import pandas as pd
import plotly.express as px
from plotly import graph_objects as go
from st_aggrid import AgGrid, GridOptionsBuilder
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────────
# Page config & CSS for KPI cards
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Leads Dashboard", layout="wide")
st.markdown("""
<style>
.cards-container {
  display: flex; gap: 16px; margin-bottom: 16px; overflow-x: auto;
}
.card {
  flex: 0 0 auto; padding: 16px; border-radius: 12px;
  background-color: #000; color: #fff;
  box-shadow: 0 4px 12px rgba(0,0,0,0.3);
  width: 150px; text-align: center;
}
.card-icon { font-size:32px; margin-bottom:8px; }
.card-value{ font-size:24px; font-weight:600; }
.card-title{ font-size:12px; color:#ddd; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Ensure log files exist
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
# Load & preprocess data
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    # read header to drop duplicate columns
    tmp = pd.read_csv("leads.csv", nrows=0)
    leads = pd.read_csv(
        "leads.csv",
        dtype={"KVA": float},
        parse_dates=[
            "Enquiry Date","Planned Followup Date","Enquiry Closure Date"
        ],
        dayfirst=True, keep_default_na=False
    ).loc[:, ~tmp.columns.duplicated()]

    leads["Enquiry Date"] = pd.to_datetime(leads["Enquiry Date"], errors="coerce")

    # drop unused columns
    drop_cols = ["Corporate Name","Tehsil","Pincode","PAN NO.",
                 "Events","Finance Required","Finance Company"]
    leads.drop(columns=[c for c in drop_cols if c in leads.columns],
               inplace=True)

    # ensure questionnaire columns
    for i in range(1,6):
        col = f"Question{i}"
        if col not in leads.columns:
            leads[col] = ""

    # ensure other columns exist
    defaults = {
        "Remarks": "",
        "No of Follow-ups": 0,
        "Next Action": "",
        "Planned Followup Date": pd.NaT
    }
    for col, default in defaults.items():
        if col not in leads.columns:
            leads[col] = default

    users = pd.read_csv("users.csv")
    return leads, users

leads_df, users_df = load_data()

# ──────────────────────────────────────────────────────────────────────────────
# Session state: login
# ──────────────────────────────────────────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
current_user = st.session_state.get("user")
role         = st.session_state.get("role")

if not st.session_state["logged_in"]:
    def _on_enter():
        st.session_state["trigger_login"] = True

    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.title("Please log in")
        u = st.text_input("Username", key="login_user")
        p = st.text_input("Password", type="password",
                         key="login_pass", on_change=_on_enter)
        btn = st.button("Login")
        if btn or st.session_state.pop("trigger_login", False):
            m = users_df[
                (users_df["Username"]==u)&(users_df["Password"]==p)
            ]
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

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar: logout & filters
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    if st.button("Logout"):
        st.session_state.clear()
        st.rerun()

    st.header("Filters")
    FILTERS = [
        ("State","State","f1"),
        ("City","Location","f2"),
        ("Dealer","Dealer","f3"),
        ("Employee","Employee Name","f4"),
        ("Segment","Segment","f5"),
    ]
    selected = {}
    for label, col, key in FILTERS:
        vals = sorted(leads_df[col].dropna().unique())
        choice = st.selectbox(label, ["All"]+vals, key=key)
        selected[col] = vals if choice=="All" else [choice]

    # KVA slider
    if pd.notna(leads_df["KVA"]).any():
        mn, mx = int(leads_df["KVA"].min()), int(leads_df["KVA"].max())
    else:
        mn, mx = 0, 1
    if mn>=mx: mx = mn+1
    kva_range = st.slider("KVA Range", mn, mx, (mn,mx), key="f6")

    # date range
    today = datetime.today().date()
    first = today.replace(day=1)
    dates = st.date_input("Enquiry Date Range", [first,today], key="f7")
    if isinstance(dates,(list,tuple)) and len(dates)==2:
        start_date, end_date = dates
    else:
        start_date, end_date = first, today

# ──────────────────────────────────────────────────────────────────────────────
# Apply filters
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data
def get_filtered(df, sel, kva, sd, ed):
    d = df.copy()
    for col, vals in sel.items():
        d = d[d[col].isin(vals)]
    d = d[(d["KVA"]>=kva[0])&(d["KVA"]<=kva[1])]
    d = d[(d["Enquiry Date"].dt.date>=sd)&(d["Enquiry Date"].dt.date<=ed)]
    return d

filtered_df = get_filtered(leads_df, selected, kva_range, start_date, end_date)
log_event(
    current_user, "Filter Applied",
    f"{selected}, KVA={kva_range}, Dates={start_date}–{end_date}"
)

open_stages = ["Prospecting","Qualified"]
won_stages  = ["Closed-Won","Order Booked"]
lost_stages = ["Closed-Dropped","Closed-Lost"]

tabs = ["KPI","Charts","Top Dealers","Top Employees","Upload New Lead","Lead Update"]
if role=="Admin":
    tabs.append("Admin")
tabs = st.tabs(tabs)

# ──────────────────────────────────────────────────────────────────────────────
# KPI Tab
# ──────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("Key Performance Indicators")
    total      = len(filtered_df)
    open_cnt   = filtered_df["Enquiry Stage"].isin(open_stages).sum()
    won_cnt    = filtered_df["Enquiry Stage"].isin(won_stages).sum()
    lost_cnt   = filtered_df["Enquiry Stage"].isin(lost_stages).sum()
    conv_pct   = f"{(won_cnt/total*100):.1f}%"    if total else "0%"
    closed_pct = f"{((won_cnt+lost_cnt)/total*100):.1f}%" if total else "0%"

    cards = [
        ("📈","Total Leads", total),
        ("🕒","Open Leads",  open_cnt),
        ("❌","Lost Leads",  lost_cnt),
        ("🏆","Won Leads",   won_cnt),
        ("🔄","Conv %",      conv_pct),
        ("✅","Closed %",    closed_pct),
    ]
    html = '<div class="cards-container">'
    for icon, title, val in cards:
        html += f'''
          <div class="card">
            <div class="card-icon">{icon}</div>
            <div class="card-value">{val}</div>
            <div class="card-title">{title}</div>
          </div>'''
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

    choice = st.radio("View details for:", ["All","Open","Lost","Won"],
                      horizontal=True, key="kpi_drill")
    if choice=="All":
        ddf = filtered_df
    elif choice=="Open":
        ddf = filtered_df[filtered_df["Enquiry Stage"].isin(open_stages)]
    elif choice=="Lost":
        ddf = filtered_df[filtered_df["Enquiry Stage"].isin(lost_stages)]
    else:
        ddf = filtered_df[filtered_df["Enquiry Stage"].isin(won_stages)]

    gb = GridOptionsBuilder.from_dataframe(ddf)
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_default_column(enableValue=True, sortable=True, filter=True)
    AgGrid(ddf, gridOptions=gb.build(), enable_enterprise_modules=False)

# ──────────────────────────────────────────────────────────────────────────────
# Charts Tab
# ──────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Leads Visualization")
    counts = filtered_df["Enquiry Stage"].value_counts().reindex(
        open_stages + won_stages + lost_stages, fill_value=0
    )
    funnel_vals = (
        [counts[s] for s in open_stages] +
        [counts[won_stages[0]] + counts[won_stages[1]]] +
        [counts[lost_stages[0]] + counts[lost_stages[1]]]
    )
    fig = go.Figure(go.Funnel(
        y=["Prospecting","Qualified","Won","Lost"], x=funnel_vals
    ))
    fig.update_layout(title="Lead Pipeline Funnel")
    st.plotly_chart(fig, use_container_width=True)

    top_d = (
        filtered_df["Dealer"]
        .value_counts()
        .nlargest(10)
        .rename_axis("Dealer")
        .reset_index(name="Leads")
    )
    st.plotly_chart(px.bar(top_d, x="Dealer", y="Leads", title="Top 10 Dealers"), use_container_width=True)

    gran = st.selectbox("Time Series Granularity", ["Daily","Weekly","Monthly"], key="ts_gran")
    freq = {"Daily":"D","Weekly":"W","Monthly":"M"}[gran]
    win  = {"D":7,"W":4,"M":3}[freq]
    ts_df = (
        filtered_df
        .set_index("Enquiry Date")
        .resample(freq).size().rename("Leads").to_frame().reset_index()
    )
    ts_df[f"MA({win})"] = ts_df["Leads"].rolling(win, min_periods=1).mean()
    st.plotly_chart(
        px.line(
            ts_df, x="Enquiry Date", y=["Leads",f"MA({win})"],
            labels={"value":"Count","variable":"Metric"},
            title=f"{gran} Leads & {win}-Period MA"
        ),
        use_container_width=True
    )

# ──────────────────────────────────────────────────────────────────────────────
# Top Dealers & Employees Tabs
# ──────────────────────────────────────────────────────────────────────────────
def top5(df, by):
    agg = df.groupby(by).agg(
        Total_Leads=("Enquiry No","count"),
        Won_Leads=("Enquiry Stage", lambda x: x.isin(won_stages).sum()),
        Lost_Leads=("Enquiry Stage", lambda x: x.isin(lost_stages).sum())
    )
    agg["Conv %"] = (agg["Won_Leads"]/agg["Total_Leads"]*100).round(1)
    return agg.sort_values("Won_Leads", ascending=False).head(5).reset_index()

with tabs[2]:
    st.subheader("Top 5 Dealers")
    st.table(top5(filtered_df, "Dealer"))

with tabs[3]:
    st.subheader("Top 5 Employees")
    st.table(top5(filtered_df, "Employee Name"))

# ──────────────────────────────────────────────────────────────────────────────
# Upload New Lead Tab
# ──────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.subheader("Upload New Lead")
    upload_file = st.file_uploader(
        "Upload leads Excel (xlsx)", type="xlsx",
        key="upload_new_lead_file"
    )
    if upload_file:
        df_new = pd.read_excel(upload_file, engine="openpyxl")
        for i in range(1,6):
            col = f"Question{i}"
            if col not in df_new.columns:
                df_new[col] = ""
        if "upload_idx" not in st.session_state:
            st.session_state.upload_idx = 0
            st.session_state.new_df   = df_new
        idx   = st.session_state.upload_idx
        total = len(st.session_state.new_df)

        if idx < total:
            lead = st.session_state.new_df.iloc[idx]
            st.write(f"**Lead {idx+1}/{total}: {lead['Name']}**")
            # use form so button always shows
            with st.form(key=f"newlead_form_{idx}"):
                q1 = st.selectbox("Q1. Status of the site",
                                  ["under construction","nearly constructed","constructed","planning"],
                                  key=f"q1_{idx}")
                q2 = st.selectbox("Q2. Contact person",
                                  ["Owner","Manager","Purchase Dept","Other"],
                                  key=f"q2_{idx}")
                q3 = st.selectbox("Q3. Decision maker?",["Yes","No"], key=f"q3_{idx}")
                q4 = st.selectbox("Q4. Customer orientation",["Price","Quality"], key=f"q4_{idx}")
                q5 = st.selectbox("Q5. Who decides?",
                                  ["contact person","owner","manager","purchase head"],
                                  key=f"q5_{idx}")
                submitted = st.form_submit_button("Submit Lead")
            if submitted:
                name, phone = lead["Name"], lead["Phone Number"]
                exists = ((leads_df["Name"]==name)&(leads_df["Phone Number"]==phone)).any()
                if exists:
                    st.warning(f"Lead '{name}' exists; skipped.")
                else:
                    for i, ans in enumerate((q1,q2,q3,q4,q5), start=1):
                        log_audit(current_user,"Create",lead["Enquiry No"],
                                  f"Question{i}","",ans,"Questionnaire answer")
                        st.session_state.new_df.at[idx, f"Question{i}"] = ans
                    entry = st.session_state.new_df.loc[idx].copy()
                    entry["Created By"] = current_user
                    leads_df.loc[len(leads_df)] = entry
                    leads_df.to_csv("leads.csv", index=False)
                    st.cache_data.clear()
                    log_event(current_user,"New Lead Uploaded",name)
                    st.success(f"Lead '{name}' added.")
                st.session_state.upload_idx += 1
                st.rerun()
# ──────────────────────────────────────────────────────────────────────────────
# Lead Update Tab
# ──────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.subheader("Lead Update")
    open_df = filtered_df[filtered_df["Enquiry Stage"].isin(open_stages)]
    if open_df.empty:
        st.info("No open leads.")
    else:
        search = st.text_input("Search Lead (Name or Enq No)", key="update_search")
        opts = (open_df["Enquiry No"].astype(str)+" - "+open_df["Name"]).tolist()
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
                new_stage = st.selectbox("Enquiry Stage", stages,
                                         index=stages.index(row["Enquiry Stage"]),
                                         key="update_stage")
                new_remark = st.text_area("Remarks",
                                         value=row.get("Remarks",""),
                                         key="update_remarks")
                new_date = st.date_input(
                    "Next Follow‑up Date",
                    value=(row["Planned Followup Date"].date()
                           if pd.notna(row["Planned Followup Date"])
                           else datetime.today().date()),
                    key="update_date"
                )
                new_fu  = st.number_input("No of Follow-ups",
                                         min_value=0,
                                         value=int(row.get("No of Follow-ups",0)),
                                         key="update_fu")
                new_act = st.text_input("Next Action",
                                        value=row.get("Next Action",""),
                                        key="update_action")
                if st.form_submit_button("Save Changes", key="update_submit"):
                    updates = {
                        "Enquiry Stage": new_stage,
                        "Remarks": new_remark,
                        "Planned Followup Date": pd.to_datetime(new_date),
                        "No of Follow-ups": new_fu,
                        "Next Action": new_act
                    }
                    for field, new_val in updates.items():
                        old_val = leads_df.at[idx, field]
                        if (pd.isna(old_val) and new_val is not None) or (old_val != new_val):
                            log_audit(current_user,"Update",enq,
                                      field, old_val, new_val, "Lead field changed")
                            leads_df.at[idx, field] = new_val
                    if new_stage in won_stages:
                        leads_df.at[idx,"EnquiryStatus"] = "Converted"
                        leads_df.at[idx,"Enquiry Closure Date"] = datetime.now()
                    elif new_stage in lost_stages:
                        leads_df.at[idx,"EnquiryStatus"] = "Closed"
                        leads_df.at[idx,"Enquiry Closure Date"] = datetime.now()
                    leads_df.to_csv("leads.csv", index=False)
                    st.cache_data.clear()
                    log_event(current_user,"Lead Updated",f"{enq} -> {new_stage}")
                    st.success("Lead updated successfully.")
                    st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# Admin Panel
# ──────────────────────────────────────────────────────────────────────────────
if role=="Admin":
    with tabs[-1]:
        st.subheader("Admin Panel")

        # Historical Upload
        hf = st.file_uploader(
            "Upload Historical Leads (xlsx/csv)",
            type=["xlsx","csv"],
            key="upload_historical_file"
        )
        if hf and st.button("Process Historical Upload", key="hist_submit"):
            if hf.name.endswith(".xlsx"):
                hdf = pd.read_excel(hf, engine="openpyxl")
            else:
                hdf = pd.read_csv(hf)
            orig  = len(leads_df)
            combo = pd.concat([leads_df, hdf], ignore_index=True)
            combo.drop_duplicates(subset=["Enquiry No"], keep="first", inplace=True)
            added = len(combo) - orig
            leads_df[:] = combo
            leads_df.to_csv("leads.csv", index=False)
            st.cache_data.clear()
            log_event(current_user,"Historical Data Upload",f"{added} added")
            st.success(f"{added} new leads added.")
            st.session_state.pop("upload_historical_file", None)
            st.rerun()

        st.markdown("---")
        confirm = st.text_input(
            "Type DELETE to confirm full dashboard reset",
            key="reset_confirm"
        )
        if st.button("Reset All Dashboard Data") and confirm=="DELETE":
            pd.DataFrame(columns=leads_df.columns).to_csv("leads.csv", index=False)
            st.cache_data.clear()
            log_event(current_user,"Dashboard Reset","All leads deleted")
            st.success("Dashboard data reset.")
            st.session_state.pop("reset_confirm", None)
            st.rerun()

        st.markdown("---")
        audit = pd.read_csv("audit_logs.csv")
        st.dataframe(audit, use_container_width=True)
        st.download_button(
            "Download Audit Logs",
            audit.to_csv(index=False).encode(),
            "audit_logs.csv",
            "text/csv"
        )

        st.markdown("---")
        st.markdown("**Users**")
        st.table(users_df[["Username","Role"]])

        st.markdown("---")
        ul = pd.read_csv("user_logs.csv")
        st.dataframe(ul, use_container_width=True)
        st.download_button(
            "Download User Log",
            ul.to_csv(index=False).encode(),
            "user_logs.csv",
            "text/csv"
        )
