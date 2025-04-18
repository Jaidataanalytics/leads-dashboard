import os
import streamlit as st
import pandas as pd
import plotly.express as px
from plotly import graph_objects as go
from datetime import datetime
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid import AgGrid, GridOptionsBuilder



# -----------------------------------------
# Setup & Logging
# -----------------------------------------
def ensure_file(path, cols):
    if not os.path.exists(path):
        pd.DataFrame(columns=cols).to_csv(path, index=False)

ensure_file("user_logs.csv", ["Timestamp","User","Action","Details"])
ensure_file("audit_logs.csv", ["Timestamp","User","Action","Enquiry No","Field","Old Value","New Value","Details"])

def log_event(user, action, details=""):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pd.DataFrame([[ts, user, action, details]],
                 columns=["Timestamp","User","Action","Details"])\
      .to_csv("user_logs.csv", mode="a", header=False, index=False)

def log_audit(user, action, enq, field, old, new, details=""):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pd.DataFrame([[ts, user, action, enq, field, old, new, details]],
                 columns=["Timestamp","User","Action","Enquiry No","Field","Old Value","New Value","Details"])\
      .to_csv("audit_logs.csv", mode="a", header=False, index=False)

st.set_page_config(page_title="Leads Dashboard", layout="wide")
st.markdown("""
<style>
.cards-container {
  display: flex;
  flex-wrap: nowrap;
  gap: 16px;
  margin-bottom: 16px;
  overflow-x: auto;
}
.card {
  flex: 0 0 auto;
  padding: 16px;
  border-radius: 12px;
  background-color: #000000;
  color: #ffffff;
  box-shadow: 0 4px 12px rgba(0,0,0,0.3);
  width: 150px;
  text-align: center;
}
.card-icon { font-size: 32px; margin-bottom: 8px; }
.card-value{ font-size: 24px; font-weight: 600; }
.card-title{ font-size: 12px; color: #dddddd; }
</style>
""", unsafe_allow_html=True)




# -----------------------------------------
# Load Data
# -----------------------------------------
@st.cache_data
def load_data():
    leads = pd.read_csv(
        "leads.csv",
        dtype={"KVA": float},
        parse_dates=["Enquiry Date","Planned Followup Date","Enquiry Closure Date"],
        dayfirst=True, keep_default_na=False
    )
    leads = leads.loc[:, ~leads.columns.duplicated()]
    leads["Enquiry Date"] = pd.to_datetime(leads["Enquiry Date"], errors="coerce")
    # drop unused columns
    for c in ["Corporate Name","Tehsil","Pincode","PAN NO.","Events","Finance Required","Finance Company"]:
        if c in leads: leads.drop(columns=c, inplace=True)
    # ensure questionnaire columns exist
    for i in range(1,6):
        col = f"Question{i}"
        if col not in leads:
            leads[col] = ""
    users = pd.read_csv("users.csv")
    return leads, users

leads_df, users_df = load_data()

# -----------------------------------------
# Session State
# -----------------------------------------
# ──────────────────────────────────────────────────────────────────────────────
# Centered login that submits on Enter in the password field
# ──────────────────────────────────────────────────────────────────────────────
if not st.session_state.get("logged_in", False):
    # define callback to mark an "enter" press
    def _on_enter():
        st.session_state.trigger_login = True

    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.title("Please log in")
        # Username input (no callback here)
        u = st.text_input("Username", key="login_user")
        # Password input: on_change fires when you hit Enter
        p = st.text_input(
            "Password",
            type="password",
            key="login_pass",
            on_change=_on_enter
        )
        # The normal button also works
        btn = st.button("Login")
        # Determine if we should attempt login
        if btn or st.session_state.pop("trigger_login", False):
            # clear any old trigger
            st.session_state.pop("trigger_login", None)
            # perform authentication
            m = users_df[
                (users_df["Username"] == u)
                & (users_df["Password"] == p)
            ]
            if not m.empty:
                st.session_state["logged_in"] = True
                st.session_state["user"]      = u
                st.session_state["role"]      = m.iloc[0]["Role"]
                log_event(u, "Login")
                st.rerun()
            else:
                st.error("Invalid credentials")
    st.stop()
# ──────────────────────────────────────────────────────────────────────────────



#################
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
current_user = st.session_state.get("user")
role         = st.session_state.get("role")

# -----------------------------------------
# Sidebar: Login & Filters
# -----------------------------------------
with st.sidebar:
    if st.session_state["logged_in"]:
        if st.button("Logout"):
            for k in ["logged_in","user","role"]:
                st.session_state.pop(k)
            st.experimental_rerun()

   

    st.header("Filters")
    def selbox(label, col, key):
        vals = sorted(leads_df[col].dropna().unique())
        choice = st.selectbox(label, ["All"]+vals, key=key)
        return vals if choice=="All" else [choice]

    state_filter  = selbox("State","State","f1")
    city_filter   = selbox("City","Location","f2")
    dealer_filter = selbox("Dealer","Dealer","f3")
    emp_filter    = selbox("Employee","Employee Name","f4")
    seg_filter    = selbox("Segment","Segment","f5")

    if pd.notna(leads_df["KVA"]).any():
        mn, mx = int(leads_df["KVA"].min()), int(leads_df["KVA"].max())
    else:
        mn, mx = 0, 1
    if mn>=mx: mx = mn+1
    kva_range = st.slider("KVA Range", mn, mx, (mn,mx), key="f6")

    today = datetime.today().date()
    first = today.replace(day=1)
    dates = st.date_input("Enquiry Date Range",[first,today], key="f7")
    if isinstance(dates,(list,tuple)) and len(dates)==2:
        start_date, end_date = dates
    else:
        start_date, end_date = first, today

# -----------------------------------------
# Apply Filters
# -----------------------------------------
@st.cache_data
def get_filtered(df, sf, cf, df_, ef, sf2, kva, sd, ed):
    d = df.copy()
    d = d[d["State"].isin(sf)]
    d = d[d["Location"].isin(cf)]
    d = d[d["Dealer"].isin(df_)]
    d = d[d["Employee Name"].isin(ef)]
    d = d[d["Segment"].isin(sf2)]
    d = d[(d["KVA"]>=kva[0])&(d["KVA"]<=kva[1])]
    d = d[(d["Enquiry Date"].dt.date>=sd)&(d["Enquiry Date"].dt.date<=ed)]
    return d

filtered_df = get_filtered(
    leads_df, state_filter, city_filter,
    dealer_filter, emp_filter, seg_filter,
    kva_range, start_date, end_date
)
log_event(current_user,"Filter Applied",
          f"States={state_filter},Cities={city_filter},Dealers={dealer_filter},"
          f"Employees={emp_filter},Segments={seg_filter},KVA={kva_range},"
          f"Date={start_date} to {end_date}")

# -----------------------------------------
# Stage Categories
# -----------------------------------------
open_stages = ["Prospecting","Qualified"]
won_stages  = ["Closed-Won","Order Booked"]
lost_stages = ["Closed-Dropped","Closed-Lost"]

# -----------------------------------------
# Tabs
# -----------------------------------------
tabs = ["KPI","Charts","Top Dealers","Top Employees","Upload New Lead","Lead Update"]
if role=="Admin":
    tabs.append("Admin")
tabs = st.tabs(tabs)

# --- KPI Tab ---
# --- KPI Tab (as cards) ---
with tabs[0]:
    st.subheader("Key Performance Indicators")

    # compute metrics
    total     = len(filtered_df)
    open_cnt  = len(filtered_df[filtered_df["Enquiry Stage"].isin(open_stages)])
    won_cnt   = len(filtered_df[filtered_df["Enquiry Stage"].isin(won_stages)])
    lost_cnt  = len(filtered_df[filtered_df["Enquiry Stage"].isin(lost_stages)])
    conv_pct  = (won_cnt/total*100)    if total else 0
    closed_pct= ((won_cnt+lost_cnt)/total*100) if total else 0

    # prepare card data
    cards = [
      ("📈","Total Leads",  total),
      ("🕒","Open Leads",   open_cnt),
      ("❌","Lost Leads",   lost_cnt),
      ("🏆","Won Leads",    won_cnt),
      ("🔄","Conversion %", f"{conv_pct:.1f}%"),
      ("✅","Closed %",     f"{closed_pct:.1f}%"),
    ]

    # build one HTML block for all cards
    html = '<div class="cards-container">'
    for icon, title, val in cards:
        html += f'''
          <div class="card">
            <div class="card-icon">{icon}</div>
            <div class="card-value">{val}</div>
            <div class="card-title">{title}</div>
          </div>'''
    html += '</div>'

    # render the cards row
    st.markdown(html, unsafe_allow_html=True)

    # drill‑down table selector
    choice = st.radio(
        "View details for:",
        ["All","Open","Lost","Won"],
        horizontal=True,
        key="kpi_drill"
    )

    if choice == "All":
        ddf = filtered_df
    elif choice == "Open":
        ddf = filtered_df[filtered_df["Enquiry Stage"].isin(open_stages)]
    elif choice == "Lost":
        ddf = filtered_df[filtered_df["Enquiry Stage"].isin(lost_stages)]
    else:
        ddf = filtered_df[filtered_df["Enquiry Stage"].isin(won_stages)]

    # display with AgGrid
    gb = GridOptionsBuilder.from_dataframe(ddf)
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_default_column(enableValue=True, sortable=True, filter=True)

    AgGrid(
        ddf,
        gridOptions=gb.build(),
        enable_enterprise_modules=False,
        theme="dark"   # or remove theme argument if you want default
    )


# --- Charts Tab ---
with tabs[1]:
    st.subheader("Leads Visualization")
    # Funnel
    fc = filtered_df["Enquiry Stage"].value_counts().reindex(
        ["Prospecting","Qualified","Closed-Won","Order Booked","Closed-Dropped","Closed-Lost"]
    ).fillna(0)
    fig_f = go.Figure(go.Funnel(
        y=["Prospecting","Qualified","Won","Lost"],
        x=[fc["Prospecting"], fc["Qualified"],
           fc["Closed-Won"]+fc["Order Booked"],
           fc["Closed-Dropped"]+fc["Closed-Lost"]]
    ))
    fig_f.update_layout(title="Lead Pipeline Funnel")
    st.plotly_chart(fig_f, use_container_width=True)

    # Top Dealers Bar
    dc = filtered_df["Dealer"].value_counts().reset_index()
    dc.columns=["Dealer","Leads"]
    fig_b = px.bar(dc.head(10), x="Dealer", y="Leads", title="Top 10 Dealers by Leads")
    st.plotly_chart(fig_b, use_container_width=True)

    # Time Series
    agg = st.selectbox("Time Series Granularity", ["Daily","Weekly","Monthly"], key="ts_gran")
    fmap = {"Daily":"D","Weekly":"W","Monthly":"M"}
    wmap = {"D":7,"W":4,"M":3}
    freq, win = fmap[agg], wmap[fmap[agg]]
    ts = (filtered_df.set_index("Enquiry Date")
          .resample(freq).size().rename("Leads").to_frame().reset_index())
    ts[f"MA({win})"] = ts["Leads"].rolling(win, min_periods=1).mean()
    fig_ts = px.line(ts, x="Enquiry Date", y=["Leads", f"MA({win})"],
                     labels={"value":"Count","variable":"Metric"},
                     title=f"{agg} Leads & {win}-Period MA")
    st.plotly_chart(fig_ts, use_container_width=True)

# --- Top Dealers ---
@st.cache_data
def top5_dealers(df):
    dg = df.groupby("Dealer").agg(
        Total_Leads=("Enquiry No","count"),
        Won_Leads=("Enquiry Stage", lambda x: x.isin(won_stages).sum()),
        Lost_Leads=("Enquiry Stage", lambda x: x.isin(lost_stages).sum())
    )
    dg["Conversion %"] = (dg["Won_Leads"]/dg["Total_Leads"]*100).round(1)
    return dg.sort_values("Won_Leads", ascending=False).head(5).reset_index()

with tabs[2]:
    st.subheader("Top 5 Dealers")
    st.table(top5_dealers(filtered_df))

# --- Top Employees ---
@st.cache_data
def top5_employees(df):
    eg = df.groupby("Employee Name").agg(
        Total_Leads=("Enquiry No","count"),
        Won_Leads=("Enquiry Stage", lambda x: x.isin(won_stages).sum()),
        Lost_Leads=("Enquiry Stage", lambda x: x.isin(lost_stages).sum())
    )
    eg["Conversion %"] = (eg["Won_Leads"]/eg["Total_Leads"]*100).round(1)
    return eg.sort_values("Won_Leads", ascending=False).head(5).reset_index()

with tabs[3]:
    st.subheader("Top 5 Employees")
    st.table(top5_employees(filtered_df))

# --- Upload New Lead ---
with tabs[4]:
    st.subheader("Upload New Lead")
    upload_file = st.file_uploader("Upload leads Excel (xlsx)", type="xlsx", key="upl_new")
    if upload_file:
        df_new = pd.read_excel(upload_file, engine="openpyxl")
        for i in range(1,6):
            col = f"Question{i}"
            if col not in df_new: df_new[col] = ""
        if "upload_idx" not in st.session_state:
            st.session_state.upload_idx = 0
            st.session_state.new_df = df_new
        idx = st.session_state.upload_idx
        if idx < len(st.session_state.new_df):
            lead = st.session_state.new_df.iloc[idx]
            st.write(f"**Lead {idx+1}/{len(st.session_state.new_df)}: {lead['Name']}**")
            q1 = st.selectbox("Q1. Status of the site",
                              ["under construction","nearly constructed","constructed","planning"],
                              key=f"q1_{idx}")
            q2 = st.selectbox("Q2. Contact person",
                              ["Owner","Manager","Purchase Dept","Other"], key=f"q2_{idx}")
            q3 = st.selectbox("Q3. Decision maker?",
                              ["Yes","No"], key=f"q3_{idx}")
            q4 = st.selectbox("Q4. Customer orientation",
                              ["Price","Quality"], key=f"q4_{idx}")
            q5 = st.selectbox("Q5. Who decides?",
                              ["contact person","owner","manager","purchase head"], key=f"q5_{idx}")
            if st.button("Submit Lead", key=f"sub_{idx}"):
                name, phone = lead["Name"], lead["Phone Number"]
                exists = ((leads_df["Name"]==name)&(leads_df["Phone Number"]==phone)).any()
                if exists:
                    st.warning(f"Lead '{name}' exists; skipped.")
                else:
                    for i, ans in enumerate((q1,q2,q3,q4,q5), start=1):
                        log_audit(current_user, "Create", lead["Enquiry No"],
                                  f"Question{i}", "", ans, "Questionnaire answer")
                        st.session_state.new_df.at[idx, f"Question{i}"] = ans
                    entry = st.session_state.new_df.loc[idx].copy()
                    entry["Created By"] = current_user
                    leads_df.loc[len(leads_df)] = entry
                    leads_df.to_csv("leads.csv", index=False)
                    log_event(current_user, "New Lead Uploaded", name)
                    st.success(f"Lead '{name}' added.")
                st.session_state.upload_idx += 1
                st.experimental_rerun()

# --- Lead Update ---
with tabs[5]:
    st.subheader("Lead Update")
    open_df = filtered_df[filtered_df["Enquiry Stage"].isin(open_stages)]
    if open_df.empty:
        st.info("No open leads.")
    else:
        search = st.text_input("Search Lead (Name or Enq No)", key="lu_search")
        opts = (open_df["Enquiry No"].astype(str)+" - "+open_df["Name"]).tolist()
        if search:
            opts = [o for o in opts if search.lower() in o.lower()]
        if not opts:
            st.warning("No leads found.")
        else:
            sel = st.selectbox("Select Lead", opts, key="lu_select")
            enq = sel.split(" - ",1)[0]
            row = open_df[open_df["Enquiry No"].astype(str)==enq].iloc[0]
            idx = row.name
            with st.form("upd_form"):
                new_stage = st.selectbox("Enquiry Stage",
                    ["Prospecting","Qualified","Closed-Dropped","Closed-Lost","Closed-Won","Order Booked"],
                    index=["Prospecting","Qualified","Closed-Dropped","Closed-Lost","Closed-Won","Order Booked"].index(row["Enquiry Stage"]))
                new_remark = st.text_area("Remarks", value=row.get("Remarks",""))
                new_date   = st.date_input("Next Follow-up Date",
                    value=(row["Planned Followup Date"].date() if pd.notna(row["Planned Followup Date"])
                           else datetime.today().date()))
                new_fu     = st.number_input("No of Follow-ups", min_value=0,
                                             value=int(row.get("No of Follow-ups",0)), step=1)
                new_act    = st.text_input("Next Action", value=row.get("Next Action",""))
                if st.form_submit_button("Save Changes"):
                    fields = {
                        "Enquiry Stage": new_stage,
                        "Remarks": new_remark,
                        "Planned Followup Date": pd.to_datetime(new_date),
                        "No of Follow-ups": new_fu,
                        "Next Action": new_act
                    }
                    for field, new_val in fields.items():
                        old_val = leads_df.at[idx, field]
                        if (pd.isna(old_val) and new_val is not None) or (old_val != new_val):
                            log_audit(current_user, "Update", enq, field, old_val, new_val, "Lead field changed")
                            leads_df.at[idx, field] = new_val
                    if new_stage in won_stages:
                        leads_df.at[idx,"EnquiryStatus"]="Converted"
                        leads_df.at[idx,"Enquiry Closure Date"]=datetime.now()
                    if new_stage in lost_stages:
                        leads_df.at[idx,"EnquiryStatus"]="Closed"
                        leads_df.at[idx,"Enquiry Closure Date"]=datetime.now()
                    leads_df.to_csv("leads.csv", index=False)
                    st.cache_data.clear() 
                    log_event(current_user, "Lead Updated", f"{enq} -> {new_stage}")
                    st.success("Lead updated successfully.")
                    st.experimental_rerun()

# --- Admin Panel ---
if role=="Admin":
    with tabs[-1]:
        st.subheader("Admin Panel")
        # Reset Data
        if st.button("Reset All Dashboard Data"):
            confirm = st.text_input("Type DELETE to confirm reset", key="reset_confirm")
            if confirm=="DELETE":
                st.cache_data.clear()
                pd.DataFrame(columns=leads_df.columns).to_csv("leads.csv", index=False)
                log_event(current_user,"Dashboard Reset","All leads deleted")
                st.success("Dashboard data reset.")
                st.experimental_rerun()
        st.markdown("---")
        # Historical Upload
        hf = st.file_uploader("Upload Historical Leads (xlsx/csv)", type=["xlsx","csv"], key="hist")
        if hf:
            if hf.name.endswith(".xlsx"):
                hdf = pd.read_excel(hf, engine="openpyxl")
            else:
                hdf = pd.read_csv(hf)
            orig = len(leads_df)
            combo = pd.concat([leads_df, hdf], ignore_index=True)
            combo.drop_duplicates(subset=["Enquiry No"], keep="first", inplace=True)
            added = len(combo) - orig
            leads_df[:] = combo
            leads_df.to_csv("leads.csv", index=False)
            log_event(current_user,"Historical Data Upload",f"{added} added")
            st.success(f"{added} new leads added.")
            st.rerun()
        st.markdown("---")
        # Audit Logs
        audit = pd.read_csv("audit_logs.csv")
        st.dataframe(audit, use_container_width=True)
        st.download_button("Download Audit Logs", audit.to_csv(index=False).encode(), "audit_logs.csv", "text/csv")
        st.markdown("---")
        # User Management
        st.markdown("**Users**")
        st.table(users_df[["Username","Role"]])
        # (Add/Edit/Delete user logic here)
        st.markdown("---")
        # User Activity Log
        ul = pd.read_csv("user_logs.csv")
        st.dataframe(ul, use_container_width=True)
        st.download_button("Download User Log", ul.to_csv(index=False).encode(), "user_logs.csv", "text/csv")
