import os
import streamlit as st
import pandas as pd
import plotly.express as px
from plotly import graph_objects as go
from st_aggrid import AgGrid, GridOptionsBuilder
from sklearn.cluster import KMeans
from datetime import datetime

# Debug flag
DEBUG = True

# ──────────────────────────────────────────────────────────────────────────────
# Page config & CSS (including mobile responsiveness)
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
/* mobile: stack cards */
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
                 columns=["Timestamp","User","Action","Details"]) \
      .to_csv("user_logs.csv", mode="a", header=False, index=False)

def log_audit(user, action, enq, field, old, new, details=""):
    pd.DataFrame([[datetime.now(), user, action, enq, field, old, new, details]],
                 columns=["Timestamp","User","Action","Enquiry No","Field","Old Value","New Value","Details"]) \
      .to_csv("audit_logs.csv", mode="a", header=False, index=False)

# ──────────────────────────────────────────────────────────────────────────────
# Load & preprocess data
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
    leads["Enquiry Date"] = pd.to_datetime(leads["Enquiry Date"], errors="coerce")

    # drop unused columns
    drop_cols = ["Corporate Name","Tehsil","Pincode","PAN NO.",
                 "Events","Finance Required","Finance Company"]
    leads.drop(columns=[c for c in drop_cols if c in leads.columns], inplace=True)

    # ensure questionnaire columns
    for i in range(1,6):
        col = f"Question{i}"
        if col not in leads.columns:
            leads[col] = ""

    # ensure other fields
    defaults = {
        "Remarks": "", "No of Follow-ups": 0,
        "Next Action": "", "Planned Followup Date": pd.NaT
    }
    for col, default in defaults.items():
        if col not in leads.columns:
            leads[col] = default

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
# Session state & Login
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
            m = users_df[(users_df["Username"]==u) & (users_df["Password"]==p)]
            if not m.empty:
                st.session_state.update({
                    "logged_in": True,
                    "user": u,
                    "role": m.iloc[0]["Role"]
                })
                log_event(u, "Login")
                st.experimental_rerun()
            else:
                st.error("Invalid credentials")
    st.stop()

current_user = st.session_state["user"]
role         = st.session_state["role"]
# ──────────────────────────────────────────────────────────────────────────────
# Sidebar filters (in expander for mobile)
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    if st.button("Logout"):
        st.session_state.clear()
        st.experimental_rerun()

    with st.expander("Filters", expanded=True):
        st.header("Filter Leads")
        FILTERS = [
            ("State","State","f1"),
            ("City","Location","f2"),
            ("Dealer","Dealer","f3"),
            ("Employee","Employee Name","f4"),
            ("Segment","Segment","f5")
        ]
        selected = {}
        for label,col,key in FILTERS:
            vals   = sorted(leads_df[col].dropna().unique())
            choice = st.selectbox(label, ["All"]+vals, key=key)
            selected[col] = vals if choice=="All" else [choice]

        mn = int(leads_df["KVA"].min()) if pd.notna(leads_df["KVA"]).any() else 0
        mx = int(leads_df["KVA"].max()) if pd.notna(leads_df["KVA"]).any() else 1
        if mn>=mx: mx=mn+1
        kva_range = st.slider("KVA Range", mn, mx, (mn,mx), key="f6")

        today = datetime.today().date()
        first = today.replace(day=1)
        dates = st.date_input("Enquiry Date Range", [first,today], key="f7")
        if isinstance(dates,(list,tuple)) and len(dates)==2:
            start_date, end_date = dates
        else:
            start_date, end_date = first, today

# ──────────────────────────────────────────────────────────────────────────────
# Filter DataFrame
# ──────────────────────────────────────────────────────────────────────────────
filtered_df = get_filtered(leads_df, selected, kva_range, start_date, end_date)
log_event(current_user, "Filter Applied",
          f"{selected}, KVA={kva_range}, Dates={start_date}–{end_date}")

open_stages = ["Prospecting","Qualified"]
won_stages  = ["Closed-Won","Order Booked"]
lost_stages = ["Closed-Dropped","Closed-Lost"]
# ──────────────────────────────────────────────────────────────────────────────
# Main tabs (including Insights & Admin at end)
# ──────────────────────────────────────────────────────────────────────────────
tabs = ["KPI","Charts","Top Dealers","Top Employees",
        "Upload New Lead","Lead Update","Insights"]
if role=="Admin":
    tabs.append("Admin")
tabs = st.tabs(tabs)

# --- KPI Tab ---
with st.container():
    with st.spinner("Loading KPI..."):
        try:
            with tabs[0]:
                st.subheader("Key Performance Indicators")
                total    = len(filtered_df)
                open_cnt = filtered_df["Enquiry Stage"].isin(open_stages).sum()
                won_cnt  = filtered_df["Enquiry Stage"].isin(won_stages).sum()
                lost_cnt = filtered_df["Enquiry Stage"].isin(lost_stages).sum()
                conv_pct = (won_cnt/total*100) if total else 0
                closed_pct = ((won_cnt+lost_cnt)/total*100) if total else 0

                cols = st.columns(6)
                cards = [
                    ("All",   total,  "All"),
                    ("Open",  open_cnt, "Open"),
                    ("Lost",  lost_cnt, "Lost"),
                    ("Won",   won_cnt,  "Won"),
                    ("Conv %", f"{conv_pct:.1f}%", None),
                    ("Closed %", f"{closed_pct:.1f}%", None),
                ]
                for c,(title,val,drill) in zip(cols,cards):
                    if drill and c.button(f"{val}\n{title}", key=f"drill_{title}"):
                        st.session_state["drill"] = drill
                    else:
                        c.markdown(
                            f"<div class='card'><div class='card-icon'>{title}</div>"
                            f"<div class='card-value'>{val}</div>"
                            f"<div class='card-title'>{title}</div></div>",
                            unsafe_allow_html=True
                        )

                choice = st.session_state.get("drill","All")
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

        except Exception as e:
            if DEBUG:
                st.error("Error in KPI tab")
                st.exception(e)

# --- Charts Tab ---
with st.container():
    with st.spinner("Loading Charts..."):
        try:
            with tabs[1]:
                st.subheader("Leads Visualization")
                counts = filtered_df["Enquiry Stage"].value_counts().reindex(
                    open_stages + won_stages + lost_stages, fill_value=0
                )
                funnel = (
                    [counts[s] for s in open_stages]
                    + [counts[won_stages[0]]+counts[won_stages[1]]]
                    + [counts[lost_stages[0]]+counts[lost_stages[1]]]
                )
                fig = go.Figure(go.Funnel(
                    y=["Prospecting","Qualified","Won","Lost"], x=funnel
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
                st.plotly_chart(
                    px.bar(top_d, x="Dealer", y="Leads", title="Top 10 Dealers"),
                    use_container_width=True
                )

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
                    ), use_container_width=True
                )
        except Exception as e:
            if DEBUG:
                st.error("Error in Charts tab")
                st.exception(e)

# --- Top Dealers & Top Employees ---
def top5(df, by):
    agg = df.groupby(by).agg(
        Total_Leads=("Enquiry No","count"),
        Won_Leads=("Enquiry Stage", lambda x: x.isin(won_stages).sum()),
        Lost_Leads=("Enquiry Stage", lambda x: x.isin(lost_stages).sum())
    )
    agg["Conv %"] = (agg["Won_Leads"]/agg["Total_Leads"]*100).round(1)
    return agg.sort_values("Won_Leads", ascending=False).head(5).reset_index()

with st.container():
    with st.spinner("Loading Top 5..."):
        try:
            with tabs[2]:
                st.subheader("Top 5 Dealers")
                st.table(top5(filtered_df, "Dealer"))
            with tabs[3]:
                st.subheader("Top 5 Employees")
                st.table(top5(filtered_df, "Employee Name"))
        except Exception as e:
            if DEBUG:
                st.error("Error in Top Dealers/Employees tab")
                st.exception(e)
# --- Upload New Lead ---
with st.container():
    with st.spinner("Preparing Upload..."):
        try:
            with tabs[4]:
                st.subheader("Upload New Lead")
                uf = st.file_uploader("Upload leads Excel (xlsx)", type="xlsx", key="upload_new_lead_file")
                if uf:
                    df_new = pd.read_excel(uf, engine="openpyxl")
                    for i in range(1,6):
                        col = f"Question{i}"
                        if col not in df_new.columns: df_new[col] = ""
                    if "upload_idx" not in st.session_state:
                        st.session_state.upload_idx = 0
                        st.session_state.new_df   = df_new
                    idx   = st.session_state.upload_idx
                    total = len(st.session_state.new_df)

                    if idx < total:
                        lead = st.session_state.new_df.iloc[idx]
                        st.write(f"**Lead {idx+1}/{total}: {lead['Name']}**")
                        with st.form(key=f"newlead_form_{idx}"):
                            q1 = st.selectbox("Q1. Status of the site",
                                              ["under construction","nearly constructed","constructed","planning"],
                                              key=f"q1_{idx}")
                            q2 = st.selectbox("Q2. Contact person",
                                              ["Owner","Manager","Purchase Dept","Other"], key=f"q2_{idx}")
                            q3 = st.selectbox("Q3. Decision maker?", ["Yes","No"], key=f"q3_{idx}")
                            q4 = st.selectbox("Q4. Customer orientation", ["Price","Quality"], key=f"q4_{idx}")
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
                            st.experimental_rerun()
        except Exception as e:
            if DEBUG:
                st.error("Error in Upload New Lead tab")
                st.exception(e)

# --- Lead Update ---
with st.container():
    with st.spinner("Loading Lead Update..."):
        try:
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
                            new_stage = st.selectbox("Enquiry Stage", stages,
                                                     index=stages.index(row["Enquiry Stage"]),
                                                     key="update_stage")
                            new_remark = st.text_area("Remarks", value=row.get("Remarks",""), key="update_remarks")
                            new_date = st.date_input(
                                "Next Followup Date",
                                value=(row["Planned Followup Date"].date()
                                       if pd.notna(row["Planned Followup Date"])
                                       else datetime.today().date()),
                                key="update_date"
                            )
                            new_fu  = st.number_input("No of Follow-ups",
                                                     min_value=0,
                                                     value=int(row.get("No of Follow-ups",0)),
                                                     key="update_fu")
                            new_act = st.text_input("Next Action", value=row.get("Next Action",""), key="update_action")
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
                                    log_audit(current_user,"Update",enq,field,old_val,new_val,"Lead field changed")
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
                            st.success("Lead updated successfully.")
                            st.experimental_rerun()
        except Exception as e:
            if DEBUG:
                st.error("Error in Lead Update tab")
                st.exception(e)

# --- Insights Tab ---
with st.container():
    with st.spinner("Computing Insights..."):
        try:
            with tabs[6]:
                insight_tabs = st.tabs(["Dealer Segmentation","Cohort Analysis"])

                with insight_tabs[0]:
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
                    if len(stats) >= 3:
                        X = stats[["Total_Leads","Conversion %"]]
                        labels = KMeans(n_clusters=3, random_state=0).fit_predict(X)
                        stats["Cluster"] = labels.astype(str)
                        fig = px.scatter(
                            stats, x="Total_Leads", y="Conversion %",
                            color="Cluster", hover_data=["Dealer"],
                            title="Dealer Clusters by Volume & Conversion"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.dataframe(stats, use_container_width=True)
                    else:
                        st.info(">=3 dealers with ≥5 leads needed to cluster.")

                with insight_tabs[1]:
                    st.subheader("Cohort Analysis")
                    df = filtered_df[filtered_df["Enquiry Stage"].isin(won_stages)].copy()
                    df["CohortMonth"]     = df["Enquiry Date"].dt.to_period("M")
                    df["ConversionMonth"] = df["Enquiry Closure Date"].dt.to_period("M")
                    df["CohortIndex"] = (df["ConversionMonth"] - df["CohortMonth"]).apply(lambda x: x.n)
                    cc = (
                        df.groupby(["CohortMonth","CohortIndex"])["Enquiry No"]
                        .count().reset_index(name="Converted")
                    )
                    cs = (
                        filtered_df.groupby(filtered_df["Enquiry Date"].dt.to_period("M"))["Enquiry No"]
                        .count().rename("Total").reset_index()
                        .rename(columns={"Enquiry Date":"CohortMonth"})
                    )
                    cohort = pd.merge(cc, cs, on="CohortMonth")
                    cohort["ConversionRate"] = cohort["Converted"]/cohort["Total"]*100
                    pivot = cohort.pivot(index="CohortMonth", columns="CohortIndex",
                                         values="ConversionRate").fillna(0)
                    st.write("Conversion Rate (%) by Cohort Month & Months Since Enquiry")
                    st.dataframe(pivot.round(1), use_container_width=True)
                    hm = px.imshow(
                        pivot,
                        labels=dict(x="Months Since Enquiry", y="Cohort Month", color="Conv %"),
                        title="Cohort Conversion Heatmap",
                        aspect="auto"
                    )
                    st.plotly_chart(hm, use_container_width=True)
        except Exception as e:
            if DEBUG:
                st.error("Error in Insights tab")
                st.exception(e)

# --- Admin Panel ---
if role=="Admin":
    with st.container():
        with st.spinner("Loading Admin..."):
            try:
                with tabs[-1]:
                    st.subheader("Admin Panel")

                    # Historical Upload
                    hf = st.file_uploader("Upload Historical Leads", type=["xlsx","csv"], key="hist")
                    if hf and st.button("Process Historical Upload", key="hist_btn"):
                        if hf.name.endswith(".xlsx"):
                            hdf = pd.read_excel(hf, engine="openpyxl")
                        else:
                            hdf = pd.read_csv(hf)
                        orig = len(leads_df)
                        combo = pd.concat([leads_df, hdf], ignore_index=True)
                        combo.drop_duplicates(subset=["Enquiry No"], keep="first", inplace=True)
                        added = len(combo) - orig
                        combo.to_csv("leads.csv", index=False)
                        st.success(f"{added} new leads added.")
                        st.cache_data.clear()
                        log_event(current_user,"Historical Upload",f"{added} added")
                        st.experimental_rerun()

                    st.markdown("---")
                    # Reset Data
                    confirm = st.text_input("Type DELETE to confirm reset", key="rst")
                    if st.button("Reset All Data") and confirm=="DELETE":
                        pd.DataFrame(columns=leads_df.columns).to_csv("leads.csv", index=False)
                        st.success("All leads wiped.")
                        st.cache_data.clear()
                        log_event(current_user,"Dashboard Reset")
                        st.experimental_rerun()

                    st.markdown("---")
                    # Audit Logs
                    audit = pd.read_csv("audit_logs.csv")
                    st.download_button("Download Audit Logs",
                                       audit.to_csv(index=False).encode(), "audit_logs.csv")
                    st.dataframe(audit, use_container_width=True)

                    st.markdown("---")
                    # User Management
                    st.subheader("Users")
                    st.dataframe(users_df[["Username","Role"]], use_container_width=True)

                    with st.form("add_user"):
                        nu = st.text_input("New Username")
                        np = st.text_input("New Password", type="password")
                        nr = st.selectbox("Role", ["Admin","Manager","Employee"])
                        if st.form_submit_button("Add User"):
                            if nu and np:
                                if nu in users_df["Username"].values:
                                    st.warning("Username exists.")
                                else:
                                    users_df.loc[len(users_df)] = [nu,np,nr]
                                    users_df.to_csv("users.csv", index=False)
                                    log_event(current_user,"User Added",nu)
                                    st.success(f"User '{nu}' added.")
                                    st.experimental_rerun()

                    del_sel = st.multiselect("Delete Users",
                                             [u for u in users_df["Username"] if u!=current_user])
                    if st.button("Delete Selected Users"):
                        if del_sel:
                            users_df = users_df[~users_df["Username"].isin(del_sel)]
                            users_df.to_csv("users.csv", index=False)
                            log_event(current_user,"User Deleted",",".join(del_sel))
                            st.success(f"Deleted: {', '.join(del_sel)}")
                            st.experimental_rerun()

                    st.markdown("---")
                    # User Activity
                    ul = pd.read_csv("user_logs.csv")
                    st.download_button("Download User Log",
                                       ul.to_csv(index=False).encode(), "user_logs.csv")
                    st.dataframe(ul, use_container_width=True)
            except Exception as e:
                if DEBUG:
                    st.error("Error in Admin tab")
                    st.exception(e)
