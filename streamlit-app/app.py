import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CRM Lead Scoring",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR  = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data" / "Leads.csv"

# ── Constants ─────────────────────────────────────────────────────────────────
BINARY_COLS = [
    "Do Not Email", "Do Not Call", "Search", "Magazine",
    "Newspaper Article", "X Education Forums", "Newspaper",
    "Digital Advertisement", "Through Recommendations",
    "A free copy of Mastering The Interview",
]

CAT_COLS = [
    "Lead Source", "Last Activity", "Country", "Specialization",
    "What is your current occupation",
    "What matters most to you in choosing a course",
    "Tags", "Lead Profile", "City", "Last Notable Activity",
]

DROP_COLS = [
    "Prospect ID", "Lead Number", "Lead Quality",
    "Asymmetrique Activity Index", "Asymmetrique Profile Index",
    "How did you hear about X Education",
    "I agree to pay the amount through cheque",
    "Get updates on DM Content", "Update me on Supply Chain Content",
    "Receive More Updates About Our Courses",
]

OHE_COLS = [
    "Lead Origin", "Lead Source", "Last Activity", "Country",
    "Specialization", "What is your current occupation",
    "What matters most to you in choosing a course",
    "Tags", "Lead Profile", "City", "Lead_Source_Group",
    "Last Notable Activity",
]

ENGAGEMENT_ORDER = {
    "No Engagement": 0, "Low (0-5 min)": 1,
    "Medium (5-15 min)": 2, "High (15+ min)": 3,
}

TIER_CONFIG = {
    "🔥 Hot":   {"min": 0.75, "color": "#e74c3c", "bg": "#fdecea"},
    "🌡️ Warm":  {"min": 0.50, "color": "#f39c12", "bg": "#fef9e7"},
    "❄️ Cold":  {"min": 0.00, "color": "#3498db", "bg": "#ebf5fb"},
}


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop meta columns if present
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)
    # Drop target if present
    df.drop(columns=["Converted"], errors="ignore", inplace=True)

    # Fill numeric
    for col in ["TotalVisits", "Page Views Per Visit",
                "Asymmetrique Activity Score", "Asymmetrique Profile Score"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Fill binary
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("No")

    # Fill categorical
    for col in CAT_COLS:
        if col in df.columns:
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val[0] if len(mode_val) > 0 else "Unknown")

    # Replace 'Select' with NaN then re-fill
    df = df.replace("Select", np.nan)
    for col in df.select_dtypes(include="object").columns:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val[0] if len(mode_val) > 0 else "Unknown")

    # Feature engineering
    def engagement_level(t):
        if t == 0:      return "No Engagement"
        elif t <= 300:  return "Low (0-5 min)"
        elif t <= 900:  return "Medium (5-15 min)"
        else:           return "High (15+ min)"

    if "Total Time Spent on Website" in df.columns:
        df["Engagement_Level"] = df["Total Time Spent on Website"].apply(engagement_level)
    else:
        df["Engagement_Level"] = "No Engagement"

    if "TotalVisits" in df.columns:
        df["High_Interaction"] = (df["TotalVisits"] >= 5).astype(int)
    else:
        df["High_Interaction"] = 0

    def group_lead_source(src):
        if src in ["Google", "Organic Search", "Bing"]:              return "Search Engine"
        elif src in ["Direct Traffic", "NC_EDM"]:                     return "Direct"
        elif src in ["Olark Chat", "Live Chat"]:                      return "Chat"
        elif src in ["WeLearn", "Welingak Website"]:                  return "Partner"
        elif src in ["Reference", "Through Recommendations"]:         return "Referral"
        elif src in ["Social Media", "Facebook", "youtube"]:          return "Social"
        else:                                                          return "Other"

    if "Lead Source" in df.columns:
        df["Lead_Source_Group"] = df["Lead Source"].apply(group_lead_source)
        df["Is_Webinar_Lead"] = (df["Lead Source"] == "Webinar").astype(int)
    else:
        df["Lead_Source_Group"] = "Other"
        df["Is_Webinar_Lead"] = 0

    df["Engagement_Score"] = (
        df.get("Total Time Spent on Website", pd.Series(0, index=df.index)) * 0.5
        + df.get("TotalVisits", pd.Series(0, index=df.index)) * 10
        + df.get("Page Views Per Visit", pd.Series(0, index=df.index)) * 5
    )

    # Encode binary Yes/No
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

    # Encode engagement level
    df["Engagement_Level"] = df["Engagement_Level"].map(ENGAGEMENT_ORDER).fillna(0).astype(int)

    # One-hot encode (drop_first=False — align_to_model selects the right columns)
    df = pd.get_dummies(df, columns=[c for c in OHE_COLS if c in df.columns], drop_first=False)

    return df


def align_to_model(df: pd.DataFrame, model_cols: list) -> pd.DataFrame:
    """Add missing columns as 0, drop extra columns, reorder."""
    for col in model_cols:
        if col not in df.columns:
            df[col] = 0
    return df[model_cols]


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    model  = joblib.load(MODEL_DIR / "lead_scoring_model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    cols   = joblib.load(MODEL_DIR / "model_columns.pkl")
    return model, scaler, cols


# ── Scoring helper ────────────────────────────────────────────────────────────
def score_df(df_raw: pd.DataFrame, model, scaler, model_cols) -> pd.DataFrame:
    processed = preprocess(df_raw)
    aligned   = align_to_model(processed, model_cols)
    scaled    = scaler.transform(aligned)
    probs     = model.predict_proba(scaled)[:, 1]

    out = df_raw.copy().reset_index(drop=True)
    out["Conversion Probability"] = probs
    out["Score (%)"] = (probs * 100).round(1)
    out["Tier"] = pd.cut(
        probs,
        bins=[-0.001, 0.50, 0.75, 1.001],
        labels=["❄️ Cold", "🌡️ Warm", "🔥 Hot"],
    )
    return out


def tier_color(tier: str) -> str:
    for t, cfg in TIER_CONFIG.items():
        if t == tier:
            return cfg["color"]
    return "#888"


# ── UI helpers ────────────────────────────────────────────────────────────────
def metric_card(label, value, color="#1a73e8"):
    st.markdown(
        f"""
        <div style="background:#f8f9fa;border-left:4px solid {color};
                    padding:12px 16px;border-radius:6px;margin-bottom:8px;">
          <div style="font-size:0.78rem;color:#6c757d;font-weight:600;
                      text-transform:uppercase;letter-spacing:0.05em">{label}</div>
          <div style="font-size:1.6rem;font-weight:700;color:#212529">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def tier_badge(tier: str) -> str:
    cfg = next((v for k, v in TIER_CONFIG.items() if k == tier), None)
    if cfg is None:
        return tier
    return (
        f'<span style="background:{cfg["bg"]};color:{cfg["color"]};'
        f'border:1px solid {cfg["color"]};padding:2px 8px;'
        f'border-radius:12px;font-weight:600;font-size:0.82rem">{tier}</span>'
    )


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🎯 Lead Scoring")
    st.caption("CRM Conversion Predictor")
    st.divider()

    st.subheader("Data Source")
    use_upload = st.radio(
        "Choose input",
        ["Use project CSV", "Upload my own CSV"],
        index=0,
    )

    uploaded_file = None
    if use_upload == "Upload my own CSV":
        uploaded_file = st.file_uploader(
            "Upload Leads CSV", type=["csv"],
            help="Must have the same column structure as the original Leads.csv"
        )

    st.divider()
    st.subheader("Score Threshold")
    hot_thresh  = st.slider("🔥 Hot (≥)", 0.50, 0.99, 0.75, 0.01)
    warm_thresh = st.slider("🌡️ Warm (≥)", 0.10, float(hot_thresh) - 0.01, 0.50, 0.01)

    st.divider()
    st.caption("Model: Logistic Regression  \nAUC-ROC: 0.951")


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Reading CSV…")
def load_csv(path_or_bytes, is_bytes=False):
    if is_bytes:
        return pd.read_csv(io.BytesIO(path_or_bytes))
    return pd.read_csv(path_or_bytes)


if use_upload == "Upload my own CSV":
    if uploaded_file is None:
        st.info("⬆️  Upload a CSV in the sidebar to get started.")
        st.stop()
    raw_df = load_csv(uploaded_file.read(), is_bytes=True)
else:
    if not DATA_PATH.exists():
        st.error(f"Project CSV not found at `{DATA_PATH}`. Upload one via the sidebar instead.")
        st.stop()
    raw_df = load_csv(str(DATA_PATH))


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATE UPLOADED CSV
# ══════════════════════════════════════════════════════════════════════════════
CRITICAL_COLS = [
    "TotalVisits", "Total Time Spent on Website", "Page Views Per Visit",
]
IMPORTANT_COLS = [
    "Lead Origin", "Lead Source", "Last Activity", "Country",
    "Do Not Email", "Do Not Call", "Specialization",
    "What is your current occupation", "Tags", "Last Notable Activity",
    "Asymmetrique Activity Score", "Asymmetrique Profile Score",
]

if use_upload == "Upload my own CSV":
    uploaded_cols  = set(raw_df.columns)
    missing_critical  = [c for c in CRITICAL_COLS  if c not in uploaded_cols]
    missing_important = [c for c in IMPORTANT_COLS if c not in uploaded_cols]
    present_important = len(IMPORTANT_COLS) - len(missing_important)
    coverage = round((len(CRITICAL_COLS) - len(missing_critical) + present_important)
                     / (len(CRITICAL_COLS) + len(IMPORTANT_COLS)) * 100)

    if missing_critical:
        st.error(
            f"**Upload blocked.** The following critical columns are missing — "
            f"scores cannot be computed without them:\n\n"
            + "\n".join(f"- `{c}`" for c in missing_critical)
        )
        st.info("These columns drive the model's engagement features. "
                "Please check your CSV structure and re-upload.")
        st.stop()

    if missing_important:
        st.warning(
            f"**Partial match ({coverage}% column coverage).** "
            f"{len(missing_important)} important column(s) not found — "
            f"scores will be less accurate:\n\n"
            + "\n".join(f"- `{c}`" for c in missing_important)
            + "\n\nScoring will continue with these set to neutral defaults."
        )
    else:
        st.success(f"✅ CSV validated — all expected columns present ({len(raw_df):,} rows loaded).")


# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════════════════════════════
try:
    model, scaler, model_cols = load_model()
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.info("Make sure the `models/` folder contains `lead_scoring_model.pkl`, `scaler.pkl`, and `model_columns.pkl`.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# SCORE
# ══════════════════════════════════════════════════════════════════════════════
has_target = "Converted" in raw_df.columns

with st.spinner("Scoring leads…"):
    scored = score_df(raw_df, model, scaler, model_cols)

# Apply sidebar thresholds
def apply_thresholds(prob, hot, warm):
    if prob >= hot:  return "🔥 Hot"
    if prob >= warm: return "🌡️ Warm"
    return "❄️ Cold"

scored["Tier"] = scored["Conversion Probability"].apply(
    lambda p: apply_thresholds(p, hot_thresh, warm_thresh)
)


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📋 All Predictions", "🔍 Single Lead", "☁️ Salesforce"])


# ─── TAB 1: Overview ─────────────────────────────────────────────────────────
with tab1:
    st.header("Dataset Overview")

    n_hot  = (scored["Tier"] == "🔥 Hot").sum()
    n_warm = (scored["Tier"] == "🌡️ Warm").sum()
    n_cold = (scored["Tier"] == "❄️ Cold").sum()
    total  = len(scored)

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Total Leads",  f"{total:,}",        "#1a73e8")
    with c2: metric_card("🔥 Hot",       f"{n_hot:,}",        "#e74c3c")
    with c3: metric_card("🌡️ Warm",      f"{n_warm:,}",       "#f39c12")
    with c4: metric_card("❄️ Cold",      f"{n_cold:,}",       "#3498db")

    st.divider()

    col_left, col_right = st.columns(2)

    # Distribution histogram
    with col_left:
        st.subheader("Probability Distribution")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(scored["Conversion Probability"], bins=40, color="#1a73e8", alpha=0.75, edgecolor="white")
        ax.axvline(x=hot_thresh,  color="#e74c3c", linestyle="--", linewidth=1.5, label=f"Hot ≥ {hot_thresh}")
        ax.axvline(x=warm_thresh, color="#f39c12", linestyle="--", linewidth=1.5, label=f"Warm ≥ {warm_thresh}")
        ax.set_xlabel("Conversion Probability")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Tier donut
    with col_right:
        st.subheader("Lead Tiers")
        sizes  = [n_hot, n_warm, n_cold]
        labels = ["🔥 Hot", "🌡️ Warm", "❄️ Cold"]
        colors = ["#e74c3c", "#f39c12", "#3498db"]
        non_zero = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0]
        if non_zero:
            s_nz, l_nz, c_nz = zip(*non_zero)
            fig2, ax2 = plt.subplots(figsize=(5, 3.5))
            wedges, _, autotexts = ax2.pie(
                s_nz, labels=None, colors=c_nz,
                autopct="%1.1f%%", startangle=90,
                wedgeprops=dict(width=0.55),
                pctdistance=0.75,
            )
            for at in autotexts:
                at.set_fontsize(9)
            ax2.legend(wedges, l_nz, loc="lower center",
                       ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.08))
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

    # Actual conversion rate if target column exists
    if has_target:
        st.divider()
        st.subheader("Actual Conversion by Tier")
        merged = scored.copy()
        merged["Converted"] = raw_df["Converted"].values
        rate = merged.groupby("Tier")["Converted"].mean().reset_index()
        rate.columns = ["Tier", "Actual Conversion Rate"]
        rate["Actual Conversion Rate"] = (rate["Actual Conversion Rate"] * 100).round(1)
        st.dataframe(
            rate.sort_values("Actual Conversion Rate", ascending=False),
            use_container_width=True, hide_index=True,
        )


# ─── TAB 2: All Predictions ───────────────────────────────────────────────────
with tab2:
    st.header("All Lead Predictions")

    # Filters
    f1, f2, f3 = st.columns([1, 1, 2])
    with f1:
        tier_filter = st.multiselect(
            "Filter by Tier",
            ["🔥 Hot", "🌡️ Warm", "❄️ Cold"],
            default=["🔥 Hot", "🌡️ Warm", "❄️ Cold"],
        )
    with f2:
        min_score = st.number_input("Min Score (%)", 0.0, 100.0, 0.0, 1.0)
    with f3:
        search_col = "Prospect ID" if "Prospect ID" in raw_df.columns else raw_df.columns[0]
        search_val = st.text_input(f"Search by {search_col}", "")

    view = scored[scored["Tier"].isin(tier_filter)].copy()
    view = view[view["Score (%)"] >= min_score]
    if search_val:
        view = view[view[search_col].astype(str).str.contains(search_val, case=False, na=False)]

    view = view.sort_values("Conversion Probability", ascending=False)

    # Display columns
    display_cols = ["Score (%)", "Tier"]
    if "Prospect ID" in view.columns:      display_cols = ["Prospect ID"] + display_cols
    if "Lead Source" in view.columns:      display_cols.append("Lead Source")
    if "Last Activity" in view.columns:    display_cols.append("Last Activity")
    if "Country" in view.columns:          display_cols.append("Country")
    if has_target and "Converted" in raw_df.columns:
        view["Actual"] = raw_df.loc[view.index, "Converted"].values
        display_cols.append("Actual")

    st.caption(f"Showing {len(view):,} of {total:,} leads")
    st.dataframe(view[display_cols].reset_index(drop=True), use_container_width=True, height=450)

    # Download
    csv_bytes = view[display_cols].to_csv(index=False).encode()
    st.download_button(
        "⬇️  Download filtered results as CSV",
        data=csv_bytes,
        file_name="scored_leads.csv",
        mime="text/csv",
    )


# ─── TAB 3: Single Lead ──────────────────────────────────────────────────────
with tab3:
    st.header("Score a Single Lead")
    st.caption("Fill in the fields below and click **Score Lead** to get an instant prediction.")

    with st.form("single_lead_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            lead_origin = st.selectbox("Lead Origin", [
                "API", "Landing Page Submission", "Lead Add Form",
                "Lead Import", "Quick Add Form",
            ])
            lead_source = st.selectbox("Lead Source", [
                "Google", "Direct Traffic", "Olark Chat", "Organic Search",
                "Reference", "Welingak Website", "WeLearn", "Live Chat",
                "Facebook", "Social Media", "NC_EDM", "Pay per Click Ads",
                "bing", "blog", "Press_Release", "Other",
            ])
            total_visits = st.number_input("Total Visits", 0, 200, 3)
            time_on_site = st.number_input("Total Time on Website (sec)", 0, 10000, 400)
            page_views   = st.number_input("Page Views Per Visit", 0.0, 50.0, 2.0, 0.5)

        with col2:
            last_activity = st.selectbox("Last Activity", [
                "Email Opened", "Email Bounced", "Email Link Clicked",
                "Email Marked Spam", "Email Received",
                "Page Visited on Website", "Form Submitted on Website",
                "SMS Sent", "Had a Phone Conversation", "Olark Chat Conversation",
                "Resubscribed to emails", "Unreachable", "Unsubscribed",
                "View in browser link Clicked", "Visited Booth in Tradeshow",
                "Converted to Lead", "Approached upfront",
            ])
            country = st.selectbox("Country", [
                "India", "United States", "United Kingdom", "Australia",
                "Canada", "Germany", "Singapore", "Other",
            ])
            specialization = st.selectbox("Specialization", [
                "Select", "Finance Management", "Human Resource Management",
                "Marketing Management", "Operations Management",
                "Business Administration", "IT Projects Management",
                "E-Business", "Healthcare Management", "Hospitality Management",
                "Media and Advertising", "Other",
            ])
            occupation = st.selectbox("Current Occupation", [
                "Unemployed", "Student", "Working Professional",
                "Housewife", "Other",
            ])

        with col3:
            do_not_email  = st.selectbox("Do Not Email", ["No", "Yes"])
            do_not_call   = st.selectbox("Do Not Call",  ["No", "Yes"])
            course_matter = st.selectbox("What matters most?", [
                "Better Career Prospects", "Flexibility & Convenience", "Other",
            ])
            last_notable  = st.selectbox("Last Notable Activity", [
                "Email Opened", "Modified", "Email Bounced", "Email Link Clicked",
                "Email Marked Spam", "Email Received",
                "Page Visited on Website", "Form Submitted on Website",
                "SMS Sent", "Had a Phone Conversation", "Olark Chat Conversation",
                "Resubscribed to emails", "Unreachable", "Unsubscribed",
                "View in browser link Clicked", "Approached upfront",
            ])
            free_copy     = st.selectbox("Free copy - Mastering The Interview", ["No", "Yes"])
            tags          = st.selectbox("Tags", [
                "Ringing", "Will revert after reading the email", "Busy",
                "Interested in other courses", "Interested in Next batch",
                "Shall take in the next coming month", "Still Thinking",
                "In confusion whether part time or DLP", "Interested  in full time MBA",
                "Not doing further education", "Want to take admission but has financial problems",
                "Lost to EINS", "Lost to Others", "Closed by Horizzon",
                "Lateral student", "Graduation in progress",
                "Diploma holder (Not Eligible)", "University not recognized",
                "Recognition issue (DEC approval)", "Already a student",
                "in touch with EINS", "invalid number", "number not provided",
                "wrong number given", "switched off", "opp hangup",
            ])

        submitted = st.form_submit_button("🎯 Score Lead", use_container_width=True)

    if submitted:
        single = pd.DataFrame([{
            "Lead Origin": lead_origin,
            "Lead Source": lead_source,
            "Do Not Email": do_not_email,
            "Do Not Call": do_not_call,
            "TotalVisits": total_visits,
            "Total Time Spent on Website": time_on_site,
            "Page Views Per Visit": page_views,
            "Last Activity": last_activity,
            "Country": country,
            "Specialization": specialization,
            "What is your current occupation": occupation,
            "What matters most to you in choosing a course": course_matter,
            "A free copy of Mastering The Interview": free_copy,
            "Tags": tags,
            "Last Notable Activity": last_notable,
            "Search": "No", "Magazine": "No", "Newspaper Article": "No",
            "X Education Forums": "No", "Newspaper": "No",
            "Digital Advertisement": "No", "Through Recommendations": "No",
            "Asymmetrique Activity Score": 15,
            "Asymmetrique Profile Score": 15,
            "Lead Profile": "Select",
            "City": "Select",
        }])

        result = score_df(single, model, scaler, model_cols).iloc[0]
        prob   = result["Conversion Probability"]
        tier   = apply_thresholds(prob, hot_thresh, warm_thresh)

        cfg = next(v for k, v in TIER_CONFIG.items() if k == tier)
        st.markdown(
            f"""
            <div style="background:{cfg['bg']};border:2px solid {cfg['color']};
                        border-radius:12px;padding:24px;margin-top:16px;text-align:center;">
              <div style="font-size:2.5rem">{tier}</div>
              <div style="font-size:3rem;font-weight:700;color:{cfg['color']}">{prob*100:.1f}%</div>
              <div style="color:#555;font-size:0.9rem">Predicted conversion probability</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Gauge
        fig3, ax3 = plt.subplots(figsize=(6, 1.2))
        ax3.barh(0, 1, color="#eee", height=0.4)
        ax3.barh(0, prob, color=cfg["color"], height=0.4)
        ax3.set_xlim(0, 1)
        ax3.set_yticks([])
        ax3.set_xticks([0, warm_thresh, hot_thresh, 1])
        ax3.set_xticklabels(["0%", f"{warm_thresh*100:.0f}%", f"{hot_thresh*100:.0f}%", "100%"], fontsize=9)
        ax3.spines[["top", "right", "left"]].set_visible(False)
        ax3.axvline(x=warm_thresh, color="#f39c12", linewidth=1.5)
        ax3.axvline(x=hot_thresh,  color="#e74c3c", linewidth=1.5)
        ax3.set_title("Score Position", fontsize=10, pad=4)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()


# ─── TAB 4: Salesforce Integration ───────────────────────────────────────────
with tab4:
    st.header("☁️ Salesforce Integration")
    st.caption("Push scored leads directly to your Salesforce org. The Flow Builder will fire automatically and set the Agent Action field.")

    st.info("""
    **How this works:**
    1. You score a lead in the Single Lead tab (or select from All Predictions)
    2. Click "Push to Salesforce" below
    3. The lead appears in your Salesforce org with Lead_Score__c populated
    4. Your Flow Builder fires automatically and sets Agent_Action__c
    5. Switch to Salesforce to see the result live
    """)

    # Salesforce credentials in sidebar-style expander
    with st.expander("🔐 Salesforce Connection Settings", expanded=True):
        st.info("Using OAuth 2.0 Client Credentials Flow — more secure than username/password.")
        sf_username = st.text_input("Consumer Key (Client ID)", 
                                    value="3MVG9GCMQoQ6rpzSPWOSvWC2aa8LW8_v2n_RQZm0vd8ZMok68NEP9K9Lxfjq7YojqlJ5aSDznl9DT0MjjzUS0")
        sf_password = st.text_input("Consumer Secret", type="password",
                                    help="From your Salesforce Connected App")

    st.divider()
    st.subheader("Push a Single Scored Lead")
    st.caption("Fill in lead details below — scores come from your trained model.")

    with st.form("sf_push_form"):
        sc1, sc2 = st.columns(2)
        with sc1:
            sf_last_name    = st.text_input("Last Name*", value="TestLead")
            sf_company      = st.text_input("Company*", value="Demo Inc")
            sf_lead_source  = st.selectbox("Lead Source", [
                "Google", "Direct Traffic", "Olark Chat", "Reference",
                "Welingak Website", "WeLearn", "Facebook", "Other"
            ])
            sf_country      = st.selectbox("Country", ["India", "United States", "United Kingdom", "Other"])
        with sc2:
            sf_visits       = st.number_input("Total Visits", 0, 200, 3)
            sf_time         = st.number_input("Time on Website (sec)", 0, 10000, 600)
            sf_last_act     = st.selectbox("Last Activity", [
                "Email Opened", "SMS Sent", "Olark Chat Conversation",
                "Email Link Clicked", "Page Visited on Website", "Unreachable"
            ])
            sf_tags         = st.selectbox("Tags", [
                "Will revert after reading the email", "Closed by Horizzon",
                "Interested in Next batch", "Lost to EINS", "Ringing", "Still Thinking"
            ])

        push_btn = st.form_submit_button("🚀 Score & Push to Salesforce", use_container_width=True, type="primary")

    if push_btn:
        if not sf_username or not sf_password:
            st.error("Please fill in your Salesforce credentials above.")
        else:
            try:
                from simple_salesforce import Salesforce as SF, SalesforceAuthenticationFailed

                # Score the lead first
                sf_lead_data = pd.DataFrame([{
                    "Lead Origin": "Landing Page Submission",
                    "Lead Source": sf_lead_source,
                    "Do Not Email": "No", "Do Not Call": "No",
                    "TotalVisits": sf_visits,
                    "Total Time Spent on Website": sf_time,
                    "Page Views Per Visit": 2.0,
                    "Last Activity": sf_last_act,
                    "Country": sf_country,
                    "Specialization": "Finance Management",
                    "What is your current occupation": "Unemployed",
                    "What matters most to you in choosing a course": "Better Career Prospects",
                    "A free copy of Mastering The Interview": "No",
                    "Tags": sf_tags,
                    "Last Notable Activity": "Modified",
                    "Search": "No", "Magazine": "No", "Newspaper Article": "No",
                    "X Education Forums": "No", "Newspaper": "No",
                    "Digital Advertisement": "No", "Through Recommendations": "No",
                    "Asymmetrique Activity Score": 15,
                    "Asymmetrique Profile Score": 15,
                    "Lead Profile": "Select", "City": "Select",
                }])

                result   = score_df(sf_lead_data, model, scaler, model_cols).iloc[0]
                sf_score = float(result["Conversion Probability"])

                # Determine agent action (4-band dissertation thresholds)
                if sf_score > 0.75:
                    agent_act = "Route to Sales Rep"
                elif sf_score >= 0.55:
                    agent_act = "Email Sequence"
                elif sf_score >= 0.45:
                    agent_act = "HITL Escalation"
                else:
                    agent_act = "Nurture + Deflect"

                # Show score before pushing
                st.success(f"**Lead Score: {sf_score*100:.1f}%** → Agent Action: **{agent_act}**")

                # Connect to Salesforce via OAuth Client Credentials
                with st.spinner("Connecting to Salesforce..."):
                    import requests as req
                    token_url = f"https://curious-shark-mgsci4-dev-ed.trailblaze.my.salesforce.com/services/oauth2/token"
                    token_resp = req.post(token_url, data={
                        "grant_type": "client_credentials",
                        "client_id": sf_username,
                        "client_secret": sf_password,
                    })
                    token_data = token_resp.json()
                    if "access_token" not in token_data:
                        raise Exception(f"OAuth failed: {token_data}")
                    from simple_salesforce import Salesforce as SF2
                    conn = SF2(
                        instance_url=token_data["instance_url"],
                        session_id=token_data["access_token"]
                    )

                # Push lead record
                with st.spinner("Pushing lead to Salesforce..."):
                    record = conn.Lead.create({
                        "LastName"          : sf_last_name,
                        "Company"           : sf_company,
                        "LeadSource"        : sf_lead_source,
                        "Country"           : sf_country,
                        "Lead_Score__c"     : round(sf_score, 4),
                        "Agent_Action__c"   : agent_act,
                    })

                if record.get("success"):
                    sf_record_id = record.get("id")
                    st.success(f"""
                    ✅ **Lead successfully pushed to Salesforce!**

                    - **Record ID:** `{sf_record_id}`
                    - **Lead Score:** `{sf_score*100:.1f}%`
                    - **Agent Action:** `{agent_act}`

                    Your Flow Builder has now fired automatically and set the Agent_Action__c field.
                    Switch to your Salesforce org to see the lead record live.
                    """)
                    st.balloons()
                else:
                    st.error(f"Salesforce returned an error: {record}")

            except SalesforceAuthenticationFailed:
                st.error("Authentication failed. Check your username, password, and security token.")
            except ImportError:
                st.error("simple-salesforce not installed. Run: pip3 install simple-salesforce")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    st.divider()
    st.subheader("Push All Hot Leads to Salesforce")
    st.caption(f"This will push all 🔥 Hot leads (score ≥ {hot_thresh}) from your dataset into Salesforce in one batch.")

    n_hot_leads = (scored["Tier"] == "🔥 Hot").sum()
    st.metric("Hot Leads Ready to Push", f"{n_hot_leads:,}")

    if st.button(f"🚀 Push {n_hot_leads:,} Hot Leads to Salesforce", type="primary"):
        if not sf_username or not sf_password or not sf_token:
            st.error("Please fill in your Salesforce credentials in the connection settings above.")
        else:
            try:
                from simple_salesforce import Salesforce as SF, SalesforceAuthenticationFailed

                import requests as req
                token_url = "https://curious-shark-mgsci4-dev-ed.trailblaze.my.salesforce.com/services/oauth2/token"
                token_resp = req.post(token_url, data={
                    "grant_type": "client_credentials",
                    "client_id": sf_username,
                    "client_secret": sf_password,
                })
                token_data = token_resp.json()
                if "access_token" not in token_data:
                    raise Exception(f"OAuth failed: {token_data}")
                from simple_salesforce import Salesforce as SF2
                conn = SF2(
                    instance_url=token_data["instance_url"],
                    session_id=token_data["access_token"]
                )

                hot_leads = scored[scored["Tier"] == "🔥 Hot"].copy()
                success_count = 0
                fail_count    = 0

                progress_bar = st.progress(0)
                status_text  = st.empty()

                for i, (idx, row) in enumerate(hot_leads.iterrows()):
                    try:
                        last_name   = f"Lead_{str(raw_df.loc[idx, 'Prospect ID'])[:8]}" if "Prospect ID" in raw_df.columns else f"Lead_{idx}"
                        company     = f"Org_{str(raw_df.loc[idx, 'Prospect ID'])[:8]}" if "Prospect ID" in raw_df.columns else f"Company_{idx}"
                        lead_source = str(raw_df.loc[idx, "Lead Source"]) if "Lead Source" in raw_df.columns else "Other"
                        country_val = str(raw_df.loc[idx, "Country"])     if "Country"     in raw_df.columns else "India"
                        sf_score    = float(row["Conversion Probability"])
                        agent_act   = "Route to Sales Rep"

                        conn.Lead.create({
                            "LastName"        : last_name,
                            "Company"         : company,
                            "LeadSource"      : lead_source,
                            "Country"         : country_val,
                            "Lead_Score__c"   : round(sf_score, 4),
                            "Agent_Action__c" : agent_act,
                        })
                        success_count += 1
                    except Exception:
                        fail_count += 1

                    progress_bar.progress((i + 1) / len(hot_leads))
                    status_text.text(f"Pushed {i+1} of {len(hot_leads)} leads...")

                status_text.empty()
                st.success(f"✅ Batch complete — {success_count:,} leads pushed to Salesforce successfully. {fail_count} failed.")
                if success_count > 0:
                    st.balloons()

            except Exception as e:
                st.error(f"Connection error: {str(e)}")

    st.divider()
    st.caption(
        "Salesforce integration uses the simple-salesforce Python library. "
        "Lead_Score__c is populated by the trained Logistic Regression model. "
        "Agent_Action__c is set by the dissertation's threshold logic (>0.75 Route to Sales Rep, "
        "0.55-0.75 Email Sequence, 0.45-0.55 HITL Escalation, <0.45 Nurture + Deflect). "
        "Once Lead_Score__c is written, your Salesforce Flow Builder fires automatically — "
        "this demonstrates the complete AI agent to CRM automation pipeline."
    )