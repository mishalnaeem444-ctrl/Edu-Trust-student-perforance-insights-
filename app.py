import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# =========================================================
# PAGE CONFIGURATION & THEME
# =========================================================
st.set_page_config(
    page_title="EduTrust Research Prototype",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Academic Styling with Strict Traffic-Light Colors
st.markdown("""
<style>
    /* Global Styling */
    .main {
        background-color: #F8FAFC;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Title Header Styling */
    .title-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        padding: 24px 32px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .title-header h1 {
        margin: 0;
        font-size: 28px;
        font-weight: 700;
        color: #FFFFFF !important;
    }
    .title-header p {
        margin: 8px 0 0 0;
        font-size: 16px;
        opacity: 0.95;
        color: #E0E7FF !important;
    }

    /* Card Containers */
    .card {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 24px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        margin-bottom: 20px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .card:hover {
        box-shadow: 0 8px 12px -2px rgba(0, 0, 0, 0.05);
    }
    .card h3 {
        margin-top: 0;
        color: #1E3A8A;
        font-size: 20px;
    }
    .card p, .card li {
        color: #334155;
        font-size: 14.5px;
        line-height: 1.6;
    }

    /* KPI Metric Cards */
    .kpi-card {
        background: #FFFFFF;
        border-radius: 10px;
        padding: 18px 20px;
        border-left: 5px solid #2563EB;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
        margin-bottom: 15px;
    }
    .kpi-title {
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #64748B;
        font-weight: 600;
        margin-bottom: 5px;
    }
    .kpi-value {
        font-size: 26px;
        font-weight: 700;
        color: #1E293B;
    }
    .kpi-sub {
        font-size: 12px;
        color: #64748B;
        margin-top: 4px;
    }

    /* Traffic Light Badges & Status Tags */
    .status-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 13px;
    }
    .badge-ontrack { background-color: #DCFCE7; color: #15803D; border: 1px solid #86EFAC; }
    .badge-improvement { background-color: #FEF3C7; color: #B45309; border: 1px solid #FDE68A; }
    .badge-atrisk { background-color: #FEE2E2; color: #B91C1C; border: 1px solid #FCA5A5; }

    /* Dynamic Recommendation Cards */
    .recommendation-card {
        background-color: #EFF6FF;
        border-left: 4px solid #2563EB;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 10px;
        color: #1E40AF;
        font-weight: 500;
        font-size: 14px;
    }

    /* Key Factor Tags */
    .factor-tag {
        display: block;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .factor-negative { background-color: #FEE2E2; color: #B91C1C; border: 1px solid #FCA5A5; }
    .factor-neutral { background-color: #FEF3C7; color: #B45309; border: 1px solid #FDE68A; }
    .factor-positive { background-color: #DCFCE7; color: #15803D; border: 1px solid #86EFAC; }

    /* Tables */
    .academic-table {
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
        font-size: 14px;
    }
    .academic-table th {
        background-color: #F1F5F9;
        color: #334155;
        text-align: left;
        padding: 12px;
        font-weight: 600;
        border-bottom: 2px solid #CBD5E1;
    }
    .academic-table td {
        padding: 12px;
        border-bottom: 1px solid #E2E8F0;
        color: #1E293B;
    }

    /* Formula Explanation Box */
    .formula-box {
        background-color: #F1F5F9;
        border-left: 4px solid #3B82F6;
        padding: 14px 18px;
        border-radius: 8px;
        font-size: 13px;
        color: #334155;
        margin-top: 10px;
    }

    /* Hide Streamlit Boilerplate */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =========================================================
# DATA ENGINE (500-STUDENT TUNED MODEL DATASET)
# =========================================================
@st.cache_data
def load_and_process_dataset():
    try:
        df = pd.read_csv("student_data.csv")
        if "student_id" not in df.columns:
            df["student_id"] = ["S" + str(1000 + i) for i in range(len(df))]

        req_cols = ["avg_score", "total_weight", "activity_days", "total_clicks"]
        for col in req_cols:
            if col not in df.columns:
                raise KeyError(f"Missing column {col} in CSV")
    except (FileNotFoundError, KeyError):
        np.random.seed(42)
        n = 500
        df = pd.DataFrame({
            "student_id": [f"S{1000 + i}" for i in range(n)],
            "avg_score": np.round(np.random.normal(68, 16, n).clip(20, 98), 1),
            "total_weight": np.round(np.random.normal(72, 18, n).clip(10, 100), 1),
            "activity_days": np.random.normal(82, 28, n).clip(10, 150).astype(int),
            "total_clicks": np.random.normal(680, 290, n).clip(60, 1600).astype(int)
        })

    df = df.head(500).copy()

    # Feature Tiers Evaluation Logic
    def get_score_tier(v):
        return "On Track" if v >= 70.0 else ("Needs Improvement" if v >= 50.0 else "At Risk")

    def get_weight_tier(v):
        return "On Track" if v >= 70.0 else ("Needs Improvement" if v >= 40.0 else "At Risk")

    def get_days_tier(v):
        return "On Track" if v >= 90 else ("Needs Improvement" if v >= 50 else "At Risk")

    def get_clicks_tier(v):
        return "On Track" if v >= 750 else ("Needs Improvement" if v >= 300 else "At Risk")

    df["score_tier"] = df["avg_score"].apply(get_score_tier)
    df["weight_tier"] = df["total_weight"].apply(get_weight_tier)
    df["days_tier"] = df["activity_days"].apply(get_days_tier)
    df["clicks_tier"] = df["total_clicks"].apply(get_clicks_tier)

    # Feature Normalization (Scaled 0 - 100%)
    norm_score = np.minimum(df["avg_score"] / 100.0, 1.0) * 100
    norm_weight = np.minimum(df["total_weight"] / 100.0, 1.0) * 100
    norm_days = np.minimum(df["activity_days"] / 90.0, 1.0) * 100
    norm_clicks = np.minimum(df["total_clicks"] / 750.0, 1.0) * 100

    df["norm_score"] = np.round(norm_score, 1)
    df["norm_weight"] = np.round(norm_weight, 1)
    df["norm_days"] = np.round(norm_days, 1)
    df["norm_clicks"] = np.round(norm_clicks, 1)

    # Overall Performance Score: Direct average of the four normalized feature scores
    df["overall_performance"] = np.round((norm_score + norm_weight + norm_days + norm_clicks) / 4.0, 1)

    # Overall Prediction Status & Risk Level Mapping
    def get_overall_status(p):
        if p >= 70.0:
            return "On Track"
        elif p >= 50.0:
            return "Needs Improvement"
        else:
            return "At Risk"

    def get_risk_level(p):
        if p >= 70.0:
            return "Low Risk"
        elif p >= 50.0:
            return "Moderate Risk"
        else:
            return "High Risk"

    df["prediction_status"] = df["overall_performance"].apply(get_overall_status)
    df["risk_level"] = df["overall_performance"].apply(get_risk_level)

    return df

df = load_and_process_dataset()

# =========================================================
# UNIFIED DASHBOARD HEADER BANNER
# =========================================================
st.title("🎓 EduTrust: Explainable Educational AI")
st.caption("Supporting Teachers with Transparent and Explainable Student Performance Insights.")

st.subheader("📌 1. Study Overview & Participant Briefing")

st.info("""
**Welcome to the Edu-Trust Evaluation**

This interface displays predictions of students’ academic performance to help educators better understand and identify students’ learning progress. It provides insights into expected performance and can support educators in recognizing students who may need additional guidance, support, or intervention to improve their academic outcomes.

**Instructions:** Please review Overall Class Performance and select individual students below to examine how the system explains its predictions. Once complete, scroll to the bottom to submit your feedback.
""")

with st.expander("ℹ️ Ethical & Data Privacy Information"):
    st.write("""
    * All student records displayed in this prototype are processed for research evaluation purposes.
    * This model utilizes an explainable glass-box approach to ensure predictive criteria are clear and auditable.
    * Predictions serve as decision-support insights to complement, rather than replace, educator assessment.
    """)

st.divider()

# =========================================================
# SECTION 2: CLASS ACADEMIC PROGRESS OVERVIEW
# =========================================================
st.header("📊 Class Academic Progress Overview")

avg_class_perf = df["overall_performance"].mean()
st.markdown(f"""
<div style="background-color: #EFF6FF; border-left: 5px solid #2563EB; padding: 12px 20px; border-radius: 8px; margin-bottom: 20px;">
    <h3 style="margin:0; color: #1E3A8A; font-size: 20px;">Overall Class Performance: <span style="color:#2563EB;">{avg_class_perf:.1f}%</span></h3>
    <p style="margin: 4px 0 0 0; color: #475569; font-size: 13px;">Aggregate performance perspective across all 500 enrolled students.</p>
</div>
""", unsafe_allow_html=True)

# 4 Top KPI Cards
k1, k2, k3, k4 = st.columns(4)

total_students = len(df)
on_track = (df["prediction_status"] == "On Track").sum()
needs_imp = (df["prediction_status"] == "Needs Improvement").sum()
at_risk = (df["prediction_status"] == "At Risk").sum()

k1.markdown(f"""
<div class="kpi-card" style="border-left-color: #2563EB;">
    <div class="kpi-title">Total Students</div>
    <div class="kpi-value">{total_students}</div>
    <div class="kpi-sub">Enrolled Class Cohort</div>
</div>
""", unsafe_allow_html=True)

k2.markdown(f"""
<div class="kpi-card" style="border-left-color: #16A34A;">
    <div class="kpi-title">Low Risk (On Track)</div>
    <div class="kpi-value" style="color: #15803D;">{on_track}</div>
    <div class="kpi-sub">{on_track/total_students*100:.1f}% of Class (≥70%)</div>
</div>
""", unsafe_allow_html=True)

k3.markdown(f"""
<div class="kpi-card" style="border-left-color: #D97706;">
    <div class="kpi-title">Moderate Risk (Needs Imp.)</div>
    <div class="kpi-value" style="color: #B45309;">{needs_imp}</div>
    <div class="kpi-sub">{needs_imp/total_students*100:.1f}% of Class (50–69%)</div>
</div>
""", unsafe_allow_html=True)

k4.markdown(f"""
<div class="kpi-card" style="border-left-color: #DC2626;">
    <div class="kpi-title">High Risk (At Risk)</div>
    <div class="kpi-value" style="color: #B91C1C;">{at_risk}</div>
    <div class="kpi-sub">{at_risk/total_students*100:.1f}% of Class (<50%)</div>
</div>
""", unsafe_allow_html=True)

st.divider()

# =========================================================
# SECTION 3: INDIVIDUAL STUDENT ANALYSIS
# =========================================================
st.header("🔍 Individual Student Analysis")

# 1. Student Selection & Risk Level / Status Display
c_select, c_status = st.columns([2, 3])

with c_select:
    selected_id = st.selectbox("Select Student ID from Class List:", df["student_id"].unique())
    student = df[df["student_id"] == selected_id].iloc[0]

with c_status:
    status_str = student["prediction_status"]
    risk_str = student["risk_level"]

    if status_str == "On Track":
        badge_html = '<span class="status-badge badge-ontrack">🟢 On Track</span>'
        risk_color = "#15803D"
    elif status_str == "Needs Improvement":
        badge_html = '<span class="status-badge badge-improvement">🟡 Needs Improvement</span>'
        risk_color = "#B45309"
    else:
        badge_html = '<span class="status-badge badge-atrisk">🔴 At Risk</span>'
        risk_color = "#B91C1C"

    st.markdown(f"""
    <div class="card" style="padding: 16px 20px;">
        <p style="margin:0; font-size: 13px; color: #64748B;">Student ID: <b>{student['student_id']}</b></p>
        <h3 style="margin:4px 0;">Prediction Status: {badge_html}</h3>
        <p style="margin:6px 0 0 0; font-size: 13.5px; color: #334155;">
            Risk Level Indicator: <b style="color: {risk_color}; font-size: 15px;">{risk_str}</b> |
            Overall Performance Score: <b>{student['overall_performance']}%</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

# 2. Overall Performance Calculation Explanation Box
with st.expander("ℹ️ How is the Overall Performance Score Calculated?", expanded=False):
    st.markdown("""
    The **Overall Performance Score** is calculated as the direct average of **four normalized performance features** (each scaled from 0% to 100%):

    1. **Average Assessment Score:** Raw percentage score achieved across assessments.
    2. **Cumulative Assessment Weight:** Submission credit percentage completed.
    3. **Active Learning Days:** Portal attendance normalized against the target benchmark ($Days / 90 \\times 100\\%$, capped at 100%).
    4. **Total VLE Interactions:** Resource clicks normalized against the target benchmark ($Clicks / 750 \\times 100\\%$, capped at 100%).
    """)
    st.markdown(f"""
    <div class="formula-box">
        <b>Student #{student['student_id']} Calculation Breakdown:</b><br>
        • Assessment Score (Normalized): <b>{student['norm_score']:.1f}%</b><br>
        • Assessment Weight (Normalized): <b>{student['norm_weight']:.1f}%</b><br>
        • Active Days ({student['activity_days']}/90 days Normalized): <b>{student['norm_days']:.1f}%</b><br>
        • VLE Interactions ({student['total_clicks']}/750 clicks Normalized): <b>{student['norm_clicks']:.1f}%</b><br>
        • <b>Overall Performance Score = ({student['norm_score']:.1f}% + {student['norm_weight']:.1f}% + {student['norm_days']:.1f}% + {student['norm_clicks']:.1f}%) / 4 = {student['overall_performance']:.1f}%</b>
    </div>
    """, unsafe_allow_html=True)

# 3. Student Performance Indicators Table
st.subheader("📋 Student Performance Indicators")

def format_status_badge(val):
    if val == "On Track":
        return '<span class="status-badge badge-ontrack">On Track</span>'
    elif val == "Needs Improvement":
        return '<span class="status-badge badge-improvement">Needs Improvement</span>'
    else:
        return '<span class="status-badge badge-atrisk">At Risk</span>'

score_interp = "Strong assessment mastery (≥ 70%)" if student['avg_score'] >= 70 else ("Moderate grade average (50-69%)" if student['avg_score'] >= 50 else "Assessment score is below the required threshold (< 50%)")
weight_interp = "Good assessment completion & credit weight" if student['total_weight'] >= 70 else ("Incomplete credit submissions (40-69%)" if student['total_weight'] >= 40 else "Severe missing coursework & low submission weight")
days_interp = "Consistent active daily portal logins" if student['activity_days'] >= 90 else ("Moderate portal login frequency (50-89 days)" if student['activity_days'] >= 50 else "Infrequent LMS logins & low attendance")
clicks_interp = "High online resource & VLE engagement" if student['total_clicks'] >= 750 else ("Moderate VLE interaction clicks (300-749)" if student['total_clicks'] >= 300 else "Low LMS engagement & resource clicks")

table_html = f"""
<table class="academic-table">
    <thead>
        <tr>
            <th>Performance Feature</th>
            <th>Student Actual Value</th>
            <th>Target Benchmark</th>
            <th>Scale / Maximum</th>
            <th>Status</th>
            <th>Interpretation</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><b>Average Assessment Score</b></td>
            <td>{student['avg_score']}%</td>
            <td>≥ 70%</td>
            <td>100%</td>
            <td>{format_status_badge(student['score_tier'])}</td>
            <td>{score_interp}</td>
        </tr>
        <tr>
            <td><b>Cumulative Assessment Weight</b></td>
            <td>{student['total_weight']}%</td>
            <td>≥ 70%</td>
            <td>100%</td>
            <td>{format_status_badge(student['weight_tier'])}</td>
            <td>{weight_interp}</td>
        </tr>
        <tr>
            <td><b>Active Learning Days</b></td>
            <td>{student['activity_days']} Days</td>
            <td>≥ 90 Days</td>
            <td>180 Days</td>
            <td>{format_status_badge(student['days_tier'])}</td>
            <td>{days_interp}</td>
        </tr>
        <tr>
            <td><b>Total VLE Interactions</b></td>
            <td>{student['total_clicks']} Clicks</td>
            <td>≥ 750 Clicks</td>
            <td>1500 Clicks</td>
            <td>{format_status_badge(student['clicks_tier'])}</td>
            <td>{clicks_interp}</td>
        </tr>
    </tbody>
</table>
"""
st.markdown(table_html, unsafe_allow_html=True)

st.divider()

# 4. Feature Metrics & Overall Score Bar Chart & Key Factors Panel
col_chart, col_factors = st.columns([3, 2])

with col_chart:
    st.subheader("📊 Performance Features Breakdown")

    categories = [
        'Assessment Score',
        'Assessment Weight',
        'Active Days',
        'VLE Interactions',
        'Overall Score'
    ]
    values = [
        student["norm_score"],
        student["norm_weight"],
        student["norm_days"],
        student["norm_clicks"],
        student["overall_performance"]
    ]

    def get_bar_color(v):
        return '#16A34A' if v >= 70.0 else ('#D97706' if v >= 50.0 else '#DC2626')

    colors = [get_bar_color(v) for v in values]

    fig_bar = go.Figure(go.Bar(
        x=categories,
        y=values,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition='auto',
        width=0.45
    ))

    fig_bar.add_shape(
        type="line", x0=-0.5, y0=70, x1=4.5, y1=70,
        line=dict(color="#1E293B", width=2, dash="dash")
    )

    fig_bar.update_layout(
        yaxis=dict(range=[0, 105], title="Score / Normalized Benchmark (%)"),
        height=340,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(
        fig_bar,
        use_container_width=True,
        config={'displayModeBar': False}  # Removes the hover/editing toolbar
    )

with col_factors:
    st.subheader("Key Factors Influencing Prediction")
    st.markdown("Key performance feature contributing to the prediction result:")

    factors = []
    if student["avg_score"] < 50.0:
        factors.append('<span class="factor-tag factor-negative">🔻 Low Assessment Score (' + str(student['avg_score']) + '%)</span>')
    elif student["avg_score"] < 70.0:
        factors.append('<span class="factor-tag factor-neutral">⚠️ Moderate Assessment Score (' + str(student['avg_score']) + '%)</span>')
    else:
        factors.append('<span class="factor-tag factor-positive">🟢 Strong Assessment Score (' + str(student['avg_score']) + '%)</span>')

    if student["total_weight"] < 40.0:
        factors.append('<span class="factor-tag factor-negative">🔻 Severe Missing Coursework (' + str(student['total_weight']) + '%)</span>')
    elif student["total_weight"] < 70.0:
        factors.append('<span class="factor-tag factor-neutral">⚠️ Partial Coursework Submitted (' + str(student['total_weight']) + '%)</span>')
    else:
        factors.append('<span class="factor-tag factor-positive">🟢 High Assessment Submission Weight (' + str(student['total_weight']) + '%)</span>')

    if student["activity_days"] < 50:
        factors.append('<span class="factor-tag factor-negative">🔻 Low Active Learning Days (' + str(student['activity_days']) + ' Days)</span>')
    elif student["activity_days"] < 90:
        factors.append('<span class="factor-tag factor-neutral">⚠️ Moderate Portal Attendance (' + str(student['activity_days']) + ' Days)</span>')
    else:
        factors.append('<span class="factor-tag factor-positive">🟢 High Portal Activity (' + str(student['activity_days']) + ' Days)</span>')

    if student["total_clicks"] < 300:
        factors.append('<span class="factor-tag factor-negative">🔻 Low VLE Interaction (' + str(student['total_clicks']) + ' Clicks)</span>')
    elif student["total_clicks"] < 750:
        factors.append('<span class="factor-tag factor-neutral">⚠️ Moderate VLE Interaction (' + str(student['total_clicks']) + ' Clicks)</span>')
    else:
        factors.append('<span class="factor-tag factor-positive">🟢 High VLE Resource Engagement (' + str(student['total_clicks']) + ' Clicks)</span>')

    for factor in factors:
        st.markdown(factor, unsafe_allow_html=True)

st.divider()

# 5. Explainable AI Diagnostic Summary & Recommendations
col_exp_diag, col_recs = st.columns([3, 2])

with col_exp_diag:
    st.subheader("Explainable AI Diagnostic Summary")
    st.markdown("#### Why This Prediction?")

    met_criteria = []
    unmet_criteria = []

    if student["avg_score"] >= 70.0:
        met_criteria.append(f"Assessment Grade (<b>{student['avg_score']}%</b> ≥ 70%)")
    else:
        unmet_criteria.append(f"Assessment Grade (<b>{student['avg_score']}%</b> < 70%)")

    if student["total_weight"] >= 70.0:
        met_criteria.append(f"Submitted Coursework Weight (<b>{student['total_weight']}%</b> ≥ 70%)")
    else:
        unmet_criteria.append(f"Submitted Coursework Weight (<b>{student['total_weight']}%</b> < 70%)")

    if student["activity_days"] >= 90:
        met_criteria.append(f"Active Learning Days (<b>{student['activity_days']} days</b> ≥ 90 days)")
    else:
        unmet_criteria.append(f"Active Learning Days (<b>{student['activity_days']} days</b> < 90 days)")

    if student["total_clicks"] >= 750:
        met_criteria.append(f"VLE Resource Clicks (<b>{student['total_clicks']} clicks</b> ≥ 750 clicks)")
    else:
        unmet_criteria.append(f"VLE Resource Clicks (<b>{student['total_clicks']} clicks</b> < 750 clicks)")

    explanation_html = f"""
    <div class="card" style="padding: 16px;">
        <p style="margin-top:0;">Student <b>#{student['student_id']}</b> is classified as
        <b><span style="color:{risk_color};">{status_str} ({risk_str})</span></b> because
        <b>{len(met_criteria)} out of 4</b> performance criteria were met.</p>
    """

    if met_criteria:
        explanation_html += "<p style='margin:4px 0;'><b>✅ Criteria Met:</b> " + ", ".join(met_criteria) + "</p>"
    if unmet_criteria:
        explanation_html += "<p style='margin:4px 0;'><b>❌ Criteria Not Met:</b> " + ", ".join(unmet_criteria) + "</p>"

    explanation_html += "</div>"
    st.markdown(explanation_html, unsafe_allow_html=True)

with col_recs:
    st.subheader("💡 Actionable Recommendations")
    st.markdown("Recommended Actions Based on Student Performance:")

    dynamic_recs = []
    if student["avg_score"] < 70.0:
        dynamic_recs.append("🎓 **Academic Support:** Schedule a 1-on-1 tutoring session to address low assessment performance.")
    if student["total_weight"] < 70.0:
        dynamic_recs.append("📝 **Submission Warning:** Issue an academic reminder to submit missing/pending weighted assignments.")
    if student["activity_days"] < 90:
        dynamic_recs.append("📅 **Attendance Intervention:** Send an automated portal engagement prompt to increase regular weekly check-ins.")
    if student["total_clicks"] < 750:
        dynamic_recs.append("💻 **VLE Resource Push:** Recommend specific online course modules and study readings to increase LMS interaction.")
    if status_str == "On Track":
        dynamic_recs.append("🌟 **Positive Reinforcement:** Maintain current study routine and encourage advanced peer mentoring roles.")

    for rec in dynamic_recs:
        st.markdown(f'<div class="recommendation-card">{rec}</div>', unsafe_allow_html=True)

st.divider()

# =========================================================
# SECTION 4: TEACHER VALIDATION SURVEY
# =========================================================
st.header("📝 Teacher Validation Evaluation Survey")
st.markdown("""
Thank you for evaluating the **EduTrust Decision Support Portal**. Please click the button below to launch the evaluation survey and share your professional feedback regarding dashboard transparency, model explainability, and diagnostic clarity.
""")

GOOGLE_FORM_URL = "https://docs.google.com/forms/d/e/1FAIpQLSfYQ1z7a9FMoAL_LzVGjq0Gt_ffOr9SWG7MGg4qn6CdaJe1rw/viewform?usp=header"

st.markdown("""
<div class="card" style="text-align: center; padding: 28px 20px 18px 20px; margin-top: 15px;">
    <h3 style="margin-bottom: 8px;">📋 Educator Feedback & Evaluation</h3>
    <p style="margin-bottom: 15px; color: #475569;">
        Click below to open the Teacher Validation Form in a new tab.
    </p>
</div>
""", unsafe_allow_html=True)

c_btn1, c_btn2, c_btn3 = st.columns([1, 2, 1])
with c_btn2:
    st.link_button(
        "📝 Open Teacher Validation Survey Form",
        GOOGLE_FORM_URL,
        type="primary",
        use_container_width=True
    )
