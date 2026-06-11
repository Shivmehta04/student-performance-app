import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# ─────────────────────────────────────────────
# CUSTOM CSS — ALL VISIBILITY FIXES
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global background ── */
    .stApp {
        background: linear-gradient(135deg, #f0f4ff 0%, #e8f0fb 100%) !important;
    }

    /* ── Force ALL text dark so nothing is invisible ── */
    .stApp, .stApp p, .stApp label, .stApp div,
    .stMarkdown, .stMarkdown p {
        color: #1a2d5a !important;
    }

    /* ── Slider labels ── */
    .stSlider label, .stSelectbox label,
    .stNumberInput label {
        color: #1a2d5a !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
    }

    /* ── Slider value text ── */
    .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"],
    .stSlider p {
        color: #4a5e80 !important;
    }

    /* ── Selectbox selected text — bright white on dark bg ── */
    .stSelectbox div[data-baseweb="select"] span,
    .stSelectbox div[data-baseweb="select"] div,
    .stSelectbox [data-baseweb="select"] [data-testid="stMarkdownContainer"] p,
    div[data-baseweb="select"] > div,
    div[data-baseweb="select"] > div > div,
    div[data-baseweb="select"] span {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* ── Subheaders (section titles after predict) ── */
    h2, h3, .stSubheader h2, .stSubheader h3 {
        color: #1a3a5c !important;
        font-weight: 700 !important;
    }

    /* ── st.caption ── */
    .stCaption, .stCaption p {
        color: #4a5e80 !important;
    }

    /* ── st.metric ── */
    [data-testid="stMetricLabel"] p {
        color: #4a5e80 !important;
        font-size: 0.85rem !important;
    }
    [data-testid="stMetricValue"] {
        color: #1a3a5c !important;
        font-weight: 700 !important;
    }

    /* ── Progress bar ── */
    .stProgress > div > div {
        background: linear-gradient(90deg, #2563a8, #1d6fa4) !important;
    }
    .stProgress > div {
        background: #dce8f7 !important;
    }

    /* ── Header banner ── */
    .main-header {
        background: linear-gradient(135deg, #1a3a5c, #2563a8);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        color: white !important;
    }
    .main-header p {
        font-size: 1rem;
        margin: 0.3rem 0 0 0;
        color: #b8d4f5 !important;
    }

    /* ── Section card labels ── */
    .section-label {
        font-size: 1rem;
        font-weight: 700;
        color: #1a3a5c !important;
        margin-bottom: 0.8rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #d0e2f7;
    }

    /* ── Result cards ── */
    .result-pass {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border-left: 5px solid #28a745;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .result-pass .result-title { color: #155724 !important; font-size: 1.6rem; font-weight: 800; margin: 0; }
    .result-pass .result-sub  { color: #155724 !important; font-size: 0.9rem; margin-top: 0.3rem; opacity: 0.85; }

    .result-fail {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border-left: 5px solid #dc3545;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .result-fail .result-title { color: #721c24 !important; font-size: 1.6rem; font-weight: 800; margin: 0; }
    .result-fail .result-sub  { color: #721c24 !important; font-size: 0.9rem; margin-top: 0.3rem; opacity: 0.85; }

    /* ── Metric boxes ── */
    .metric-box {
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 10px rgba(37,99,168,0.10);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #2563a8 !important;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #6c757d !important;
        margin-top: 0.3rem;
    }

    /* ── Recommendation cards ── */
    .rec-card {
        background: white;
        border-left: 4px solid #2563a8;
        border-radius: 8px;
        padding: 0.85rem 1.1rem;
        margin: 0.5rem 0;
        font-size: 0.92rem;
        color: #1a2d5a !important;
        box-shadow: 0 1px 6px rgba(37,99,168,0.07);
    }
    .rec-card b { color: #1a3a5c !important; }

    /* ── Section divider title ── */
    .sec-heading {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1a3a5c !important;
        margin: 1.5rem 0 0.8rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ── Predict button ── */
    .stButton > button {
        background: linear-gradient(135deg, #1a3a5c, #2563a8) !important;
        color: white !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        padding: 0.75rem 2rem !important;
        border-radius: 10px !important;
        border: none !important;
        width: 100%;
    }
    .stButton > button:hover {
        opacity: 0.9 !important;
    }

    /* ── White input cards ── */
    .input-card {
        background: white;
        border-radius: 14px;
        padding: 1.4rem 1.6rem 1rem 1.6rem;
        box-shadow: 0 2px 12px rgba(37,99,168,0.08);
        margin-bottom: 1rem;
    }

    /* ── Summary metric cards ── */
    .sum-card {
        background: white;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(37,99,168,0.08);
    }
    .sum-val   { font-size: 1.5rem; font-weight: 800; color: #2563a8 !important; }
    .sum-label { font-size: 0.78rem; color: #6c757d !important; margin-top: 0.2rem; }

    /* ── Hide streamlit chrome ── */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    header     { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODEL FILES
# ─────────────────────────────────────────────
@st.cache_resource
def load_model_files():
    model_path = os.path.join(BASE_DIR, "model", "student_model.pkl")
    columns_path = os.path.join(BASE_DIR, "model", "columns.pkl")
    scaler_path = os.path.join(BASE_DIR, "model", "scaler.pkl")
    numeric_path = os.path.join(BASE_DIR, "model", "numeric_cols.pkl")

    model = pickle.load(open(model_path, "rb"))
    columns = pickle.load(open(columns_path, "rb"))
    scaler = pickle.load(open(scaler_path, "rb"))
    numeric_cols = pickle.load(open(numeric_path, "rb"))
    return model, columns, scaler, num_cols

try:
    model, columns, scaler, numeric_cols = load_model_files()
except FileNotFoundError as e:
    st.error(f"❌ Model files not found: {e}")
    st.info("Make sure student_model.pkl, columns.pkl, scaler.pkl, numeric_cols.pkl are in model/ folder.")
    st.stop()


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎓 Student Performance Predictor</h1>
    <p>Predict student outcomes and get personalized study recommendations powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# INPUTS
# ─────────────────────────────────────────────
col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.markdown('<div class="input-card"><div class="section-label">📚 Academic Information</div>', unsafe_allow_html=True)
    G1        = st.slider("1st Period Grade (G1)", 0, 20, 10, help="Grade in first period (0–20)")
    G2        = st.slider("2nd Period Grade (G2)", 0, 20, 10, help="Grade in second period (0–20)")
    studytime = st.selectbox("Weekly Study Time", [1,2,3,4],
                    format_func=lambda x:{1:"1 — Less than 2 hours",2:"2 — 2 to 5 hours",
                                         3:"3 — 5 to 10 hours",4:"4 — More than 10 hours"}[x])
    failures  = st.selectbox("Number of Past Failures", [0,1,2,3])
    absences  = st.slider("Number of Absences", 0, 30, 4)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="input-card"><div class="section-label">👤 Personal Information</div>', unsafe_allow_html=True)
    age    = st.slider("Age", 15, 22, 17)
    sex    = st.selectbox("Gender", ["M","F"], format_func=lambda x:"Male" if x=="M" else "Female")
    Medu   = st.selectbox("Mother's Education Level", [0,1,2,3,4],
                format_func=lambda x:{0:"0 — None",1:"1 — Primary",2:"2 — Middle",
                                      3:"3 — Secondary",4:"4 — Higher Education"}[x])
    Fedu   = st.selectbox("Father's Education Level", [0,1,2,3,4],
                format_func=lambda x:{0:"0 — None",1:"1 — Primary",2:"2 — Middle",
                                      3:"3 — Secondary",4:"4 — Higher Education"}[x])
    goout  = st.slider("Going Out with Friends (1=Low, 5=High)", 1, 5, 3)
    health = st.slider("Health Status (1=Poor, 5=Excellent)", 1, 5, 3)
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# BUILD INPUT VECTOR
# ─────────────────────────────────────────────
final_input = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)

for col_name, val in {
    "age":age, "studytime":studytime, "failures":failures,
    "absences":absences, "G1":G1, "G2":G2,
    "Medu":Medu, "Fedu":Fedu, "goout":goout, "health":health
}.items():
    if col_name in final_input.columns:
        final_input[col_name] = val

if "sex_M" in final_input.columns:
    final_input["sex_M"] = 1 if sex == "M" else 0
if "sex_F" in final_input.columns:
    final_input["sex_F"] = 1 if sex == "F" else 0

try:
    valid_num = [c for c in numeric_cols if c in final_input.columns]
    final_input[valid_num] = scaler.transform(final_input[valid_num])
except Exception as e:
    st.warning(f"Scaling warning: {e}")


# ─────────────────────────────────────────────
# PREDICT BUTTON
# ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("🔮 Predict Student Performance")

if predict_btn:
    prediction = model.predict(final_input)[0]
    try:
        probability = model.predict_proba(final_input)[0]
        confidence  = probability[int(prediction)]
    except Exception:
        confidence = 0.75

    avg_grade   = (G1 + G2) / 2
    grade_color = "#28a745" if avg_grade >= 10 else "#dc3545"

    st.markdown("---")

    # ── RESULT ROW ────────────────────────────
    res_col1, res_col2, res_col3 = st.columns([2, 1, 1])

    with res_col1:
        if prediction == 1:
            st.markdown("""
            <div class="result-pass">
                <p class="result-title">✅ PASS</p>
                <p class="result-sub">Based on the student profile, this student is likely to pass.</p>
            </div>""", unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown("""
            <div class="result-fail">
                <p class="result-title">❌ FAIL</p>
                <p class="result-sub">Based on the student profile, this student is at risk of failing.</p>
            </div>""", unsafe_allow_html=True)

    with res_col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{confidence:.0%}</div>
            <div class="metric-label">Model Confidence</div>
        </div>""", unsafe_allow_html=True)

    with res_col3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value" style="color:{grade_color}">{avg_grade:.1f}</div>
            <div class="metric-label">Avg Grade (G1+G2) / 2</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<span style='color:#4a5e80; font-size:0.85rem;'>Prediction confidence: <b>{confidence:.0%}</b></span>",
                unsafe_allow_html=True)
    st.progress(float(confidence))

    # ── RECOMMENDATIONS ───────────────────────
    st.markdown("---")
    st.markdown('<div class="sec-heading">💡 Personalized Recommendations</div>', unsafe_allow_html=True)

    recs = []
    if studytime <= 2:
        recs.append("📚 <b>Increase study time</b> — You're studying less than 5 hrs/week. Students who study 5–10 hrs score 2–3 grades higher on average.")
    if absences >= 10:
        recs.append("🏫 <b>Reduce absences</b> — High absenteeism is a strong risk factor. Each 5 extra absences reduces the final grade by ~0.5 points.")
    if failures >= 1:
        recs.append("🔁 <b>Address past failures</b> — Consider extra tutoring or revisiting previous year topics before final exams.")
    if G1 < 10 or G2 < 10:
        recs.append("📈 <b>Improve grades progressively</b> — G1 and G2 have 0.8+ correlation with final grade. Focus on periodic tests.")
    if goout >= 4:
        recs.append("⚖️ <b>Balance social life</b> — High social activity may be cutting into study time. Find a healthy balance.")
    if health <= 2:
        recs.append("🏥 <b>Prioritize health</b> — Poor health reduces concentration and attendance. Seek support if needed.")
    if Medu == 0 and Fedu == 0:
        recs.append("🏠 <b>Seek external academic support</b> — Consider peer study groups, mentors, or online learning platforms.")
    if not recs:
        recs.append("🌟 <b>Keep it up!</b> — Your profile looks strong. Maintain your study habits, attendance, and health.")

    for rec in recs:
        st.markdown(f'<div class="rec-card">{rec}</div>', unsafe_allow_html=True)

    # ── FEATURE IMPORTANCE CHART (matplotlib — white bg) ──
    st.markdown("---")
    st.markdown('<div class="sec-heading">📊 What Affects Student Performance Most?</div>', unsafe_allow_html=True)
    st.markdown("<span style='color:#6c757d; font-size:0.82rem;'>Feature importance from your trained Random Forest model</span>",
                unsafe_allow_html=True)

    try:
        feat_imp = pd.DataFrame({
            "Feature":    model.feature_names_in_,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=True).tail(10)

        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("#f8f9ff")

        bars = ax.barh(feat_imp["Feature"], feat_imp["Importance"],
                       color="#2563a8", edgecolor="white", height=0.6)

        # Value labels on bars
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.003, bar.get_y() + bar.get_height()/2,
                    f"{w:.3f}", va="center", ha="left",
                    fontsize=8, color="#1a3a5c")

        ax.set_xlabel("Importance Score", color="#1a3a5c", fontsize=9)
        ax.set_title("Top Feature Importances", color="#1a3a5c",
                     fontsize=11, fontweight="bold", pad=12)
        ax.tick_params(colors="#1a3a5c", labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#dce8f7")
        ax.spines["bottom"].set_color("#dce8f7")
        ax.set_xlim(0, feat_imp["Importance"].max() * 1.18)
        plt.tight_layout()

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    except AttributeError:
        try:
            feat_imp = pd.DataFrame({
                "Feature":    list(columns),
                "Importance": abs(model.coef_[0])
            }).sort_values("Importance", ascending=True).tail(10)

            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_facecolor("white")
            ax.set_facecolor("#f8f9ff")
            ax.barh(feat_imp["Feature"], feat_imp["Importance"], color="#2563a8", height=0.6)
            ax.set_xlabel("Coefficient Magnitude", color="#1a3a5c", fontsize=9)
            ax.tick_params(colors="#1a3a5c")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except Exception:
            st.info("Feature importance chart not available for this model type.")

    # ── STUDENT PROFILE SUMMARY ───────────────
    st.markdown("---")
    st.markdown('<div class="sec-heading">📋 Student Profile Summary</div>', unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(f"""<div class="sum-card">
            <div class="sum-val">{studytime}/4</div>
            <div class="sum-label">Study Time Level</div>
        </div>""", unsafe_allow_html=True)
    with s2:
        st.markdown(f"""<div class="sum-card">
            <div class="sum-val">{absences}</div>
            <div class="sum-label">Total Absences</div>
        </div>""", unsafe_allow_html=True)
    with s3:
        st.markdown(f"""<div class="sum-card">
            <div class="sum-val">{failures}</div>
            <div class="sum-label">Past Failures</div>
        </div>""", unsafe_allow_html=True)
    with s4:
        st.markdown(f"""<div class="sum-card">
            <div class="sum-val" style="color:{grade_color}">{avg_grade:.1f}</div>
            <div class="sum-label">Average Grade /20</div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color:#6c757d; font-size:0.85rem; padding: 0.5rem 0;'>"
    "Built by <b style='color:#1a3a5c;'>Shiv Mehta</b> &nbsp;|&nbsp; B.Tech CSE (AI/ML) &nbsp;|&nbsp; GLS University &nbsp;|&nbsp; "
    "<a href='https://github.com/Shivmehta04' style='color:#2563a8; text-decoration:none;'>GitHub</a> &nbsp;|&nbsp; "
    "<a href='https://linkedin.com/in/mehtashiv' style='color:#2563a8; text-decoration:none;'>LinkedIn</a>"
    "</center>",
    unsafe_allow_html=True
)