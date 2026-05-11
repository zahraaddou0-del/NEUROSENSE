import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# ========== تكوين صفحة التطبيق ==========
st.set_page_config(
    page_title="NeuroSense AI+ | تشخيص التوحد بالذكاء الاصطناعي",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========== الأنماط (CSS) ==========
# (سأحتفظ بنفس الأنماط الجميلة التي كانت لديك، فهي ممتازة)
st.markdown("""
<style>
    /* ... كل الـ CSS الذي كتبته سابقاً يبقى كما هو ... */
    .main { background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%); }
    .result-card { border-radius: 20px; padding: 25px; margin: 20px 0; text-align: center; animation: fadeIn 0.5s ease-in; }
    .risk-high { background: linear-gradient(135deg, #ff6b6b, #ee5a24); color: white; }
    .risk-moderate { background: linear-gradient(135deg, #feca57, #ff9f43); color: white; }
    .risk-low { background: linear-gradient(135deg, #48dbfb, #0abde3); color: white; }
    .risk-very-low { background: linear-gradient(135deg, #10ac84, #1dd1a1); color: white; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
    .header { background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); border-radius: 20px; padding: 30px; text-align: center; color: white; margin-bottom: 30px; }
    .progress-bar { background: linear-gradient(90deg, #667eea, #764ba2); border-radius: 10px; height: 8px; transition: width 0.3s ease; }
    .footer { text-align: center; padding: 20px; color: #666; font-size: 0.8rem; border-top: 1px solid #ddd; margin-top: 40px; }
</style>
""", unsafe_allow_html=True)

# ========== 1. تحميل النموذج والأدوات (Model & Preprocessors) ==========
# هذه هي أهم خطوة: سنقوم بتحميل نموذج ML تم تدريبه مسبقاً
# يجب أن يكون لديك هذه الملفات في نفس المسار
@st.cache_resource
def load_ml_artifacts():
    """تحميل نموذج ML وأدوات معالجة البيانات"""
    try:
        # سنقوم بتدريب نموذج بسيط في البداية، ولكن يمكنك لاحقاً تحميل model.pkl حقيقي
        # model = joblib.load('best_model.pkl')
        # scaler = joblib.load('scaler.pkl')
        # encoders = joblib.load('encoders.pkl')
        
        # **بدون ملفات حقيقية، سنقوم بإنشاء نموذج وهمي (dummy)**
        # **ولكن في التطبيق الحقيقي، ستستخدم السطور أعلاه**
        
        # إنشاء نموذج افتراضي للتوضيح (يرجى استبداله بنموذجك الحقيقي)
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()
        scaler = StandardScaler()
        le_gender = LabelEncoder()
        le_ethnicity = LabelEncoder()
        le_country = LabelEncoder()
        
        # تدريب الأدوات على بيانات نموذجية (وهذا فقط لجعل الكود يعمل دون أخطاء)
        dummy_data = pd.DataFrame({
            'A1_Score': [0], 'A2_Score': [0], 'A3_Score': [0], 'A4_Score': [0],
            'A5_Score': [0], 'A6_Score': [0], 'A7_Score': [0], 'A8_Score': [0],
            'A9_Score': [0], 'A10_Score': [0], 'age': [5], 'gender': ['m'], 
            'ethnicity': ['?'], 'country': ['US']
        })
        scaler.fit(dummy_data[['age']])
        le_gender.fit(dummy_data['gender'])
        le_ethnicity.fit(dummy_data['ethnicity'])
        le_country.fit(dummy_data['country'])
        model.fit(scaler.transform(dummy_data[['age']]), [0]) # تدريب وهمي
        
        return model, scaler, le_gender, le_ethnicity, le_country
    except Exception as e:
        st.error(f"حدث خطأ في تحميل النموذج: {e}")
        return None, None, None, None, None

model, scaler, le_gender, le_ethnicity, le_country = load_ml_artifacts()

# ========== 2. تهيئة حالة الجلسة (Session State) ==========
# (نفس ما كان لديك مع بعض الإضافات)
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
if 'user_type' not in st.session_state:
    st.session_state.user_type = ""
# ... (كل المتغيرات الأخرى التي كنت تستخدمها) ...
# سأضيف متغيرات لتخزين إجابات الأسئلة الـ 10
for i in range(1, 11):
    if f'q{i}' not in st.session_state:
        st.session_state[f'q{i}'] = 0

# ========== 3. واجهة التطبيق (Application Interface) ==========

# --- العناوين والرأس ---
st.markdown("""
<div class="header">
    <h1 style="font-size: 3rem; margin:0;">🧠 NeuroSense AI+</h1>
    <p style="font-size: 1.2rem; margin-top:10px;">
        نظام ذكي للكشف المبكر عن التوحد باستخدام تقنيات تعلم الآلة
    </p>
</div>
""", unsafe_allow_html=True)

# --- شريط التقدم ---
if st.session_state.current_step > 1 and st.session_state.current_step < 5:
    progress = (st.session_state.current_step - 1) / 4 * 100
    st.markdown(f"""
    <div style="margin-bottom: 20px;">
        <p style="margin-bottom: 5px;">📊 تقدم التقييم</p>
        <div style="background: #e0e0e0; border-radius: 10px;">
            <div class="progress-bar" style="width: {progress}%; border-radius: 10px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- منطق التطبيق الرئيسي (Main App Logic) ---

# **الخطوة 1: اختيار المستخدم**
if st.session_state.current_step == 1:
    st.markdown("## 📋 مرحباً بك! من فضلك اختر نوع المستخدم")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👨‍👩‍👧 أب / أم", use_container_width=True):
            st.session_state.user_type = "parent"
            st.session_state.current_step = 2
            st.rerun()
    with col2:
        if st.button("👨‍⚕️ أخصائي / طبيب", use_container_width=True):
            st.session_state.user_type = "professional"
            st.session_state.current_step = 2
            st.rerun()

# **الخطوة 2: المعلومات الأساسية**
elif st.session_state.current_step == 2:
    st.markdown("## 👤 معلومات أساسية")
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("الاسم الكامل")
        age = st.number_input("العمر (بالشهور للأطفال، أو بالسنوات للبالغين)", min_value=0, max_value=100, step=1)
    with col2:
        gender = st.selectbox("الجنس", ["ذكر", "أنثى"])
        ethnicity = st.selectbox("العرق (اختياري)", ["أبيض", "أسود", "آسيوي", "هسباني/لاتيني", "أخرى", "أفضل عدم الإجابة"])
    
    if st.button("➡️ التالي", use_container_width=True):
        if name and age:
            st.session_state.name = name
            st.session_state.age = age
            st.session_state.gender = 'm' if gender == "ذكر" else 'f'
            st.session_state.ethnicity = ethnicity[0] if ethnicity != "أفضل عدم الإجابة" else '?' # ترميز بسيط
            st.session_state.current_step = 3
            st.rerun()
        else:
            st.error("يرجى إدخال الاسم والعمر")

# **الخطوة 3: الأسئلة السلوكية (AQ-10)**
elif st.session_state.current_step == 3:
    st.markdown("## 📝 التقييم السلوكي")
    st.info("يرجى الإجابة على الأسئلة التالية بناءً على سلوك الطفل أو الشخص خلال الأشهر الثلاثة الماضية.")
    
    # الأسئلة المعتمدة في الأبحاث (AQ-10)
    questions = {
        1: "يلاحظ التفاصيل الصغيرة (أصوات، روائح، ملمس) لا يلاحظها الآخرون؟",
        2: "يجد صعوبة في فهم معنى الإيماءات البسيطة أو تعابير الوجه؟",
        3: "يجد صعوبة في تكوين صداقات؟",
        4: "يأخذ كل شيء بشكل حرفي ولا يفهم النكات أو الاستعارات؟",
        5: "يجد صعوبة في تخيل ما قد يفكر أو يشعر به الآخرون؟",
        6: "يشعر بالانزعاج الشديد عندما تتغير روتيناته اليومية؟",
        7: "يجد صعوبة في معرفة متى يأتي دوره في محادثة؟",
        8: "ينزعج من أصوات معينة لا يبدو أن الآخرين ينزعجون منها؟",
        9: "يجد صعوبة في فهم ما هو مناسب اجتماعياً في المواقف المختلفة؟",
        10: "يفضل القيام بالأنشطة بمفرده بدلاً من القيام بها مع الآخرين؟"
    }
    
    with st.form("questionnaire_form"):
        for i, question in questions.items():
            st.session_state[f'q{i}'] = st.radio(
                f"**السؤال {i}:** {question}",
                options=[("نعم", 1), ("لا", 0), ("أحياناً", 0.5)],
                format_func=lambda x: x[0],
                horizontal=True,
                key=f"radio_{i}"
            )[1]
        
        if st.form_submit_button("📊 تحليل الإجابات"):
            st.session_state.current_step = 4
            st.rerun()

# **الخطوة 4: التنبؤ بنتيجة الذكاء الاصطناعي**
elif st.session_state.current_step == 4:
    st.markdown("## 🤖 تحليل الذكاء الاصطناعي والنتيجة")
    
    with st.spinner("جاري تحليل البيانات وإجراء التنبؤ..."):
        time.sleep(1.5)  # محاكاة لعملية المعالجة
        
        # 4.1. تجهيز البيانات لخوارزمية تعلم الآلة
        # تجميع إجابات الأسئلة العشرة في قائمة Scores
        aq_scores = [st.session_state[f'q{i}'] for i in range(1, 11)]
        total_aq_score = sum(aq_scores)
        
        # إنشاء DataFrame للمدخلات الجديدة
        input_data = pd.DataFrame([{
            'A1_Score': aq_scores[0], 'A2_Score': aq_scores[1], 'A3_Score': aq_scores[2],
            'A4_Score': aq_scores[3], 'A5_Score': aq_scores[4], 'A6_Score': aq_scores[5],
            'A7_Score': aq_scores[6], 'A8_Score': aq_scores[7], 'A9_Score': aq_scores[8],
            'A10_Score': aq_scores[9], 'age': st.session_state.age,
            'gender': st.session_state.gender, 'ethnicity': st.session_state.ethnicity,
            'country': 'US' # قيمة افتراضية
        }])
        
        # 4.2. معالجة البيانات (Preprocessing) مثلما تم معالجتها أثناء التدريب
        # تحويل الجنس (Gender)
        input_data['gender'] = le_gender.transform(input_data['gender'])
        # تحويل العرق (Ethnicity) وتعامل مع القيم الجديدة
        try:
            input_data['ethnicity'] = le_ethnicity.transform(input_data['ethnicity'])
        except ValueError:
            input_data['ethnicity'] = le_ethnicity.transform(['?'])[0] # قيمة افتراضية للأعراق غير المعروفة
        # تحويل الدولة
        try:
            input_data['country'] = le_country.transform(input_data['country'])
        except ValueError:
            input_data['country'] = le_country.transform(['US'])[0]
        
        # تطبيع العمر (Age Scaling)
        input_data['age'] = scaler.transform(input_data[['age']])
        
        # اختيار الميزات النهائية (Feature Selection) - نفس تلك المستخدمة في التدريب
        final_features = ['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 
                          'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 
                          'age', 'gender', 'ethnicity', 'country']
        
        final_input = input_data[final_features]
        
        # 4.3. التنبؤ باستخدام النموذج (Model Prediction)
        prediction_proba = model.predict_proba(final_input)[0]
        asd_probability = prediction_proba[1] * 100  # احتمال وجود توحد
        prediction = model.predict(final_input)[0]
        
        # 4.4. تخزين النتائج في الجلسة
        st.session_state.asd_probability = asd_probability
        st.session_state.prediction = prediction
        
        # 4.5. تحديد مستوى الخطر والرسالة
        if asd_probability >= 70:
            risk_level = "مرتفع 🔴"
            recommendation = "بناءً على التحليل، هناك احتمالية عالية. نوصي بشدة بمراجعة أخصائي نفسي أو طبيب نمو سلوكي لإجراء تقييم شامل."
        elif asd_probability >= 50:
            risk_level = "متوسط 🟠"
            recommendation = "تظهر النتائج بعض المؤشرات. يُنصح بمتابعة دقيقة مع أخصائي ولمزيد من التقييمات السلوكية."
        elif asd_probability >= 30:
            risk_level = "منخفض 🟡"
            recommendation = "الاحتمالية منخفضة. راقب التطور بشكل طبيعي، لكن استشر طبيباً إذا ظهرت أي مخاوف جديدة."
        else:
            risk_level = "منخفض جداً 🟢"
            recommendation = "النتائج مطمئنة وتشير إلى نمط تطوري نموذجي. استمر في الرعاية والمتابعة الطبيعية."
        
        st.session_state.risk_level = risk_level
        st.session_state.recommendation = recommendation
        
        # 4.6. عرض النتائج بطريقة جذابة ومفصلة
        st.markdown("---")
        st.markdown("## 📊 نتيجة التحليل")
        
        # البطاقة الرئيسية
        if asd_probability >= 70:
            risk_class = "risk-high"
        elif asd_probability >= 50:
            risk_class = "risk-moderate"
        elif asd_probability >= 30:
            risk_class = "risk-low"
        else:
            risk_class = "risk-very-low"
        
        st.markdown(f"""
        <div class="result-card {risk_class}">
            <div style="font-size: 4rem;">{'⚠️' if asd_probability >= 50 else '✅'}</div>
            <h2>مستوى الخطر: {risk_level}</h2>
            <div style="font-size: 3rem; font-weight: bold;">{asd_probability:.1f}%</div>
            <p>احتمالية وجود اضطراب طيف التوحد وفقاً لنموذج الذكاء الاصطناعي</p>
            <div style="background: rgba(255,255,255,0.2); border-radius: 15px; padding: 15px; margin-top: 15px;">
                <p><strong>📌 التوصية:</strong> {recommendation}</p>
                <p><strong>🧠 ملاحظة هامة:</strong> هذه النتيجة هي أداة مساعدة للتنبؤ وليست تشخيصاً طبياً نهائياً.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # عرض تفصيلي (Score vs. Probability)
        col1, col2 = st.columns(2)
        with col1:
            fig_score = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=total_aq_score,
                title={'text': "مجموع نقاط الأسئلة (AQ-10)"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={'axis': {'range': [None, 10]}, 'bar': {'color': "#1e3c72"},
                       'steps': [{'range': [0, 3], 'color': "lightgreen"}, {'range': [3, 7], 'color': "orange"}, {'range': [7, 10], 'color': "salmon"}],
                       'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 6}}))
            fig_score.update_layout(height=300)
            st.plotly_chart(fig_score, use_container_width=True)
            
        with col2:
            fig_prob = go.Figure(go.Indicator(
                mode="gauge+number",
                value=asd_probability,
                title={'text': "احتمالية التوحد حسب النموذج"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "#764ba2"},
                       'steps': [{'range': [0, 30], 'color': "lightgreen"}, {'range': [30, 50], 'color': "lightyellow"}, {'range': [50, 100], 'color': "lightcoral"}],
                       'threshold': {'line': {'color': "darkred", 'width': 4}, 'thickness': 0.75, 'value': 70}}))
            fig_prob.update_layout(height=300)
            st.plotly_chart(fig_prob, use_container_width=True)
        
        # إعادة تعيين أو إنهاء
        if st.button("🔄 بدء تقييم جديد"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()

# ========== تذييل الصفحة (Footer) ==========
st.markdown("""
<div class="footer">
    <p>🧠 NeuroSense AI+ | الإصدار 3.0 - نظام تنبؤ بالتوحد قائم على تعلم الآلة</p>
    <p>⚠️ أداة مساعدة للفحص المبكر فقط وليست بديلاً عن التشخيص السريري</p>
</div>
""", unsafe_allow_html=True)
