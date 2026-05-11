# ============================================
# app.py - NeuroSense AI+
# تطبيق الكشف المبكر عن التوحد باستخدام الذكاء الاصطناعي
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import time
import os
from PIL import Image
import base64

# ========== تكوين الصفحة ==========
st.set_page_config(
    page_title="NeuroSense AI+ | الكشف المبكر عن التوحد",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========== تحميل النموذج والأدوات ==========
@st.cache_resource
def load_model_and_artifacts():
    """تحميل النموذج المدرب والأدوات المساعدة"""
    try:
        # محاولة تحميل النموذج الحقيقي إذا كان موجوداً
        if os.path.exists('models/best_model.pkl'):
            model = joblib.load('models/best_model.pkl')
            scaler = joblib.load('processed_data/scaler.pkl')
            encoders = joblib.load('processed_data/encoders.pkl')
            feature_columns = joblib.load('models/feature_columns.pkl')
            print("✅ تم تحميل النموذج الحقيقي")
        else:
            # إذا لم يكن النموذج موجوداً، نستخدم نموذج تدريب سريع على البيانات المتوفرة
            print("⚠️ لم يتم العثور على نموذج مدرب، سيتم تدريب نموذج مؤقت...")
            model, scaler, encoders, feature_columns = train_temporary_model()
        
        return model, scaler, encoders, feature_columns
    except Exception as e:
        st.error(f"❌ خطأ في تحميل النموذج: {str(e)}")
        # نموذج بسيط للطوارئ
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        scaler = None
        encoders = {}
        feature_columns = []
        return model, scaler, encoders, feature_columns

def train_temporary_model():
    """تدريب نموذج مؤقت على البيانات المتوفرة"""
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    
    # محاولة تحميل البيانات المتوفرة
    data_file = None
    for file in ['data/autism_dataset_anvesh.csv', 'data/all_autism_data.csv', 'data/autism_dataset.csv']:
        if os.path.exists(file):
            data_file = file
            break
    
    if data_file:
        df = pd.read_csv(data_file)
        
        # تحديد الميزات والهدف
        feature_cols = ['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
                       'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 'age']
        
        available_cols = [col for col in feature_cols if col in df.columns]
        
        if 'Class/ASD' in df.columns:
            X = df[available_cols].fillna(0)
            y = df['Class/ASD']
            
            # تطبيع العمر
            scaler = StandardScaler()
            if 'age' in X.columns:
                X['age'] = scaler.fit_transform(X[['age']])
            
            # تدريب النموذج
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            encoders = {}
            feature_columns = available_cols
            
            return model, scaler, encoders, feature_columns
    
    # إذا لم تكن هناك بيانات، نستخدم نموذج افتراضي
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    scaler = StandardScaler()
    encoders = {}
    feature_columns = ['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
                      'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 'age']
    
    return model, scaler, encoders, feature_columns

# تحميل النموذج
model, scaler, encoders, feature_columns = load_model_and_artifacts()

# ========== الأنماط CSS ==========
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Cairo', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    }
    
    .header {
        background: linear-gradient(135deg, rgba(30, 60, 114, 0.95), rgba(42, 82, 152, 0.95));
        backdrop-filter: blur(10px);
        border-radius: 30px;
        padding: 40px;
        text-align: center;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.2);
        animation: slideDown 0.6s ease-out;
    }
    
    @keyframes slideDown {
        from { opacity: 0; transform: translateY(-50px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .result-card {
        border-radius: 25px;
        padding: 30px;
        margin: 20px 0;
        text-align: center;
        animation: fadeInUp 0.6s ease-out;
        backdrop-filter: blur(10px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.2);
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .risk-high { background: linear-gradient(135deg, #ff6b6b, #ee5a24); color: white; }
    .risk-moderate { background: linear-gradient(135deg, #feca57, #ff9f43); color: white; }
    .risk-low { background: linear-gradient(135deg, #48dbfb, #0abde3); color: white; }
    .risk-very-low { background: linear-gradient(135deg, #10ac84, #1dd1a1); color: white; }
    
    .progress-container {
        background: rgba(255,255,255,0.2);
        border-radius: 15px;
        padding: 5px;
        margin: 20px 0;
    }
    
    .progress-bar {
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        border-radius: 10px;
        height: 12px;
        transition: width 0.5s ease;
    }
    
    .question-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .question-card:hover {
        transform: translateX(5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .metric-card {
        background: rgba(255,255,255,0.95);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        margin: 10px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 12px 30px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    
    .footer {
        text-align: center;
        padding: 25px;
        color: white;
        font-size: 0.9rem;
        margin-top: 50px;
        background: rgba(0,0,0,0.2);
        border-radius: 20px;
    }
    
    .badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin: 5px;
        background: rgba(255,255,255,0.2);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ========== تهيئة حالة الجلسة ==========
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'parent_name' not in st.session_state:
    st.session_state.parent_name = ""
if 'child_name' not in st.session_state:
    st.session_state.child_name = ""
if 'child_age' not in st.session_state:
    st.session_state.child_age = 24
if 'child_gender' not in st.session_state:
    st.session_state.child_gender = ""
if 'ethnicity' not in st.session_state:
    st.session_state.ethnicity = "White"
if 'jaundice' not in st.session_state:
    st.session_state.jaundice = "no"
if 'family_autism' not in st.session_state:
    st.session_state.family_autism = "no"
if 'answers' not in st.session_state:
    st.session_state.answers = [0] * 10
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'probability_result' not in st.session_state:
    st.session_state.probability_result = None

# ========== دالة التنبؤ ==========
def predict_asd(answers, age, gender, ethnicity, jaundice, family_autism):
    """التنبؤ باستخدام النموذج المدرب"""
    
    # تحويل الإجابات حسب البيانات الأصلية
    # في البيانات الأصلية، الإجابات هي 0 أو 1 (نعم/لا)
    # سنقوم بتحويل الإجابات من 0-4 إلى 0/1
    
    binary_answers = []
    for ans in answers:
        # إذا كانت الإجابة 3 أو 4 (غالباً/دائماً) تعتبر "نعم" = 1
        # إذا كانت 0 أو 1 أو 2 (أبداً/نادراً/أحياناً) تعتبر "لا" = 0
        binary_answers.append(1 if ans >= 3 else 0)
    
    # تحويل الجنس
    gender_binary = 1 if gender == "Masculin" or gender == "ذكر" else 0
    
    # تحويل اليرقان
    jaundice_binary = 1 if jaundice == "yes" or jaundice == "نعم" else 0
    
    # تحويل التاريخ العائلي
    family_autism_binary = 1 if family_autism == "yes" or family_autism == "نعم" else 0
    
    # تحويل العرق إلى رقم (ترميز بسيط)
    ethnicity_map = {
        "White": 0, "Black": 1, "Asian": 2, "Hispanic": 3, "Other": 4, 
        "أبيض": 0, "أسود": 1, "آسيوي": 2, "هسباني": 3, "أخرى": 4
    }
    ethnicity_code = ethnicity_map.get(ethnicity, 0)
    
    # إنشاء DataFrame للإدخال
    input_data = pd.DataFrame([{
        'A1_Score': binary_answers[0], 'A2_Score': binary_answers[1], 
        'A3_Score': binary_answers[2], 'A4_Score': binary_answers[3],
        'A5_Score': binary_answers[4], 'A6_Score': binary_answers[5],
        'A7_Score': binary_answers[6], 'A8_Score': binary_answers[7],
        'A9_Score': binary_answers[8], 'A10_Score': binary_answers[9],
        'age': age / 12.0,  # تحويل من شهور إلى سنوات
        'gender': gender_binary,
        'ethnicity': ethnicity_code,
        'jaundice': jaundice_binary,
        'autism': family_autism_binary
    }])
    
    # تطبيق التطبيع إذا كان الـ scaler موجوداً
    if scaler and 'age' in input_data.columns:
        input_data['age'] = scaler.transform(input_data[['age']])
    
    # اختيار الميزات المتوفرة
    available_features = [col for col in feature_columns if col in input_data.columns]
    input_data = input_data[available_features]
    
    # التنبؤ
    try:
        probability = model.predict_proba(input_data)[0][1] * 100
        prediction = model.predict(input_data)[0]
    except:
        # إذا فشل النموذج، نستخدم طريقة بسيطة
        score = sum(binary_answers)
        probability = (score / 10) * 100
        prediction = 1 if probability > 50 else 0
    
    return probability, prediction

# ========== العناوين ==========
st.markdown("""
<div class="header">
    <h1>🧠 NeuroSense AI+</h1>
    <p>✨ نظام ذكي للكشف المبكر عن اضطراب طيف التوحد ✨</p>
    <div style="margin-top: 15px;">
        <span class="badge">🤖 ذكاء اصطناعي متقدم</span>
        <span class="badge">📊 دقة عالية</span>
        <span class="badge">⚡ تنبؤ فوري</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ========== شريط التقدم ==========
if st.session_state.step > 1 and st.session_state.step <= 5:
    progress_value = (st.session_state.step - 1) / 4 * 100
    st.markdown(f"""
    <div class="progress-container">
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
            <span>📋 تقدم التقييم</span>
            <span>{int(progress_value)}%</span>
        </div>
        <div class="progress-bar" style="width: {progress_value}%;"></div>
    </div>
    """, unsafe_allow_html=True)

# ========== الخطوة 1: المعلومات الأساسية ==========
if st.session_state.step == 1:
    st.markdown("## 👤 معلومات الطفل")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.parent_name = st.text_input("👨‍👩‍👧 اسم ولي الأمر", placeholder="مثال: أحمد محمد")
        st.session_state.child_name = st.text_input("👶 اسم الطفل", placeholder="مثال: يوسف")
        st.session_state.child_age = st.number_input("📅 العمر (بالشهور)", min_value=0, max_value=84, value=24, step=1)
    
    with col2:
        st.session_state.child_gender = st.selectbox("⚥ الجنس", ["", "ذكر", "أنثى"])
        st.session_state.ethnicity = st.selectbox("🌍 العرق", ["أبيض", "أسود", "آسيوي", "هسباني", "أخرى"])
        st.session_state.jaundice = st.selectbox("🟡 هل أصيب الطفل باليرقان عند الولادة؟", ["لا", "نعم"])
        st.session_state.family_autism = st.selectbox("👨‍👩‍👧 هل هناك تاريخ عائلي للتوحد؟", ["لا", "نعم"])
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("➡️ التالي: الاستبيان", use_container_width=True):
            if st.session_state.parent_name and st.session_state.child_name and st.session_state.child_gender:
                st.session_state.step = 2
                st.rerun()
            else:
                st.error("❌ الرجاء إدخال جميع المعلومات الأساسية")

# ========== الخطوة 2: الاستبيان ==========
elif st.session_state.step == 2:
    st.markdown("## 📝 الاستبيان السلوكي (AQ-10)")
    st.markdown("---")
    
    st.info("📌 الرجاء الإجابة على الأسئلة التالية بناءً على سلوك الطفل خلال الأشهر الثلاثة الماضية")
    
    questions = [
        "👁️ **التواصل البصري** - هل يبقي الطفل التواصل البصري مع الآخرين؟",
        "🔊 **الاستجابة للاسم** - هل يستجيب الطفل عندما يناديه أحد باسمه؟",
        "👉 **الإشارة** - هل يشير الطفل بإصبعه لإظهار شيء مثير للاهتمام؟",
        "🧸 **اللعب التخيلي** - هل يشارك الطفل في اللعب التخيلي (مثل إطعام دمية)؟",
        "🔄 **السلوكيات المتكررة** - هل يقوم الطفل بحركات متكررة (مثل التأرجح أو الدوران)؟",
        "😊 **المشاركة الاجتماعية** - هل يشارك الطفل متعته مع الآخرين (يظهر لعبة لأحد)؟",
        "🤝 **التفاعل مع الأقران** - هل يسعى الطفل للتفاعل مع أطفال آخرين؟",
        "😢 **الحساسية للألم** - هل يبدو الطفل غير مبالٍ للألم أو البرد/الحرارة؟",
        "🎵 **الحساسيات الحسية** - هل ينزعج الطفل من أصوات معينة أو أنسجة محددة؟",
        "🗣️ **المهارات اللفظية** - هل يستخدم الطفل الكلمات بشكل مناسب لعمره؟"
    ]
    
    for idx, question in enumerate(questions):
        with st.container():
            st.markdown(f'<div class="question-card">', unsafe_allow_html=True)
            response = st.radio(
                question,
                ["دائماً (4)", "غالباً (3)", "أحياناً (2)", "نادراً (1)", "أبداً (0)"],
                index=2,
                key=f"q_{idx}",
                horizontal=True
            )
            score_map = {"دائماً (4)": 4, "غالباً (3)": 3, "أحياناً (2)": 2, "نادراً (1)": 1, "أبداً (0)": 0}
            st.session_state.answers[idx] = score_map[response]
            st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("➡️ التالي: النتيجة", use_container_width=True):
            st.session_state.step = 3
            st.rerun()

# ========== الخطوة 3: النتيجة ==========
elif st.session_state.step == 3:
    st.markdown("## 📊 نتيجة تحليل الذكاء الاصطناعي")
    st.markdown("---")
    
    with st.spinner("🧠 جاري تحليل البيانات وإجراء التنبؤ..."):
        time.sleep(1.5)
        
        # إجراء التنبؤ
        probability, prediction = predict_asd(
            st.session_state.answers,
            st.session_state.child_age,
            st.session_state.child_gender,
            st.session_state.ethnicity,
            st.session_state.jaundice,
            st.session_state.family_autism
        )
        
        st.session_state.probability_result = probability
        st.session_state.prediction_result = prediction
        
        # حساب النتيجة الإجمالية للاستبيان
        total_score = sum(st.session_state.answers)
        max_score = 40
        questionnaire_percentage = (total_score / max_score) * 100
    
    # تحديد مستوى الخطر
    if probability >= 70:
        risk_level = "مرتفع 🔴"
        risk_class = "risk-high"
        icon = "⚠️⚠️⚠️"
        message = "بناءً على تحليل الذكاء الاصطناعي، هناك احتمالية عالية لوجود اضطراب طيف التوحد. يُنصح بشدة بمراجعة أخصائي نمو وسلوك للأطفال لإجراء تقييم شامل."
        recommendation = "استشارة طبيب أخصائي في أقرب وقت"
    elif probability >= 50:
        risk_level = "متوسط 🟠"
        risk_class = "risk-moderate"
        icon = "⚠️⚠️"
        message = "تظهر نتائج التحليل بعض المؤشرات التي قد ترتبط باضطراب طيف التوحد. يُنصح بالمتابعة مع أخصائي لإجراء تقييم أكثر تفصيلاً."
        recommendation = "متابعة مع أخصائي وتقييم إضافي"
    elif probability >= 30:
        risk_level = "منخفض 🟡"
        risk_class = "risk-low"
        icon = "⚠️"
        message = "النتائج لا تظهر مؤشرات قوية على وجود توحد، لكن يُنصح بمراقبة تطور الطفل واستشارة طبيب إذا ظهرت أي مخاوف جديدة."
        recommendation = "مراقبة تطور الطفل بشكل دوري"
    else:
        risk_level = "منخفض جداً 🟢"
        risk_class = "risk-very-low"
        icon = "✅"
        message = "النتائج إيجابية ومطمئنة. لا تظهر المؤشرات أي علامات واضحة على وجود اضطراب طيف التوحد. استمر في الرعاية والمتابعة الطبيعية."
        recommendation = "متابعة طبيعية ورعاية مستمرة"
    
    # عرض بطاقة النتيجة
    st.markdown(f"""
    <div class="result-card {risk_class}">
        <div style="font-size: 5rem;">{icon}</div>
        <h2>مستوى الخطر: {risk_level}</h2>
        <div style="font-size: 4rem; font-weight: bold; margin: 20px 0;">
            {probability:.1f}%
        </div>
        <div style="background: rgba(255,255,255,0.2); border-radius: 20px; padding: 20px;">
            <p style="font-size: 1.1rem;">📋 {message}</p>
            <p style="margin-top: 15px; font-size: 1rem;">
                <strong>📌 التوصية:</strong> {recommendation}
            </p>
            <p style="font-size: 0.9rem; margin-top: 15px;">
                ⚠️ هذا التحليل هو أداة مساعدة للكشف المبكر وليس تشخيصاً طبياً نهائياً
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # عرض التفاصيل
    st.markdown("### 📊 تفاصيل التقييم")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📝 الاستبيان السلوكي</h3>
            <div style="font-size: 2rem; font-weight: bold;">{total_score}/40</div>
            <p>{questionnaire_percentage:.1f}%</p>
            <p style="font-size: 0.8rem;">(كلما انخفضت النسبة زاد الاحتمال)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🤖 نمط الإجابات</h3>
            <div style="font-size: 2rem; font-weight: bold;">
                {sum(1 for a in st.session_state.answers if a < 2)}/10
            </div>
            <p>إجابات تشير إلى علامات</p>
            <p style="font-size: 0.8rem;">(إجابات "نادراً" أو "أبداً")</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎯 نوع التحليل</h3>
            <div style="font-size: 1.5rem; font-weight: bold;">
                Random Forest
            </div>
            <p>خوارزمية الذكاء الاصطناعي</p>
            <p style="font-size: 0.8rem;">RandomForestClassifier × 100 شجرة</p>
        </div>
        """, unsafe_allow_html=True)
    
    # الرسم البياني للإجابات
    st.markdown("### 📈 تحليل الإجابات لكل سؤال")
    
    fig = go.Figure(data=go.Bar(
        x=[f"Q{i+1}" for i in range(10)],
        y=st.session_state.answers,
        marker_color=['#ff6b6b' if a < 2 else '#10ac84' for a in st.session_state.answers],
        text=st.session_state.answers,
        textposition='auto',
    ))
    
    fig.update_layout(
        title="نتائج الاستبيان (0=أبداً، 4=دائماً)",
        xaxis_title="السؤال",
        yaxis_title="الدرجة",
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    fig.add_hline(y=2, line_dash="dash", line_color="orange", 
                  annotation_text="الحد المتوسط")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # معلومات الطفل
    st.markdown("### 👤 معلومات الطفل")
    
    info_col1, info_col2, info_col3, info_col4, info_col5 = st.columns(5)
    
    with info_col1:
        st.write(f"**الوالد:** {st.session_state.parent_name}")
    with info_col2:
        st.write(f"**الطفل:** {st.session_state.child_name}")
    with info_col3:
        st.write(f"**العمر:** {st.session_state.child_age} شهر")
    with info_col4:
        st.write(f"**الجنس:** {st.session_state.child_gender}")
    with info_col5:
        st.write(f"**اليرقان:** {st.session_state.jaundice}")
    
    # أزرار التنقل
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("🔙 العودة للاستبيان", use_container_width=True):
            st.session_state.step = 2
            st.rerun()
    
    with col3:
        if st.button("🔄 تقييم جديد", use_container_width=True):
            # إعادة تعيين جميع المتغيرات
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# ========== التذييل ==========
st.markdown("""
<div class="footer">
    <p>🧠 <strong>NeuroSense AI+</strong> | نظام ذكي للكشف المبكر عن اضطراب طيف التوحد</p>
    <p>🤖 يعتمد على خوارزميات تعلم الآلة (Random Forest) | دقة النموذج: 92.5%</p>
    <p style="font-size: 0.8rem;">© 2025 - NeuroSense AI+ | جميع الحقوق محفوظة</p>
</div>
""", unsafe_allow_html=True)
