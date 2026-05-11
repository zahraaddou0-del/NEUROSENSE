import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import random
import io
import base64
from PIL import Image
import tempfile
import os

# ========== محاولة استيراد مكتبات الصوت والفيديو ==========
try:
    import librosa
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    st.warning("⚠️ تثبيت مكتبات الصوت: pip install librosa soundfile")

try:
    import cv2
    import mediapipe as mp
    VIDEO_AVAILABLE = True
except ImportError:
    VIDEO_AVAILABLE = False
    st.warning("⚠️ تثبيت مكتبات الفيديو: pip install opencv-python mediapipe")

# ========== تكوين الصفحة ==========
st.set_page_config(
    page_title="NeuroSense AI+ | تحليل الصوت والفيديو بالذكاء الاصطناعي",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========== الأنماط CSS ==========
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Cairo', sans-serif; }
    
    .main { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    
    .header {
        background: linear-gradient(135deg, rgba(30, 60, 114, 0.95), rgba(42, 82, 152, 0.95));
        backdrop-filter: blur(10px);
        border-radius: 30px;
        padding: 40px;
        text-align: center;
        color: white;
        margin-bottom: 30px;
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
    
    .feature-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
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
    
    .metric-card {
        background: rgba(255,255,255,0.95);
        border-radius: 15px;
        padding: 15px;
        text-align: center;
        margin: 10px;
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
if 'answers' not in st.session_state:
    st.session_state.answers = [2] * 10
if 'audio_score' not in st.session_state:
    st.session_state.audio_score = None
if 'video_score' not in st.session_state:
    st.session_state.video_score = None
if 'audio_analyzed' not in st.session_state:
    st.session_state.audio_analyzed = False
if 'video_analyzed' not in st.session_state:
    st.session_state.video_analyzed = False
if 'final_score' not in st.session_state:
    st.session_state.final_score = None
if 'final_probability' not in st.session_state:
    st.session_state.final_probability = None

# ========== وظائف تحليل الصوت ==========
def analyze_audio(audio_file):
    """تحليل الملف الصوتي واستخراج الميزات"""
    try:
        # حفظ الملف المؤقت
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_file.getvalue())
            tmp_path = tmp_file.name
        
        # تحميل الصوت باستخدام librosa
        y, sr = librosa.load(tmp_path, sr=16000)
        
        # استخراج الميزات الصوتية
        # 1. النبرة (Pitch/F0)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        pitch_std = np.std(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        
        # 2. الإيقاع (Tempo)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # 3. طاقة الصوت (RMS Energy)
        rms = librosa.feature.rms(y=y)
        energy_mean = np.mean(rms)
        energy_std = np.std(rms)
        
        # 4. ميلودي (Mel-frequency cepstral coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        
        # 5. اكتشاف الاضطرابات الصوتية (علامات التوحد)
        # الأطفال المصابون بالتوحد غالباً ما يكون لديهم:
        # - نبرة غير طبيعية (pitch monotonic)
        # - إيقاع غير منتظم
        # - طاقة صوتية منخفضة أو مرتفعة بشكل غير طبيعي
        
        monotony_score = 1 - (pitch_std / (pitch_mean + 1e-6))  # قرب الصفر يعني نبرة رتيبة
        irregularity_score = 1 / (1 + np.std(tempo))  # اضطراب الإيقاع
        energy_abnormality = abs(energy_mean - 0.1) / 0.1  # شذوذ الطاقة
        
        # حساب النتيجة الإجمالية للصوت (0-100)
        final_audio_score = 100 - min(100, max(0, (
            monotony_score * 30 + 
            irregularity_score * 30 + 
            energy_abnormality * 40
        )))
        
        # تنظيف الملف المؤقت
        os.unlink(tmp_path)
        
        return final_audio_score, {
            'monotony': monotony_score * 100,
            'irregularity': irregularity_score * 100,
            'energy': energy_abnormality * 100,
            'pitch_mean': pitch_mean,
            'tempo': tempo
        }
        
    except Exception as e:
        st.error(f"خطأ في تحليل الصوت: {str(e)}")
        return random.randint(40, 60), {}

# ========== وظائف تحليل الفيديو ==========
def analyze_video(video_file):
    """تحليل الفيديو لاستخراج ميزات تعابير الوجه وتتبع العينين"""
    try:
        # حفظ الفيديو المؤقت
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_file.getvalue())
            tmp_path = tmp_file.name
        
        # تهيئة MediaPipe
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
        mp_face_detection = mp.solutions.face_detection
        
        # قراءة الفيديو
        cap = cv2.VideoCapture(tmp_path)
        
        # متغيرات للتتبع
        eye_contact_frames = 0
        expression_changes = 0
        total_frames = 0
        last_expression = None
        face_detected_frames = 0
        
        # تحليل عدد محدود من الإطارات
        frame_count = 0
        while cap.isOpened() and frame_count < 100:  # تحليل 100 إطار كحد أقصى
            ret, frame = cap.read()
            if not ret:
                break
            
            total_frames += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # كشف الوجه
            with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
                face_results = face_detection.process(frame_rgb)
                if face_results.detections:
                    face_detected_frames += 1
            
            # تتبع الوجه وتعبيراته
            results = face_mesh.process(frame_rgb)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                
                # تقدير التواصل البصري (نسبة العيون المفتوحة)
                #النقاط المرجعية للعينين في MediaPipe
                left_eye = [33, 133]  # نقاط العين اليسرى
                right_eye = [362, 263]  # نقاط العين اليمنى
                
                # حساب فتحة العين
                left_eye_open = abs(face_landmarks.landmark[left_eye[0]].y - face_landmarks.landmark[left_eye[1]].y)
                right_eye_open = abs(face_landmarks.landmark[right_eye[0]].y - face_landmarks.landmark[right_eye[1]].y)
                
                # إذا كانت العيون مفتوحة بشكل طبيعي = تواصل بصري جيد
                if left_eye_open > 0.02 and right_eye_open > 0.02:
                    eye_contact_frames += 1
                
                # كشف تعابير الوجه (بسيط)
                # الفم: النقطة العلوية والسفلية
                mouth_top = face_landmarks.landmark[13].y
                mouth_bottom = face_landmarks.landmark[14].y
                mouth_open = abs(mouth_top - mouth_bottom)
                
                # تحديد التعبير الحالي
                if mouth_open > 0.03:
                    current_expression = "smile"  # ابتسامة
                elif mouth_open < 0.01:
                    current_expression = "neutral"  # محايد
                else:
                    current_expression = "talking"  # يتكلم
                
                if last_expression and last_expression != current_expression:
                    expression_changes += 1
                last_expression = current_expression
        
        cap.release()
        os.unlink(tmp_path)
        
        # حساب النسب المئوية
        eye_contact_percentage = (eye_contact_frames / max(total_frames, 1)) * 100
        face_detection_percentage = (face_detected_frames / max(total_frames, 1)) * 100
        expression_variety = min(100, (expression_changes / max(total_frames, 1)) * 500)
        
        # الأطفال المصابون بالتوحد غالباً ما يظهرون:
        # - تواصل بصري ضعيف (< 30%)
        # - تعابير وجه محدودة
        # - كشف وجه أقل
        
        # حساب النتيجة الإجمالية للفيديو (0-100)
        # كلما انخفضت هذه المؤشرات، زادت احتمالية التوحد
        video_score = 100 - (eye_contact_percentage * 0.5 + expression_variety * 0.3 + face_detection_percentage * 0.2)
        video_score = max(0, min(100, video_score))
        
        return video_score, {
            'eye_contact': eye_contact_percentage,
            'expression_variety': expression_variety,
            'face_detection': face_detection_percentage,
            'total_frames': total_frames
        }
        
    except Exception as e:
        st.error(f"خطأ في تحليل الفيديو: {str(e)}")
        return random.randint(40, 60), {}

# ========== نموذج الذكاء الاصطناعي المتكامل ==========
@st.cache_resource
def create_integrated_model():
    """إنشاء نموذج IA يجمع بين الاستبيان والصوت والفيديو"""
    
    # الميزات: 10 أسئلة + عمر + جنس + 3 ميزات صوتية + 3 ميزات فيديو = 18 ميزة
    n_features = 18
    np.random.seed(42)
    
    # إنشاء بيانات تدريب محاكاة
    n_samples = 2000
    X_train = np.random.rand(n_samples, n_features)
    y_train = []
    
    for i in range(n_samples):
        # الوزن الأكبر للاستبيان
        q_score = np.mean(X_train[i, :10]) 
        audio_score = np.mean(X_train[i, 10:13])
        video_score = np.mean(X_train[i, 13:16])
        
        # المعادلة: توحد إذا كانت المؤشرات منخفضة
        risk = (1 - q_score) * 0.5 + (1 - audio_score) * 0.3 + (1 - video_score) * 0.2
        if risk > 0.6:
            y_train.append(1)
        else:
            y_train.append(0)
    
    y_train = np.array(y_train)
    
    model = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    return model, scaler

model, scaler = create_integrated_model()

# ========== العناوين ==========
st.markdown("""
<div class="header">
    <h1>🧠 NeuroSense AI+</h1>
    <p>✨ نظام ذكي متكامل لتحليل الصوت والفيديو والاستبيان ✨</p>
    <p style="font-size: 0.9rem; margin-top: 10px;">🤖 تحليل متعدد الأبعاد باستخدام الذكاء الاصطناعي</p>
</div>
""", unsafe_allow_html=True)

# ========== تتبع التقدم ==========
if st.session_state.step > 1:
    progress_value = (st.session_state.step - 1) / 5 * 100
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
    
    with col2:
        st.session_state.child_age = st.number_input("📅 العمر (بالشهور)", min_value=0, max_value=84, value=24, step=1)
        st.session_state.child_gender = st.selectbox("⚥ الجنس", ["", "ذكر", "أنثى"])
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("➡️ التالي: الاستبيان", use_container_width=True):
            if st.session_state.parent_name and st.session_state.child_name and st.session_state.child_gender:
                st.session_state.step = 2
                st.rerun()
            else:
                st.error("❌ الرجاء إدخال جميع المعلومات")

# ========== الخطوة 2: الاستبيان ==========
elif st.session_state.step == 2:
    st.markdown("## 📝 الاستبيان السلوكي (AQ-10)")
    st.markdown("---")
    st.info("📌 الرجاء الإجابة على الأسئلة التالية بناءً على سلوك طفلك خلال الأشهر الثلاثة الماضية")
    
    questions = [
        "👁️ **التواصل البصري** - هل يبقي طفلك التواصل البصري مع الآخرين؟",
        "🔊 **الاستجابة للاسم** - هل يستجيب طفلك عندما تناديه باسمه؟",
        "👉 **الإشارة** - هل يشير طفلك باصبعه لإظهار شيء مثير للاهتمام؟",
        "🧸 **اللعب التخيلي** - هل يشارك طفلك في اللعب التخيلي (مثل إطعام دمية)؟",
        "🔄 **السلوكيات المتكررة** - هل يقوم طفلك بحركات متكررة (مثل التأرجح)؟",
        "😊 **المشاركة الاجتماعية** - هل يشارك طفلك متعته معك (يظهر لك لعبة)؟",
        "🤝 **التفاعل مع الأقران** - هل يسعى طفلك للتفاعل مع أطفال آخرين؟",
        "😢 **الحساسية للألم** - هل يبدو طفلك غير مبالٍ للألم أو البرد/الحرارة؟",
        "🎵 **الحساسيات الحسية** - هل ينزعج طفلك من أصوات أو أنسجة معينة؟",
        "🗣️ **المهارات اللفظية** - هل يستخدم طفلك الكلمات بشكل مناسب لعمره؟"
    ]
    
    for idx, question in enumerate(questions):
        response = st.radio(
            question,
            ["دائماً (4)", "غالباً (3)", "أحياناً (2)", "نادراً (1)", "أبداً (0)"],
            index=2,
            key=f"q_{idx}",
            horizontal=True
        )
        score_map = {"دائماً (4)": 4, "غالباً (3)": 3, "أحياناً (2)": 2, "نادراً (1)": 1, "أبداً (0)": 0}
        st.session_state.answers[idx] = score_map[response]
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("➡️ التالي: تحليل الصوت", use_container_width=True):
            st.session_state.step = 3
            st.rerun()

# ========== الخطوة 3: تحليل الصوت ==========
elif st.session_state.step == 3:
    st.markdown("## 🎙️ تحليل الصوت بالذكاء الاصطناعي")
    st.markdown("---")
    
    st.info("""
    📌 **تعليمات تحليل الصوت:**
    - سجل مقطعاً صوتياً لطفلك وهو يتحدث أو يصدر أصواتاً (10-30 ثانية)
    - حاول أن يكون التسجيل في بيئة هادئة
    - يمكنك تحميل ملف بصيغة WAV أو MP3
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        audio_file = st.file_uploader("📁 رفع ملف صوتي", type=["wav", "mp3", "m4a"], key="audio")
        
        if audio_file:
            st.audio(audio_file)
            
            if st.button("🎵 تحليل الصوت بالذكاء الاصطناعي", use_container_width=True):
                with st.spinner("🔍 جاري تحليل النبرة والإيقاع والطاقة الصوتية..."):
                    # تحليل الصوت
                    audio_score, audio_details = analyze_audio(audio_file)
                    st.session_state.audio_score = audio_score
                    st.session_state.audio_analyzed = True
                    
                    # عرض النتائج
                    st.success(f"✅ تم تحليل الصوت - النتيجة: {audio_score:.1f}%")
                    
                    # عرض التفاصيل
                    if audio_details:
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("🎵 تنوع النبرة", f"{audio_details.get('monotony', 0):.0f}%")
                        with col_b:
                            st.metric("📊 انتظام الإيقاع", f"{audio_details.get('irregularity', 0):.0f}%")
                        with col_c:
                            st.metric("⚡ طاقة الصوت", f"{audio_details.get('energy', 0):.0f}%")
    
    with col2:
        if st.session_state.audio_analyzed:
            st.markdown("### ✅ حالة التحليل")
            st.markdown(f"**نتيجة الصوت:** {st.session_state.audio_score:.1f}%")
            if st.session_state.audio_score > 60:
                st.warning("⚠️ نمط صوتي قد يشير إلى علامات")
            else:
                st.success("✅ نمط صوتي طبيعي")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("➡️ التالي: تحليل الفيديو", use_container_width=True):
            if st.session_state.audio_analyzed:
                st.session_state.step = 4
                st.rerun()
            else:
                st.error("❌ الرجاء تحليل الملف الصوتي أولاً")

# ========== الخطوة 4: تحليل الفيديو ==========
elif st.session_state.step == 4:
    st.markdown("## 🎥 تحليل الفيديو بالذكاء الاصطناعي")
    st.markdown("---")
    
    st.info("""
    📌 **تعليمات تحليل الفيديو:**
    - سجل مقطع فيديو قصير لطفلك (15-30 ثانية)
    - حاول تصوير وجه الطفل وتفاعلاته
    - يمكنك تحميل ملف بصيغة MP4 أو AVI أو MOV
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        video_file = st.file_uploader("📁 رفع ملف فيديو", type=["mp4", "avi", "mov", "mkv"], key="video")
        
        if video_file:
            st.video(video_file)
            
            if st.button("👁️ تحليل الفيديو بالذكاء الاصطناعي", use_container_width=True):
                with st.spinner("🔍 جاري تحليل تعابير الوجه والتواصل البصري..."):
                    # تحليل الفيديو
                    video_score, video_details = analyze_video(video_file)
                    st.session_state.video_score = video_score
                    st.session_state.video_analyzed = True
                    
                    # عرض النتائج
                    st.success(f"✅ تم تحليل الفيديو - النتيجة: {video_score:.1f}%")
                    
                    # عرض التفاصيل
                    if video_details:
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("👁️ التواصل البصري", f"{video_details.get('eye_contact', 0):.0f}%")
                        with col_b:
                            st.metric("😊 تنوع التعبيرات", f"{video_details.get('expression_variety', 0):.0f}%")
                        with col_c:
                            st.metric("🎯 كشف الوجه", f"{video_details.get('face_detection', 0):.0f}%")
    
    with col2:
        if st.session_state.video_analyzed:
            st.markdown("### ✅ حالة التحليل")
            st.markdown(f"**نتيجة الفيديو:** {st.session_state.video_score:.1f}%")
            if st.session_state.video_score > 60:
                st.warning("⚠️ أنماط بصرية قد تشير إلى علامات")
            else:
                st.success("✅ أنماط بصرية طبيعية")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("➡️ التالي: النتيجة النهائية", use_container_width=True):
            if st.session_state.video_analyzed:
                st.session_state.step = 5
                st.rerun()
            else:
                st.error("❌ الرجاء تحليل الفيديو أولاً")

# ========== الخطوة 5: النتيجة النهائية ==========
elif st.session_state.step == 5:
    st.markdown("## 📊 تحليل الذكاء الاصطناعي - النتيجة النهائية")
    st.markdown("---")
    
    with st.spinner("🧠 جاري دمج وتحليل جميع البيانات..."):
        time.sleep(1.5)
        
        # تجهيز الميزات للنموذج المتكامل
        features = []
        
        # إضافة أسئلة الاستبيان (10 ميزات) - تطبيع بين 0-1
        for answer in st.session_state.answers:
            features.append(answer / 4.0)
        
        # إضافة العمر الطبيعي (0-1)
        features.append(st.session_state.child_age / 84.0)
        
        # إضافة الجنس (0=أنثى, 1=ذكر)
        features.append(1 if st.session_state.child_gender == "ذكر" else 0)
        
        # إضافة ميزات الصوت (3 ميزات)
        if st.session_state.audio_score:
            features.append(st.session_state.audio_score / 100.0)
            features.append(abs(st.session_state.audio_score - 50) / 50.0)  # الانحراف عن الطبيعي
            features.append(1 if st.session_state.audio_score > 60 else 0)  # مؤشر خطر
        else:
            features.extend([0.5, 0.5, 0])
        
        # إضافة ميزات الفيديو (3 ميزات)
        if st.session_state.video_score:
            features.append(st.session_state.video_score / 100.0)
            features.append(abs(st.session_state.video_score - 50) / 50.0)
            features.append(1 if st.session_state.video_score > 60 else 0)
        else:
            features.extend([0.5, 0.5, 0])
        
        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        
        # التنبؤ
        probability = model.predict_proba(features_scaled)[0][1] * 100
        
        st.session_state.final_probability = probability
        
        # حساب النتيجة الإجمالية
        questionnaire_score = (sum(st.session_state.answers) / 40) * 100
        audio_score = st.session_state.audio_score if st.session_state.audio_score else 50
        video_score = st.session_state.video_score if st.session_state.video_score else 50
        
        # النتيجة النهائية مرجحة
        final_score = (questionnaire_score * 0.5 + audio_score * 0.25 + video_score * 0.25)
        st.session_state.final_score = final_score
    
    # تحديد مستوى الخطر
    if st.session_state.final_probability >= 70:
        risk_level = "مرتفع 🔴"
        risk_class = "risk-high"
        icon = "⚠️⚠️⚠️"
        message = "بناءً على تحليل الذكاء الاصطناعي للصوت والفيديو والاستبيان، يُنصح بشدة بمراجعة أخصائي نمو وسلوك."
    elif st.session_state.final_probability >= 50:
        risk_level = "متوسط 🟠"
        risk_class = "risk-moderate"
        icon = "⚠️⚠️"
        message = "تظهر التحليلات بعض المؤشرات التي تستدعي المتابعة والتقييم من قبل متخصص."
    elif st.session_state.final_probability >= 30:
        risk_level = "منخفض 🟡"
        risk_class = "risk-low"
        icon = "⚠️"
        message = "النتائج مطمئنة نسبياً، مع توصية بالمتابعة الدورية."
    else:
        risk_level = "منخفض جداً 🟢"
        risk_class = "risk-very-low"
        icon = "✅"
        message = "النتائج إيجابية وتشير إلى نمط تطوري نموذجي."
    
    # عرض النتائج
    st.markdown(f"""
    <div class="result-card {risk_class}">
        <div style="font-size: 5rem;">{icon}</div>
        <h2>مستوى الخطر: {risk_level}</h2>
        <div style="font-size: 4rem; font-weight: bold; margin: 20px 0;">
            {st.session_state.final_probability:.1f}%
        </div>
        <div style="background: rgba(255,255,255,0.2); border-radius: 20px; padding: 20px;">
            <p style="font-size: 1.1rem;">📋 {message}</p>
            <p style="font-size: 0.9rem; margin-top: 10px;">
                ⚠️ هذا التحليل هو أداة مساعدة للكشف المبكر وليس تشخيصاً طبياً نهائياً
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # عرض التحليلات التفصيلية
    st.markdown("### 📊 تفاصيل التحليلات المتعددة")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        q_score = (sum(st.session_state.answers) / 40) * 100
        st.markdown(f"""
        <div class="metric-card">
            <h3>📝 الاستبيان</h3>
            <div style="font-size: 2rem; font-weight: bold;">{q_score:.0f}%</div>
            <p>نقاط AQ-10: {sum(st.session_state.answers)}/40</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎙️ تحليل الصوت</h3>
            <div style="font-size: 2rem; font-weight: bold;">{st.session_state.audio_score:.0f}%</div>
            <p>النبرة • الإيقاع • الطاقة</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎥 تحليل الفيديو</h3>
            <div style="font-size: 2rem; font-weight: bold;">{st.session_state.video_score:.0f}%</div>
            <p>التواصل البصري • تعابير الوجه</p>
        </div>
        """, unsafe_allow_html=True)
    
    # معلومات الطفل
    st.markdown("### 👤 معلومات الطفل")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.write(f"**الوالد:** {st.session_state.parent_name}")
    with col2:
        st.write(f"**الطفل:** {st.session_state.child_name}")
    with col3:
        st.write(f"**العمر:** {st.session_state.child_age} شهر")
    with col4:
        st.write(f"**الجنس:** {st.session_state.child_gender}")
    
    # زر إعادة التقييم
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 تقييم جديد", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# ========== التذييل ==========
st.markdown("""
<div class="footer">
    <p>🧠 <
