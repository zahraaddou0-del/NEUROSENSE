import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from utils import save_assessment_to_db, get_model, predict_asd

# إعدادات الصفحة
st.set_page_config(
    page_title="NeuroSense - الكشف المبكر للتوحد",
    page_icon="🧠",
    layout="wide"
)

# العنوان الرئيسي
st.title("🧠 NeuroSense")
st.markdown("### الكشف المبكر لاضطراب طيف التوحد باستخدام الذكاء الاصطناعي")

# العمودين: واحد للشرح والثاني للاستبيان
col1, col2 = st.columns([1, 2])

with col1:
    st.info("""
    **ما هو NeuroSense؟**
    
    NeuroSense هو نظام ذكي يساعد في الكشف المبكر لعلامات التوحد 
    لدى الأطفال باستخدام تقنيات الذكاء الاصطناعي.
    
    **كيف يعمل؟**
    1. أجب عن أسئلة الاستبيان
    2. سيقوم النظام بتحليل إجاباتك
    3. ستحصل على تقييم أولي ومؤشر للمخاطر
    """)

with col2:
    st.subheader("📝 استبيان التقييم")
    st.markdown("يرجى الإجابة عن الأسئلة التالية بدقة:")

    # أسئلة Q-Chat-10 (10 أسئلة رئيسية)
    questions = {
        "a1": "هل يحدق الطفل في الأشياء دون هدف واضح؟",
        "a2": "هل يفضل الطفل اللعب بمفرده؟",
        "a3": "هل يتجنب الطفل التواصل البصري؟",
        "a4": "هل يكرر الطفل نفس الحركات بشكل متكرر؟",
        "a5": "هل يتأثر الطفل بشكل غير طبيعي بالأصوات؟",
        "a6": "هل يواجه الطفل صعوبة في فهم مشاعر الآخرين؟",
        "a7": "هل يقوم الطفل بترتيب الأشياء بطريقة معينة ويغضب إذا تغيرت؟",
        "a8": "هل يبدو الطفل غير مهتم بالتفاعل مع الآخرين؟",
        "a9": "هل يتأخر الطفل في تطوير مهارات الكلام؟",
        "a10": "هل يفضل الطفل الروتين الثابت ويصعب عليه التغيير؟"
    }

    responses = {}
    for key, question in questions.items():
        responses[key] = st.radio(
            question, 
            [0, 1], 
            format_func=lambda x: "نعم ✅" if x == 1 else "لا ❌",
            horizontal=True,
            key=key
        )
    
    # معلومات إضافية
    st.subheader("📋 معلومات إضافية")
    
    col_age, col_gender = st.columns(2)
    with col_age:
        age_months = st.number_input("عمر الطفل (بالشهور)", min_value=0, max_value=72, value=24)
    with col_gender:
        gender = st.selectbox("الجنس", ["ذكر", "أنثى"])
        gender_val = 1 if gender == "ذكر" else 0
    
    col_family, col_jaundice = st.columns(2)
    with col_family:
        family_asd = st.selectbox("هل يوجد تاريخ عائلي للتوحد؟", ["لا", "نعم"])
        family_val = 1 if family_asd == "نعم" else 0
    with col_jaundice:
        jaundice = st.selectbox("هل عانى الطفل من اليرقان عند الولادة؟", ["لا", "نعم"])
        jaundice_val = 1 if jaundice == "نعم" else 0
    
    # زر التنبؤ
    if st.button("🔍 تحليل النتائج والتنبؤ", type="primary", use_container_width=True):
        with st.spinner("جاري تحليل البيانات..."):
            # حساب النقاط
            total_score = sum(responses.values())
            
            # تحضير البيانات للحفظ
            assessment_data = {
                "timestamp": datetime.now(),
                "age_months": age_months,
                "gender": gender_val,
                "family_asd": family_val,
                "jaundice": jaundice_val,
                "responses": responses,
                "total_score": total_score,
                "prediction": None
            }
            
            # التنبؤ المبدئي (يعتمد على النقاط)
            if total_score >= 6:
                prediction = 1
                risk_level = "مرتفع"
                message = """
                ⚠️ **نتيجة التقييم: احتمالية مرتفعة**
                
                بناءً على إجاباتك، هناك مؤشرات قد تستدعي استشارة أخصائي.
                هذه النتيجة ليست تشخيصًا نهائيًا، ولكنها دليل للمتابعة.
                
                **نصائح:**
                - استشر أخصائي نمو وسلوك أطفال
                - واصل مراقبة تطور طفلك
                - سجل ملاحظاتك لمشاركتها مع الطبيب
                """
            elif total_score >= 4:
                prediction = 0
                risk_level = "متوسط"
                message = """
                🟡 **نتيجة التقييم: احتمالية متوسطة**
                
                بعض المؤشرات موجودة، لكنها ليست حاسمة.
                
                **نصائح:**
                - تابع تطور طفلك لمدة 3-6 أشهر
                - إذا لاحظت تطورًا في الأعراض، استشر طبيبًا
                """
            else:
                prediction = 0
                risk_level = "منخفض"
                message = """
                🟢 **نتيجة التقييم: احتمالية منخفضة**
                
                المؤشرات التي سجلتها ضمن المعدلات الطبيعية.
                
                **نصائح:**
                - استمر في تحفيز طفلك وتفاعله
                - تابع الفحوصات الدورية للطفل
                """
            
            assessment_data["prediction"] = prediction
            assessment_data["risk_level"] = risk_level
            
            # حفظ في قاعدة البيانات
            try:
                saved_id = save_assessment_to_db(assessment_data)
                st.success(f"✅ تم حفظ التقييم بنجاح! (معرف: {saved_id})")
            except Exception as e:
                st.error(f"حدث خطأ أثناء حفظ البيانات: {str(e)}")
            
            # عرض النتيجة
            st.markdown("---")
            st.subheader("📊 نتيجة التقييم")
            st.markdown(message)
            
            # عرض التفاصيل
            with st.expander("عرض تفاصيل الإجابات"):
                st.write(f"**مجموع النقاط:** {total_score} من 10")
                st.write("**تفاصيل الإجابات:**")
                for key, question in questions.items():
                    status = "✅ نعم" if responses[key] == 1 else "❌ لا"
                    st.write(f"- {question}: {status}") 
