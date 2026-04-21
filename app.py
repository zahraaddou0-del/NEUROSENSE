cat > app.py << 'EOF'
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from utils import save_assessment_to_db

st.set_page_config(page_title="NeuroSense", page_icon="🧠", layout="wide")

st.title("🧠 NeuroSense - الكشف المبكر للتوحد")

st.info("""
**NeuroSense** هو نظام ذكي يساعد في الكشف المبكر لعلامات التوحد 
عند الأطفال باستخدام تقنيات الذكاء الاصطناعي.
""")

st.subheader("📝 استبيان التقييم")

questions = {
    "a1": "هل يحدق الطفل في الأشياء دون هدف واضح؟",
    "a2": "هل يفضل الطفل اللعب بمفرده؟",
    "a3": "هل يتجنب الطفل التواصل البصري؟",
}

responses = {}
for key, question in questions.items():
    responses[key] = st.radio(question, [0, 1], 
                               format_func=lambda x: "نعم ✅" if x == 1 else "لا ❌",
                               horizontal=True, key=key)

if st.button("🔍 تحليل النتائج", type="primary"):
    total_score = sum(responses.values())
    if total_score >= 2:
        st.warning("⚠️ احتمالية مرتفعة - يُنصح باستشارة أخصائي")
    elif total_score >= 1:
        st.info("🟡 احتمالية متوسطة - تابع تطور الطفل")
    else:
        st.success("🟢 احتمالية منخفضة - الوضع طبيعي")
    
    st.write(f"**مجموع النقاط:** {total_score} من 3")
    
    # حفظ في قاعدة البيانات (سيظهر خطأ إذا لم يكن هناك MongoDB)
    # save_assessment_to_db({"score": total_score, "time": datetime.now()})
EOF
