import os
from pymongo import MongoClient
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from dotenv import load_dotenv
import urllib.parse
import pickle

# تحميل المتغيرات من ملف .env
load_dotenv()

def get_mongo_connection():
    """ربط قاعدة البيانات MongoDB"""
    # رابط الاتصال من MongoDB Atlas
    MONGODB_URI = os.getenv("MONGODB_URI")
    if not MONGODB_URI:
        raise ValueError("لم يتم العثور على MONGODB_URI في ملف .env")
    
    client = MongoClient(MONGODB_URI)
    db = client["NEUROSENSE"]  # اسم قاعدة البيانات
    collection = db["assessments"]  # اسم المجموعة
    return client, db, collection

def save_assessment_to_db(data):
    """حفظ بيانات التقييم في قاعدة البيانات"""
    client, db, collection = get_mongo_connection()
    result = collection.insert_one(data)
    client.close()
    return result.inserted_id

def get_model():
    """إنشاء وتدريب نموذج بسيط (سيتم تحسينه لاحقًا)"""
    # هذا نموذج مؤقت للاختبار
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    return model

def predict_asd(features, model):
    """التنبؤ باستخدام النموذج"""
    prediction = model.predict([features])
    probability = model.predict_proba([features])
    return prediction[0], probability[0]