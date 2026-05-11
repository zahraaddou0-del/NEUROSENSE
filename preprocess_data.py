# ============================================
# preprocess_data.py
# معالجة وتنظيف البيانات قبل التدريب
# ============================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

print("=" * 60)
print("🛠️ الخطوة 3: معالجة البيانات (Preprocessing)")
print("=" * 60)

# 1. تحميل البيانات التي قمنا بتحميلها سابقاً
print("\n📂 1. تحميل البيانات...")

# تحميل الملف المدمج (إذا كان موجوداً)
if os.path.exists('data/all_autism_data.csv'):
    df = pd.read_csv('data/all_autism_data.csv')
    print(f"   ✅ تم تحميل الملف المدمج: {len(df)} عينة")
else:
    # أو تحميل ملف فردي
    df = pd.read_csv('data/autism_dataset_anvesh.csv')
    print(f"   ✅ تم تحميل autism_dataset_anvesh.csv: {len(df)} عينة")

print(f"   📊 عدد الأعمدة: {len(df.columns)}")
print(f"   📋 الأعمدة: {df.columns.tolist()}")

# 2. عرض القيم المفقودة
print("\n🔍 2. فحص القيم المفقودة...")
missing = df.isnull().sum()
missing = missing[missing > 0]
if len(missing) > 0:
    print("   ⚠️ توجد قيم مفقودة:")
    for col, count in missing.items():
        print(f"      - {col}: {count} قيمة مفقودة")
    
    # معالجة القيم المفقودة
    print("\n   🛠️ معالجة القيم المفقودة...")
    # للقيم الرقمية: نملأ بالمتوسط
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # للقيم النصية: نملأ بأكثر قيمة متكررة
    text_cols = df.select_dtypes(include=['object']).columns
    for col in text_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else "Unknown")
    
    print("   ✅ تم معالجة القيم المفقودة")
else:
    print("   ✅ لا توجد قيم مفقودة")

# 3. اختيار الميزات المهمة (مثل المشاريع الأصلية)
print("\n🎯 3. اختيار الميزات للتدريب...")

# الأعمدة المهمة للتنبؤ (حسب المشاريع التي أرسلتها)
important_features = [
    'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
    'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
    'age', 'gender', 'ethnicity', 'jaundice', 'autism'
]

# التأكد من وجود كل الأعمدة
available_features = [col for col in important_features if col in df.columns]
missing_features = [col for col in important_features if col not in df.columns]

print(f"   ✅ الميزات المتوفرة: {len(available_features)}/{len(important_features)}")
if missing_features:
    print(f"   ⚠️ الميزات المفقودة: {missing_features}")

# 4. تحويل النصوص إلى أرقام (Label Encoding)
print("\n🔄 4. تحويل البيانات النصية إلى أرقام...")

encoders = {}  # لحفظ أدوات التحويل لاستخدامها لاحقاً

# الأعمدة النصية التي نحتاج لتحويلها
text_columns = ['gender', 'ethnicity', 'jaundice', 'autism']

for col in text_columns:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"   ✅ {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# 5. تطبيع العمر (Normalization)
print("\n📊 5. تطبيع الأرقام...")

if 'age' in df.columns:
    scaler = StandardScaler()
    df['age_normalized'] = scaler.fit_transform(df[['age']])
    print(f"   ✅ age: تم التطبيع (المتوسط=0, الانحراف=1)")
else:
    print("   ⚠️ لا يوجد عمود age")

# 6. التأكد من وجود عمود الهدف (التشخيص)
print("\n🎯 6. التحقق من عمود التشخيص...")

target_column = None
for col in ['Class/ASD', 'ASD', 'target', 'diagnosis']:
    if col in df.columns:
        target_column = col
        break

if target_column:
    print(f"   ✅ عمود الهدف: {target_column}")
    print(f"   📊 توزيع البيانات:")
    print(df[target_column].value_counts())
else:
    print("   ❌ لم يتم العثور على عمود الهدف!")
    print("   📋 الأعمدة المتوفرة:", df.columns.tolist())

# 7. حفظ البيانات المعالجة
print("\n💾 7. حفظ البيانات بعد المعالجة...")

# إنشاء مجلد للمعالجة
os.makedirs('processed_data', exist_ok=True)

# حفظ البيانات بالكامل
df.to_csv('processed_data/processed_data.csv', index=False)
print(f"   ✅ تم حفظ البيانات المعالجة: processed_data/processed_data.csv")

# حفظ الأدوات (encoders, scaler) لاستخدامها في التطبيق
import joblib
joblib.dump(encoders, 'processed_data/encoders.pkl')
joblib.dump(scaler, 'processed_data/scaler.pkl')
print(f"   ✅ تم حفظ أدوات التحويل: encoders.pkl, scaler.pkl")

print("\n" + "=" * 60)
print("✅ اكتملت معالجة البيانات بنجاح!")
print("=" * 60)
