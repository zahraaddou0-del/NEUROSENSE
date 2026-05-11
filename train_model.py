# ============================================
# train_model.py
# تدريب نموذج Random Forest على البيانات المعالجة
# ============================================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

print("=" * 60)
print("🤖 الخطوة 4: تدريب نموذج الذكاء الاصطناعي")
print("=" * 60)

# 1. تحميل البيانات المعالجة
print("\n📂 1. تحميل البيانات المعالجة...")
df = pd.read_csv('processed_data/processed_data.csv')
print(f"   ✅ تم تحميل {len(df)} عينة")

# 2. تحديد الميزات (X) والهدف (y)
print("\n🎯 2. تحديد الميزات والهدف...")

# الميزات التي سنستخدمها للتدريب
feature_columns = [
    'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
    'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
    'age_normalized', 'gender', 'ethnicity', 'jaundice', 'autism'
]

# التأكد من وجود كل الميزات
available_features = [col for col in feature_columns if col in df.columns]
print(f"   ✅ الميزات المستخدمة: {len(available_features)}")
print(f"   📋 {available_features}")

X = df[available_features]

# العمود الهدف
target_column = 'Class/ASD' if 'Class/ASD' in df.columns else df.columns[-1]
y = df[target_column]

print(f"   🎯 الهدف: {target_column}")
print(f"   📊 توزيع الهدف:")
print(y.value_counts())

# 3. تقسيم البيانات إلى تدريب واختبار
print("\n✂️ 3. تقسيم البيانات...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   📚 تدريب: {len(X_train)} عينة")
print(f"   📖 اختبار: {len(X_test)} عينة")

# 4. تدريب نموذج Random Forest
print("\n🌲 4. تدريب نموذج Random Forest...")

model = RandomForestClassifier(
    n_estimators=150,      # 150 شجرة قرار
    max_depth=10,          # عمق كل شجرة
    min_samples_split=5,   # الحد الأدنى لتقسيم العقدة
    min_samples_leaf=2,    # الحد الأدنى للأوراق
    random_state=42,
    n_jobs=-1              # استخدام جميع المعالجات
)

model.fit(X_train, y_train)
print("   ✅ تم تدريب النموذج")

# 5. تقييم النموذج
print("\n📊 5. تقييم النموذج...")

# التنبؤ على بيانات التدريب
y_train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)

# التنبؤ على بيانات الاختبار
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"   🎯 دقة التدريب: {train_accuracy:.2%}")
print(f"   🎯 دقة الاختبار: {test_accuracy:.2%}")

# تقرير مفصل
print("\n   📋 تقرير التصنيف:")
print(classification_report(y_test, y_test_pred, target_names=['لا توحد', 'توحد']))

# 6. التحقق المتقاطع (Cross Validation)
print("\n🔄 6. التحقق المتقاطع (5-fold)...")
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"   📊 متوسط الدقة: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")

# 7. أهمية الميزات
print("\n📈 7. أهمية الميزات:")
feature_importance = pd.DataFrame({
    'feature': available_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for i, row in feature_importance.head(10).iterrows():
    print(f"   {row['feature']}: {row['importance']:.2%}")

# 8. حفظ النموذج
print("\n💾 8. حفظ النموذج...")

os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/best_model.pkl')
print(f"   ✅ تم حفظ النموذج: models/best_model.pkl")

# حفظ قائمة الميزات المستخدمة
joblib.dump(available_features, 'models/feature_columns.pkl')
print(f"   ✅ تم حفظ قائمة الميزات: models/feature_columns.pkl")

print("\n" + "=" * 60)
print(f"🎉 اكتمل التدريب! دقة النموذج: {test_accuracy:.2%}")
print("=" * 60)
