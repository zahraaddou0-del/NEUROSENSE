# explore_data.py
import pandas as pd
import os

print("=" * 60)
print("🔍 استكشاف البيانات المحملة")
print("=" * 60)

# تحميل الملف المدمج
df = pd.read_csv('data/all_autism_data_merged.csv')

print(f"\n📊 إحصائيات عامة:")
print(f"   عدد العينات: {len(df):,}")
print(f"   عدد الخصائص: {len(df.columns)}")
print(f"   حجم البيانات: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

print(f"\n📋 أسماء الأعمدة:")
for i, col in enumerate(df.columns, 1):
    print(f"   {i:2}. {col}")

print(f"\n📈 توزيع التشخيص (Class/ASD):")
if 'Class/ASD' in df.columns:
    counts = df['Class/ASD'].value_counts()
    print(f"   0 (لا توحد): {counts.get(0, 0):,} عينة ({counts.get(0, 0)/len(df)*100:.1f}%)")
    print(f"   1 (توحد): {counts.get(1, 0):,} عينة ({counts.get(1, 0)/len(df)*100:.1f}%)")
else:
    print("   ⚠️ عمود 'Class/ASD' غير موجود")

print(f"\n🔢 القيم المفقودة:")
missing = df.isnull().sum()
missing = missing[missing > 0]
if len(missing) > 0:
    for col, count in missing.items():
        print(f"   {col}: {count} قيمة مفقودة")
else:
    print("   ✅ لا توجد قيم مفقودة")

print(f"\n📊 إحصائيات الأسئلة (A1_Score إلى A10_Score):")
for col in ['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
            'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score']:
    if col in df.columns:
        print(f"   {col}: min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.2f}")
