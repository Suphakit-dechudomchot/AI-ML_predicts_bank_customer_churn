# ------------------------------
# Customer Churn EDA Script
# ------------------------------

import pandas as pd                # ใช้จัดการข้อมูลในรูปแบบ DataFrame
import matplotlib.pyplot as plt     # ใช้สำหรับวาดกราฟทั่วไป
import seaborn as sns               # ใช้สำหรับวาดกราฟสวย ๆ ที่อยู่บน matplotlib

# ------------------------------
# 1. โหลดข้อมูล
# ------------------------------
# อ่านไฟล์ CSV ที่เก็บข้อมูลลูกค้าและสถานะการยกเลิกบริการ (churn)
df = pd.read_csv("C:/Users/UserPRO/Downloads/archive/Customer-Churn-Records.csv")

# แสดง 5 แถวแรกของข้อมูล เพื่อดูโครงสร้างเบื้องต้น
print("=== Head of Data ===")
print(df.head())

# ดูขนาดของข้อมูล (จำนวนแถว, จำนวนคอลัมน์)
print("\n=== Shape ===")
print(df.shape)

# ตรวจสอบชนิดข้อมูล (int, float, object) และจำนวนค่าที่ไม่เป็น null ในแต่ละคอลัมน์
print("\n=== Info ===")
print(df.info())

# ดูค่าทางสถิติพื้นฐานของข้อมูลตัวเลข เช่น mean, std, min, max
print("\n=== Describe ===")
print(df.describe())

# ตรวจสอบว่ามีข้อมูลหาย (missing values) ในแต่ละคอลัมน์หรือไม่
print("\n=== Missing Values ===")
print(df.isnull().sum())

# เช็คค่า complain
print(df['Complain'].value_counts())

# ------------------------------
# 2. วิเคราะห์ Categorical Features
# ------------------------------
# เลือกคอลัมน์ที่เป็นข้อมูลประเภทหมวดหมู่ เช่น เพศ ภูมิภาค สถานะการเป็นสมาชิก ฯลฯ
categorical_cols = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember', 'Exited']

# ลูปผ่านแต่ละคอลัมน์ เพื่อดูการกระจายของค่าหมวดหมู่ (count และร้อยละ)
for col in categorical_cols:
    print(f"\n=== Value Counts for {col} ===")
    print(df[col].value_counts())   # นับจำนวนแต่ละประเภท เช่น Male/Female
    print(f"Percentage:\n{df[col].value_counts(normalize=True)*100}")  # แปลงเป็นเปอร์เซ็นต์

# ------------------------------
# 3. Visualization
# ------------------------------
sns.set(style="whitegrid")  # ตั้งสไตล์กราฟให้ดูสะอาดตา

# --------------------
# Distribution by Geography
# --------------------
plt.figure(figsize=(6,4))
sns.countplot(x='Geography', data=df)
plt.title('Customer Distribution by Geography')  # ชื่อกราฟ
plt.show()

# --------------------
# Distribution by Gender
# --------------------
plt.figure(figsize=(6,4))
sns.countplot(x='Gender', data=df)
plt.title('Customer Distribution by Gender')
plt.show()

# --------------------
# Churn Distribution (Exited = 1 หมายถึง ลูกค้ายกเลิก)
# --------------------
plt.figure(figsize=(6,4))
sns.countplot(x='Exited', data=df)
plt.title('Churn Distribution (Exited Yes/No)')
plt.show()

# --------------------
# Correlation Heatmap (Numerical Features)
# --------------------
# เลือกเฉพาะคอลัมน์ที่เป็นตัวเลข เพื่อนำมาหาความสัมพันธ์ (correlation)
numerical_cols = df.select_dtypes(include=['int64','float64']).columns

# สร้าง heatmap เพื่อดูว่าตัวแปรไหนสัมพันธ์กัน เช่น Age, Balance, CreditScore
plt.figure(figsize=(12,10))
sns.heatmap(df[numerical_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# --------------------
# Boxplot: CreditScore vs Exited
# --------------------
# ใช้ดูการกระจายของ CreditScore ในแต่ละกลุ่ม Exited (0 = อยู่ต่อ, 1 = ยกเลิก)
plt.figure(figsize=(6,4))
sns.boxplot(x='Exited', y='CreditScore', data=df)
plt.title('Credit Score vs Churn')
plt.show()

# --------------------
# Boxplot: Age vs Exited
# --------------------
# ใช้ดูว่าช่วงอายุไหนมีแนวโน้มยกเลิกบริการมากกว่า
plt.figure(figsize=(6,4))
sns.boxplot(x='Exited', y='Age', data=df)
plt.title('Age vs Churn')
plt.show()


# Data Preparation ------------------------------------------------------------------

# ตรวจสอบ missing values อีกครั้ง
df.isnull().sum()

df = df.dropna()  # ลบแถวที่มี missing

# ลบคอลัมน์ที่ไม่จำเป็นสำหรับการเทรนโมเดล
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# ลบ Complain (ทดสอบว่าโมเดลยังดีไหมถ้าไม่มีมัน)
df = df.drop(columns=['Complain'])()

# One-hot encoding (แปลงข้อมูลหมวดหมู่เป็นตัวเลข) ----------------------------------------

# โมเดล ML อ่าน string ไม่ออก เช่น “France”, “Germany” → ต้องแปลงเป็น column ตัวเลขแทน
df = pd.get_dummies(df, columns=['Geography', 'Gender', 'Card Type'], drop_first=True)


# ตรวจสอบข้อมูลประเภท object
print("Non-numeric columns:", df.select_dtypes(include=['object']).columns)

# Feature Scaling (ปรับขนาดข้อมูล) ------------------------------------------------------

# โมเดลอย่าง Logistic Regression และ SVM ต้องการให้ค่าของ features อยู่ในช่วงใกล้ ๆ กัน
# เช่น Balance อาจมีค่าหลักแสน แต่ Age แค่หลักสิบ → ถ้าไม่ normalize โมเดลจะ bias กับตัวเลขใหญ่
# StandardScaler() จะปรับให้แต่ละคอลัมน์มี mean = 0 และ std = 1
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
numerical_features = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Train-Test Split (แยกข้อมูลฝึกและทดสอบ) -----------------------------------------------

from sklearn.model_selection import train_test_split

X = df.drop('Exited', axis=1)   # features
y = df['Exited']                # target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# X = ข้อมูลลูกค้า (features)
# y = label (1 = ลูกค้ายกเลิก, 0 = อยู่ต่อ)
# แบ่งข้อมูลเป็น 80% train / 20% test


# ------------------------------
# 5. Model Training and Evaluation (Optimized Random Forest)
# ------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 5.1 สร้างและฝึกโมเดล Random Forest ที่ปรับปรุงแล้ว
# ใช้ max_depth=10 ตามผลลัพธ์ที่ดีที่สุดจาก GridSearchCV
rf_optimized = RandomForestClassifier(
    n_estimators=200, 
    max_depth=9, 
    class_weight='balanced',  # ใช้การถ่วงน้ำหนักเพื่อเพิ่ม Recall ของ Class 1 (Churn)
    random_state=42
)
print("\n--- Training Optimized Random Forest Model ---")
rf_optimized.fit(X_train, y_train)

# 5.2 ทำนายผล
# ทำนายผลบน Training Set (ข้อมูลที่ใช้ฝึก)
y_train_pred = rf_optimized.predict(X_train)
# ทำนายผลบน Test Set (ข้อมูลที่ไม่เคยเห็น)
y_test_pred = rf_optimized.predict(X_test)

# 5.3 แสดงผลลัพธ์
print("\n==============================================")
print("=== Performance on TRAINING Data (X_train) ===")
print("==============================================")
print("Confusion Matrix (Training):\n", confusion_matrix(y_train, y_train_pred))
print("\nClassification Report (Training):\n", classification_report(y_train, y_train_pred))

print("\n==============================================")
print("=== Performance on TEST Data (X_test) ===")
print("==============================================")
print("Confusion Matrix (Testing):\n", confusion_matrix(y_test, y_test_pred))
print("\nClassification Report (Testing):\n", classification_report(y_test, y_test_pred))

print("\n--- Class Distribution (Target 'Exited') ---")
print(y.value_counts(normalize=True))


# ------------------------------
# Feature Importance Visualization
# ------------------------------
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ดึงค่าความสำคัญของแต่ละฟีเจอร์ออกมา
importances = rf_optimized.feature_importances_
features = X.columns

# เรียงค่าจากมากไปน้อยเพื่อให้กราฟอ่านง่าย
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,6))
sns.barplot(
    x=importances[indices],
    y=np.array(features)[indices],
    palette='viridis'
)
plt.title("Feature Importance (Optimized Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()