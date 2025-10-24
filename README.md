Project Overview

โปรเจกต์นี้มีจุดประสงค์เพื่อ ทำนายการยกเลิกบริการของลูกค้า (Customer Churn)
โดยใช้ข้อมูลลูกค้าของธนาคาร เช่น อายุ คะแนนเครดิต ยอดคงเหลือ รายได้ สถานะสมาชิก ฯลฯ
จากนั้นนำมาฝึกโมเดล Machine Learning เพื่อระบุว่า

ลูกค้าคนใด “มีแนวโน้มจะยกเลิกบริการ” (Exited = 1)

Dataset Information
Dataset: Customer-Churn-Records.csv
จำนวนแถวข้อมูล: ~10,000 รายการ
จำนวนฟีเจอร์: ~14 คอลัมน์
Target Variable: Exited
0 = ลูกค้ายังอยู่
1 = ลูกค้ายกเลิกบริการ

ตัวอย่างคอลัมน์หลัก:
Feature	Description
CreditScore	คะแนนเครดิตของลูกค้า
Age	อายุ
Balance	ยอดเงินในบัญชี
Geography	ประเทศที่อยู่
Gender	เพศ
NumOfProducts	จำนวนผลิตภัณฑ์ที่ลูกค้าใช้
IsActiveMember	สถานะการเป็นสมาชิกที่ยังใช้งานอยู่
EstimatedSalary	เงินเดือนโดยประมาณ
Exited	ลูกค้ายกเลิกบริการหรือไม่ (Target)

🔍 Workflow Summary
1. Exploratory Data Analysis (EDA)
ตรวจสอบข้อมูลเบื้องต้น (.head(), .info(), .describe())
ตรวจสอบ missing values
วิเคราะห์ฟีเจอร์เชิงหมวดหมู่ เช่น Gender, Geography, IsActiveMember
วาดกราฟการกระจายข้อมูลด้วย Matplotlib และ Seaborn
Countplot สำหรับ Geography / Gender / Exited
Heatmap ของ correlation matrix
Boxplot ระหว่างฟีเจอร์หลักกับการ churn เช่น Age vs Exited

2. Data Cleaning & Preparation
ลบคอลัมน์ที่ไม่จำเป็น: RowNumber, CustomerId, Surname, Complain
ลบแถวที่มี missing values
แปลงข้อมูล categorical เป็น numerical ด้วย One-Hot Encoding
(Geography, Gender, Card Type)
ทำการ scaling ฟีเจอร์เชิงตัวเลขด้วย StandardScaler
เพื่อให้ค่ามีสเกลใกล้เคียงกัน (mean = 0, std = 1)

3. Train-Test Split
แบ่งข้อมูลเป็น 80% สำหรับฝึกโมเดล (Training)
และ 20% สำหรับทดสอบความแม่นยำ (Testing)

5. Model Training (Random Forest Classifier)
ใช้ RandomForestClassifier จาก scikit-learn
พารามิเตอร์ที่ใช้:
n_estimators = 200
max_depth = 9
class_weight = 'balanced'
random_state = 42

จุดเด่นคือ Random Forest สามารถเรียนรู้ pattern ที่ไม่เชิงเส้นได้ดี
และจัดการกับฟีเจอร์หลายตัวพร้อมกันโดยไม่ต้องทำ normalization เพิ่ม

5. Model Evaluation
ใช้ metrics:
Accuracy
Precision, Recall, F1-score
Confusion Matrix
ตัวอย่างผลลัพธ์:
Accuracy: ~0.86
Recall (class 1): ~0.41
Precision (class 1): ~0.76

หมายเหตุ: Recall ของ class 1 อาจต่ำเพราะ dataset ไม่สมดุล
(ลูกค้าที่ไม่ยกเลิกเยอะกว่ามาก)
สามารถแก้ได้โดยใช้เทคนิค SMOTE หรือ oversampling เพิ่มเติม

6. Feature Importance
แสดงผลด้วย bar chart ของ Feature Importance จากโมเดล Random Forest
ฟีเจอร์ที่มีผลต่อการยกเลิกมากที่สุด:
Age
NumOfProducts
IsActiveMember
Balance
Geography_Germany

📈 Key Insights
ลูกค้าที่มี อายุสูงกว่า และ ใช้ผลิตภัณฑ์น้อยกว่า มีแนวโน้มยกเลิกบริการสูง
สมาชิกที่ ไม่ active มัก churn มากกว่า
ประเทศ Germany มี churn rate สูงกว่าประเทศอื่น
การใช้โมเดล Random Forest ช่วยเพิ่มความแม่นยำจาก ~71% (Logistic) → ~86%

🧠 Future Improvements
ใช้เทคนิค SMOTE เพื่อแก้ class imbalance
ปรับ hyperparameter ด้วย GridSearchCV เพื่อ optimize recall
เพิ่มฟีเจอร์ใหม่ เช่น ระยะเวลาการเป็นลูกค้า หรือความถี่ในการใช้บริการ
ทดลองโมเดลอื่น เช่น XGBoost หรือ LightGBM

💻 Requirements
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn

🚀 How to Run
วางไฟล์ Customer-Churn-Records.csv ไว้ใน path ที่กำหนด
รันสคริปต์ใน Python หรือ Jupyter Notebook
python churn_analysis.py

โปรแกรมจะ:
แสดงสถิติข้อมูลและกราฟต่าง ๆ
เทรนโมเดล Random Forest
แสดงผล Confusion Matrix + Classification Report
และแสดงกราฟ Feature Importance
