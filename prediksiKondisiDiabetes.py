#!/usr/bin/env python
# coding: utf-8

# ## 1. Import Library

# In[20]:


# === Library Dasar Python ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pickle
import os

# === Library untuk Pemodelan & Machine Learning ===
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier

# === Library untuk Evaluasi Model ===
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix

# === Library untuk Hyperparameter Tuning ===
import optuna

# === Library untuk Penanganan Data Imbalanced ===
from imblearn.over_sampling import SMOTE

# === Library untuk Dataset Eksternal ===
from datasets import load_dataset


# ## 2. Data Understanding

# In[7]:


ds = load_dataset("m2s6a8/diabetes_prediction_dataset")
data = pd.DataFrame(ds['train'])


# ### 2.1 Mengetahui Ringkasan Statistik dari Data

# In[8]:


data.describe(include='all')


# **Penjelasan :**
# 
# Berdasarkan deksripsi statistik dari data yang diperoleh, terlihat bahwa dataset berjumlah 100.000 observasi. Terdapat 9 variabel pada dataset yang digunakan. Variabel kategorik yang digunakan adalah gender (Jenis Kelamin) dan smoking_history (Riwayat Merokok). Berdasarkan data yang diberikan terdapat 3 nilai pada variabel gender, yaitu Male, Female, dan Other. Pada variabel smoking_history terdaapt 6 nilai bertipe nominal. 
# 
# Pada variabel numerik terdapat 7 variabel seperti age, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level, dan diabetes. Terlihat berdasarkan keseluruhan data, responden berada pada usia 41 - 42 tahun. Selain itu terlihat standar deviasi usia berada di nilai 22 - 23 tahun, hal tersebut menunjukkan kondisi dimana terdapat keberagaman usia responden. Selain itu, pada variabel hypertension terlihat rata-rata nilai berada di angka 0.07 dan memiliki standar deviasi cukup besar yaitu 0.26, di mana nilainya lebih besar dibanding rataannya. Hal ini juga ditunjukkan pada variabel heart_disease, di mana nilainya juga lebih besar dibanding rataannya.
# 
# Pada variabel bmi yang menyatakan Body Mass Index terlihat bahwa rataan responden berada pada nilai 27 - 28. Pada variabel HbA1c_level terlihat nilai minimum 3.5 dan max 9. Nilai rataan menunjukkan nilai 5.52 karena mendekati nilai minimum. Pada variabel blood_glucose_level terlihat bahwa responden berada ada gula rata-rata 138. Terakhir, pada variabel target yaitu status diabetes terlihat bahwa responden cenderung berada pada kondisi tidak terkena diabetes.

# ### 2.2 Visualisasi Sebaran Data pada Setiap Variabel

# In[9]:


def visualisasiData(df, label_column):
    # Setup tampilan
    sns.set(style="whitegrid")
    n_cols = 2
    cols = df.columns.drop(label_column)
    n_rows = (len(cols) + 1) // n_cols

    plt.figure(figsize=(20, n_rows * 4))

    for idx, col in enumerate(cols):
        plt.subplot(n_rows, n_cols, idx + 1)
        if pd.api.types.is_numeric_dtype(df[col]):
            sns.histplot(df[col], kde=True, color="skyblue")
            plt.title(f'Histogram of {col}')
        else:
            df[col].value_counts().plot(kind='bar', color='coral')
            plt.title(f'Bar Chart of {col}')
            plt.xlabel(col)
            plt.ylabel("Count")

    # Visualisasi pie chart untuk label
    label_counts = df[label_column].value_counts()
    labels = label_counts.index.tolist()
    plt.figure(figsize=(4, 4))
    plt.pie(label_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#ff9999'])
    plt.title('Distribution of Diabetes Label')
    plt.legend(title=label_column)
    plt.axis('equal')  # Circle pie chart

    plt.tight_layout()
    plt.show()

# Panggil fungsi
visualisasiData(data, 'diabetes')


# **Penjelasan :**
# 
# Pada visualisasi yang diperoleh terlihat pada distribusi label didominasi dengan kondisi tidak terkena diabetes. Hal ini umum terjadi pada data medis, di mana kondisi tidak terkena penyakit lebih sering dijumpai dibanding dengan kondisi terkena penyakit. Sebaran nilai age itu terlihat bahwa usia didominasi pada responden lansia, dimana frekuensi tertinggi pada usia 80 tahun. Selain itu, responden didominasi pada jenis kelamin laki-laki.

# ## 3. Data Preparation

# ### 3.1 Encoding Variabel Kategorik

# In[10]:


def preprocessingData(df, kolom, jenis=1, save_dir="Encoder Tersimpan"):
    os.makedirs(save_dir, exist_ok=True)

    if jenis == 1:
        le = LabelEncoder()
        df[kolom] = le.fit_transform(df[kolom])

        # Simpan encoder
        encoder_path = os.path.join(save_dir, f"{kolom}_label_encoder.pkl")
        with open(encoder_path, 'wb') as file:
            pickle.dump(le, file)

        print(f"LabelEncoder untuk '{kolom}' disimpan di: {encoder_path}")

        return df, le
    
    elif jenis == 2:
        df = pd.get_dummies(df, columns=[kolom], prefix=kolom)
        return df, None
    
    else:
        raise ValueError("Parameter 'jenis' harus 1 (Label Encoding) atau 2 (One Hot Encoding)")


data1, encoder_gender = preprocessingData(data, 'gender', 1)
data2, encoder_smokinghist = preprocessingData(data1, 'smoking_history', 2)
data2


# **Penjelasan :**
# 
# Berdasarkan perbedaan jenis variabel sebelumnya, dilakukan proses encoding untuk variabel kategorik agar berubah pada tipe numerik. Jenis encoding untuk variabel gender adalah Label Encoder sehingga jenis kelamin akan dinyatakan oleh nilai 0, 1, dan 2. Variabel smoking_history menggunakan encoder berupa One Hot Encoding, sehingga setiap kategori akan dipisahkan menjadi sebanyak 6 kolom yang berisi nilai boolean True (1) atau False (0).

# ### 3.2 Splitting Data

# In[11]:


y = data2["diabetes"]
X = data2.drop(columns=["diabetes"])

# Membagi data menjadi train dan test
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()

# Fit dan transform data training
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

os.makedirs("Encoder Tersimpan", exist_ok=True)
scaler_path = "Encoder Tersimpan/minmax_scaler.pkl"
# Simpan scaler dengan benar
with open(scaler_path, 'wb') as file:
    pickle.dump(scaler, file)
print(f"Scaler disimpan di: {scaler_path}")


# **Penjelasan :**
# 
# Didefinisikan y dan X berdasarkan seluruh variabel yang digunakan di awal. Selain itu dilakukan splitting dengan proporsi 80% : 20% menggunakan mekanisme stratify terhadap kategori untuk memperhatikan distribusi label untuk data latih dan data uji. Selanjutnya akan dilakukan normalisasi menggunakan metode MinMaxScaler terhadap data latih dan data uji. 

# ### 3.3 Resampling

# In[12]:


def overSampling(X_train, y_train, rasio):
    # Cek distribusi awal kelas
    print("Distribusi sebelum SMOTE:", Counter(y_train))

    # Inisialisasi SMOTE dengan sampling_strategy 0.5
    smote = SMOTE(sampling_strategy=rasio, random_state=42)

    # Terapkan SMOTE pada data latih
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Cek distribusi setelah resampling
    print("Distribusi setelah SMOTE:", Counter(y_train))
    return X_train, y_train


# **Penjelasan :**
# 
# Fungsi ini digunakan untuk melakukan proses oversampling sebagai mekanisme mengatasi efek *imbalanced dataset* sehingga model tidak didominasi class mayoritas. Input pada fungsi ini berupa variabel prediktor untuk data latih **(X_train)** dan variabel target pada data latih **(y_train)** serta rasio, dimana **rasio** menyatakan rasio distribusi class minoritas yang ingin ditambahkan.

# ## 4. Modelling

# ### 4.1 Modelling dengan KNN yang diikuti Hyperparameter Tuning menggunakan Optuna

# In[13]:


def objective_knn(trial):
    params = {
        'n_neighbors': trial.suggest_int('n_neighbors', 3, 10),
        'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
        'p': trial.suggest_int('p', 1, 2)  # Parameter untuk metrik Minkowski (1=Manhattan, 2=Euclidean)
    }
    
    model = KNeighborsClassifier(**params)
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    score = balanced_accuracy_score(y_test, y_pred_test)
    return score

study_knn = optuna.create_study(direction='maximize')
study_knn.optimize(objective_knn, n_trials=10)

# Menampilkan hasil terbaik
print("Best KNN Parameters:", study_knn.best_params)

modelKNN1 = KNeighborsClassifier(**study_knn.best_params)


# ### 4.2 Modelling dengan CatBoost Classifier yang diikuti Hyperparameter Tuning menggunakan Optuna

# In[14]:


def objective_catboost(trial):
    params = {
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10),
        'random_state': 42,
        'verbose': 0
    }
    
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    score = balanced_accuracy_score(y_test, y_pred_test)
    return score

study_catboost = optuna.create_study(direction='maximize')
study_catboost.optimize(objective_catboost, n_trials=10)

print("Best CatBoost Parameters:", study_catboost.best_params)
modelCB1 = CatBoostClassifier(**study_catboost.best_params, random_state=42, verbose=0)


# **Penjelasan :**
# 
# Tahapan selanjutnya adalah pemodelan yang dilakukan menggunakan algoritma KNN dan CatBoost. Dalam menentukan hyperparameter yang dapat mengoptimalkan kedua algoritma, digunakan library optuna yang menerapkan **Algoritma Tree Parzen Structure** untuk melakukan pencarian hyperparameter atau dikenal dengan *hyperparameter tuning* untuk algoritma KNN dan CatBoost.

# ## 5. Evaluation

# ### 5.1 Pendefinisian Fungsi Evaluasi dan Visualisasi Performa Model

# In[15]:


def evaluasiModel(judul, model, X_train, y_train, X_test, y_test, save_dir="Model Tersimpan"):
    print(judul)
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)

    # Evaluasi metrik
    bal_acc = balanced_accuracy_score(y_test, y_test_pred)
    f1_macro = f1_score(y_test, y_test_pred, average='macro')
    acc = accuracy_score(y_test, y_test_pred)

    # Cek apakah model mendukung predict_proba atau decision_function
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = None

    # Hitung AUC jika memungkinkan
    auc_score = roc_auc_score(y_test, y_score) if y_score is not None else None

    print("Final Balanced Accuracy pada Data Uji :", bal_acc)
    print("F1 Score (Macro) : ", f1_macro)
    print("Accuracy : ", acc)
    if auc_score is not None:
        print("AUC Score:", auc_score)
    else:
        print("AUC Score: Tidak tersedia")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['Pred 0', 'Pred 1'], 
                yticklabels=['True 0', 'True 1'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

    # Simpan model
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{judul.replace(' ', '_')}.pkl")
    # Simpan model dengan benar
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model disimpan di: {model_path}")

    return {
        "Skenario": judul,
        "Balanced Accuracy": bal_acc,
        "F1 Score (Macro)": f1_macro,
        "Accuracy": acc,
        "AUC": auc_score,
        "Model Path": model_path
    }


# **Penjelasan :**
# 
# Fungsi ini bertujuan untuk melakukan evaluasi terhadap model machine learning yang dibangun. Terdapat 6 inputan judul, model, X_train, y_train, X_test, dan y_test. Kode ini bertujuan untuk menghasilkan ukuran evaluasi berupa Balanced Accuracy, Classification Report, ROC-AUC, dan Confusion Matrix. 

# ### 5.2 Evaluasi Model Klasifikasi

# In[16]:


hasil1 = evaluasiModel("====== Evaluasi Model KNN ======", modelKNN1, X_train, y_train, X_test, y_test)
hasil2 = evaluasiModel("====== Evaluasi Model Catboost ======",modelCB1, X_train, y_train, X_test, y_test)


# **Penjelasan :**
# 
# Berdasarkan pelatihan yang dilakukan terlihat bahwa model yang dibangun sudah cukup baik, di mana nilai akurasi berada di nilai > 95%. Pada hasil pencarian optuna untuk kedua algoritma diperoleh hyperparameter terbaik adalah sebagai berikut.
# 
# ***Model KNN = {'n_neighbors': 3, 'weights': 'distance', 'p': 2}*** <br>
# 
# ***Model CatBoost = {'depth': 9, 'learning_rate': 0.23318511591740917, 'iterations': 926, 'border_count': 99, 'bagging_temperature': 0.5500581986623337, 'l2_leaf_reg': 6.570372884152365}***
# 
# Namun, pada nilai F-1 Score Macro dan Balanced Accuracy yang cenderung lebih tepat digunakan untuk mengukur performa model klasifikasi pada kondisi imbalanced dataset, terlihat bahwa nilai Balanced Accuracy dan F-1 Score Macro pada model CatBoost lebih tinggi dibanding KNN secara berturut-turut adalah 85 dan 89. Hal ini menunjukkan model CatBoost lebih baik dalam memprediksi pada kedua class. 

# ### 5.3 Pemberian perlakuan resampling dengan SMOTE untuk mengatasi efek *imbalanced dataset*

# In[17]:


X_train, y_train = overSampling(X_train, y_train, 0.5)

print("========== KNN dengan SMOTE ==========")
study_knn = optuna.create_study(direction='maximize')
study_knn.optimize(objective_knn, n_trials=10)

# Menampilkan hasil terbaik
print("Best KNN Parameters:", study_knn.best_params)

modelKNN2 = KNeighborsClassifier(**study_knn.best_params)

print("========== CatBoost dengan SMOTE ==========")
study_catboost = optuna.create_study(direction='maximize')
study_catboost.optimize(objective_catboost, n_trials=10)

print("Best CatBoost Parameters:", study_catboost.best_params)
modelCB2 = CatBoostClassifier(**study_catboost.best_params, random_state=42, verbose=0)


# ### 5.4 Evaluasi Hasil Perlakuan Resampling pada Algoritma KNN dan Catboost

# In[18]:


hasil3 = evaluasiModel("====== Evaluasi Model KNN-SMOTE ======", modelKNN2, X_train, y_train, X_test, y_test)
hasil4 = evaluasiModel("====== Evaluasi Model CatBoost-SMOTE ======",modelCB2, X_train, y_train, X_test, y_test)


# ### 5.5 Final Evaluasi Model

# In[19]:


df_hasil = pd.DataFrame([hasil1, hasil2, hasil3, hasil4])
df_hasil


# **Penjelasan :**
# 
# Berdasarkan hasil evaluasi pada keempat skenario yang diujikan terlihat bahwa penerapan teknik oversampling pada algoritma KNN berhasil meningkatkan balanced accuracy dan AUC pada model KNN namun menurunkan ukuran evaluasi F1-Score Macro dan Accuracy pada moodel tersebut. Namun, kondisi berbeda ditunjukkan oleh algoritma CatBoost, dimana tidak terdapat peningkatan hasil setelah SMOTE, bahkan terdapat penurunan performa, walaupun tidak signifikan. 
# 
# Berdasarkan analisis ini dapat disimpulkan penerapan teknik oversampling secara umum tidak memberikan peningkatan performa yang signifikan pada model yang digunakan. Selain itu model yang dibangun sudah cukup mengklasifikasikan data dengan baik, sehingga dapat diimplementasikan dalam pembuatan model prediksi diabetes.
