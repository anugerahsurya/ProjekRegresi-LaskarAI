# Laporan Proyek Machine Learning - Anugerah Surya Atmaja

- **Nama:** Anugerah Surya Atmaja
- **Email Dicoding :** atmajasuryaanugerah@gmail.com
- **Email Laskar :** a406ybm064@devacademy.id
- **ID Dicoding:** anugerahhsurya

## Domain Proyek

Diabetes melitus merupakan salah satu penyakit tidak menular yang paling mematikan dan membebani kondisi kesehatan global. Berdasarkan data dari International Diabetes Federation, pada tahun 2025 dilaporkan 11.1% populasi penduduk dunia atau 1 dari 9 orang dewasa berusia 20 - 79 tahun hidup dengan diabetes, di mana hampir 4 dari 10 orang tidak peduli terhadap kondisi yang mereka hadapi [(IDF, 2025)](https://idf.org/about-diabetes/diabetes-facts-figures/). Menurut IDF, Indonesia menduduki peringkat kelima negara dengan jumlah diabetes terbanyak dengan 19,5 juta penderita di tahun 2021 dan diprediksi akan menjadi 28,6 juta pada 2045 [(Kemenkes, 2024)](https://sehatnegeriku.kemkes.go.id/baca/blog/20240110/5344736/saatnya-mengatur-si-manis/#:~:text=Angka%20ini%20diprediksi%20akan%20terus%20meningkat%20mencapai,diabetes%20melitus%20merupakan%20ibu%20dari%20segala%20penyakit.). Perkiraan peningkatan tersebut menunjukkan peluang penyakit tersebut untuk berkembang tentu cukup besar.

Penyakit diabetes, terutama tipe 2, berkembang secara perlahan dan seringkali tidak terdiagnosis sampai terjadi komplikasi serius. Komplikasi tersebut dapat berupa penyakit jantung, stroke, gagal ginjal, kebutaan, bahkan amputasi anggota tubuh. Beban ekonomi yang ditimbulkan pun sangat besar, baik dari segi biaya pengobatan jangka panjang maupun hilangnya produktivitas. Salah satu upaya penting dalam menangani masalah ini adalah dengan melakukan deteksi dini terhadap individu yang berisiko tinggi terkena diabetes. Mengacu pada kondisi tersebut, pengembangan model prediksi untuk mengukur kerentanan seseorang dapat terkena diabetes dapat menjadi sangat relevan. Hal ini dikarenakan metode tersebut memungkinkan identifikasi dini terhadap individu dengan kerentanan tinggi, sehingga intervensi medis dan perubahan gaya hidup bisa dilakukan lebih awal.

Dalam membuat model prediksi khususnya pada lingkup medis tentu dibutuhkan metode yang dapat menangkap hubungan kompleks dari beberapa variabel prediktor untuk memperkirakan kondisi medis secara akurat. Dalam mencapai kondisi tersebut, algoritma Machine Learning sering digunakan karena memiliki performa yang baik dalam menangkap hubungan kompleks terutama pada data dengan jumlah besar [(Olisah et al., 2022)](https://www.sciencedirect.com/science/article/pii/S0169260722001596). Selain itu, implementasi machine learning dalam membentuk model prediksi untuk kasus medis terbukti memiliki performa yang baik dalam menangkap hubungan antar variabel [(Wu et al., 2022)](https://www.sciencedirect.com/science/article/abs/pii/S0167739X21004325)[(Wu et al., 2020)](https://academic.oup.com/jcem/article/106/3/e1191/6031346).

Oleh karena itu, penelitian ini bertujuan untuk mengembangkan dan mengevaluasi model prediksi kerentanan diabetes berbasis data, yang diharapkan dapat membantu pihak medis maupun individu untuk mengambil keputusan preventif secara lebih efektif.

## Business Understanding

Berdasarkan latar belakang yang disampaikan, dapat didefinisikan hal yang ingin diselesaikan pada analisis ini sebagai berikut :

### Problem Statements

- Tingginya prevalensi diabetes di Indonesia dan dunia menunjukkan perlunya upaya deteksi dini yang lebih efektif dan efisien untuk mencegah komplikasi jangka panjang.
- Proses diagnosis konvensional seperti pemeriksaan laboratorium memerlukan waktu, biaya, dan tenaga medis yang tidak sedikit, sehingga belum optimal untuk skrining massal.
- Dibutuhkan algoritma yang optimal dalam membentuk model prediksi yang akurat.
- Diperlukan mekanisme sederhana serta dapat dengan mudah diterapkan untuk setiap masyarakat dalam mendeteksi kerentanan diabetes.

### Goals

- Mengembangkan model prediksi kerentanan terhadap diabetes menggunakan algoritma machine learning berbasis data kesehatan individu.
- Membandingkan performa machine learning dalam membentuk model prediksi kerentanan diabetes berdasarkan ukuran evaluasi.
- Menguji efektivitas teknik oversampling dalam mengatasi kondisi imbalanced dataset pada data medis.
- Membentuk dashboard yang dapat digunakan untuk mendeteksi kerentanan diabetes seseorang.

### Solution statements

- Dalam membentuk model prediksi, akan digunakan algoritma machine learning serta menggunakan dataset yang diperoleh dari hugging face terkait kondisi diabetes dengan jumlah observasi sebanyak 100.000. Dalam mengevaluasi performa model akan digunakan ukuran evaluasi berupa Accuracy, Balanced Accuracy, F-1 Score Macro, dan ROC-AUC.
- Dalam menguji efektivitas teknik oversampling, digunakan metode SMOTE yang bertujuan menambah distribusi data pada class minoritas. Selanjutnya perlakuan ini akan dibandingkan dengan model yang dibentuk tanpa perlakuan oversampling. Hasil pemodelan akan dibanding menggunakan ukuran evaluasi yang sama.
- Dalam membuat mekanisme yang dapat diakses dengan mudah, dibentuk dashboard yang menerima inputan pengguna berdasarkan variabel penyusun untuk model klasifikasi kerentanan diabetes. Dashboard dapat diujikan dengan mencoba input serta meninjau hasil prediksi yang diberikan.

## Data Understanding

Dataset yang digunakan dalam penelitian ini diperoleh dari Hugging Face Hub terkait Dataset Diabetes [Dataset Diabetes](https://huggingface.co/datasets/m2s6a8/diabetes_prediction_dataset). Dataset tersebut terdiri dari 9 variabel dengan 100.000 observasi.

### Variabel penyusun dataset dapat dilihat pada keterangan berikut.

- gender : menyatakan jenis kelamin dari responden, terdapat 3 isian yang diberikan pada data yaitu Female, Male, dan Other.
- age : menyatakan usia dari responden sehingga menerima inputan berupa nilai numerik positif.
- hypertension : menyatakan riwayat hipertensi dari responden sehingga isian akan berupa jawaban 0 (Tidak) atau 1 (Ya).
- heart_disease : menyatakan riwayat penyakit hati dari responden sehingga isian akan berupa jawaban 0 (Tidak) atau 1 (Ya).
- smoking_history : menyatakan riwayat merokok dari responden. Terdapat 6 opsi yang dapat dipilih yaitu No Info, Never, Former, Current, Not Current, dan Ever.
- bmi : menyatakan body mass index dari responden.
- HbA1c_level : menyatakan kadar hemoglobin terglikasi dalam darah yang umum digunakan untuk mengevaluasi kadar gula darah rata-rata seseorang pada periode tertentu.
- blood_glucose_level : menyatakan kadar gula darah dalam tubuh seseorang.
- diabetes **(variabel target)** : menyatakan klasifikasi seseorang dikategorikan diabetes (1) atau tidak terkena diabetes (0).

### Analisis Deksriptif Data

<figure>
  <img src="Visualisasi/Analisis Deskriptif Data.png" alt="Analisis Deskriptif Data" width="600"/>
  <figcaption><b>Gambar 1.</b> Analisis deskriptif data</figcaption>
</figure>
Berdasarkan analisis deskriptif yang dilakukan pada Gambar 1, terlihat bahwa tidak terdapat missing value pada data yang ditandai dengan baris count, di mana jumlahnya sama untuk seluruh variabel. Pada variabel numerik terdapat 7 variabel seperti age, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level, dan diabetes. Terlihat berdasarkan keseluruhan data, responden berada pada usia 41 - 42 tahun. Selain itu terlihat standar deviasi usia berada di nilai 22 - 23 tahun, hal tersebut menunjukkan kondisi dimana terdapat keberagaman usia responden. Selain itu, pada variabel hypertension terlihat rata-rata nilai berada di angka 0.07 dan memiliki standar deviasi cukup besar yaitu 0.26, di mana nilainya lebih besar dibanding rataannya. Hal ini juga ditunjukkan pada variabel heart_disease, di mana nilainya juga lebih besar dibanding rataannya.

Pada variabel bmi yang menyatakan Body Mass Index terlihat bahwa rataan responden berada pada nilai 27 - 28. Pada variabel HbA1c_level terlihat nilai minimum 3.5 dan max 9. Nilai rataan menunjukkan nilai 5.52 karena mendekati nilai minimum. Pada variabel blood_glucose_level terlihat bahwa responden berada ada gula rata-rata 138. Terakhir, pada variabel target yaitu status diabetes terlihat bahwa responden cenderung berada pada kondisi tidak terkena diabetes.

### Visualisasi Variabel Prediktor

<figure>
  <img src="Visualisasi/Visualisasi Variabel Prediktor.png" alt="Visualisasi Prediktor" width="600"/>
  <figcaption><b>Gambar 2.</b> Sebaran Distribusi Variabel Prediktor</figcaption>
</figure>

Pada sebaran untuk variabel prediktor, terlihat bahwa responden didominasi pada jenis kelamin Laki-laki. Selain itu usia responden juga didominasi pada penduduk Lansia, di mana berusia 80 Tahun. Pada visualisasi terkait Riwayat Merokok juga terlihat bahwa responden didominasi oleh keterangan Tanpa Informasi.

<figure>
  <img src="Visualisasi/Visualisasi Variabel Target.png" alt="Visualisasi Distribusi Label" width="300"/>
  <figcaption><b>Gambar 3.</b> Distribusi Class pada Data</figcaption>
</figure>

Pada visualisasi yang diperoleh terlihat pada distribusi label didominasi dengan kondisi tidak terkena diabetes. Hal ini umum terjadi pada data medis, di mana kondisi tidak terkena penyakit lebih sering dijumpai dibanding dengan kondisi terkena penyakit. Kondisi ini dapat menjadi indikasi efek dari imbalanced dataset berupa model yang mendominasi class mayoritas sehingga class minoritas sulit dikenali.

## Data Preparation

Tahapan ini bertujuan untuk memberikan perlakuan pada data agar dapat menyesuaikan dengan algoritma yang digunakan. Berikut pemrosesan yang dilakukan pada Tahapan ini.

### 1. Encoding Variabel Kategorik

Berdasarkan perbedaan jenis variabel sebelumnya, dilakukan proses encoding untuk variabel kategorik agar berubah pada tipe numerik. Jenis encoding untuk variabel gender adalah Label Encoder sehingga jenis kelamin akan dinyatakan oleh nilai 0, 1, dan 2. Variabel smoking_history menggunakan encoder berupa One Hot Encoding, sehingga setiap kategori akan dipisahkan menjadi sebanyak 6 kolom yang berisi nilai boolean True (1) atau False (0).

### 2. Splitting Data

Didefinisikan y dan X berdasarkan seluruh variabel yang digunakan di awal. Selain itu dilakukan splitting dengan proporsi 80% : 20% menggunakan mekanisme stratify terhadap kategori untuk memperhatikan distribusi label untuk data latih dan data uji.

### 3. Normalisasi Data

Tahapan selanjutnya adalah akan dilakukan normalisasi menggunakan metode MinMaxScaler terhadap data latih dan data uji. Metode ini membuat setiap data pada variabel prediktor akan berada pada rentang [0, 1]. Hal ini digunakan agar meminimalisir efek skala data pada algoritma KNN. Hal ini pada dasarnya tidak diperlukan pada algoritma CatBoost karena algoritma berbasis tree tidak dipengaruhi skala data. Namun penggunaan normalisasi tetap tidak akan merubah hasil analisis pada algoritma CatBoost.

## Modeling

### Hyperparameter Tuning Algoritma Machine Learning
Pemodelan yang dilakukan pada analisis ini akan menggunakan Algoritma KNN dan Algoritma CatBoost. Kedua metode ini dipilih karena kemampuannya yang baik untuk diimplementasikan pada kasus klasifikasi [(Suyal & Goyal, 2022)](https://ijettjournal.org/archive/ijett-v70i7p205)[(Chang et al., 2023)](https://doi.org/10.3390/s23041811). K-Nearest Neighbors (KNN) merupakan algoritma klasifikasi berbasis instance-based learning, di mana model tidak melakukan proses pelatihan eksplisit (lazy learner). Sebaliknya, saat diberikan data baru, KNN membandingkannya dengan seluruh data latih yang tersedia. Algoritma ini bekerja dengan cara mencari sejumlah 𝑘tetangga terdekat berdasarkan ukuran jarak tertentu. Kelas dari data baru kemudian diprediksi berdasarkan mayoritas kelas dari tetangga-tetangga tersebut. Semakin kecil nilai k, model akan lebih sensitif terhadap noise, sedangkan nilai k yang terlalu besar dapat menyebabkan prediksi menjadi kurang akurat karena mempertimbangkan terlalu banyak tetangga yang tidak relevan [(Cover & Hart, 1967)](https://doi.org/10.1109/TIT.1967.1053964). KNN memiliki kelebihan dalam kesederhanaan implementasi dan efektivitas pada dataset kecil, namun kelemahannya terletak pada kebutuhan komputasi tinggi saat prediksi, terutama jika jumlah data latih sangat besar.

Algoritma selanjutnya yang digunakan yaitu CatBoost . Algoritma ini merupakan algoritma berbasis gradient boosting yang dirancang khusus untuk menangani data kategorikal secara lebih efisien tanpa perlu pra-pemrosesan yang kompleks [(Prokhorenkova et al., 2018)](https://doi.org/10.48550/arXiv.1706.09516). Dalam klasifikasi, CatBoost membangun model secara iteratif melalui serangkaian pohon keputusan decision trees yang mengoptimalkan kesalahan prediksi secara bertahap. Keunggulan utama CatBoost dibandingkan algoritma boosting lainnya adalah kemampuannya mengurangi overfitting melalui teknik pengaturan seperti Ordered Boosting dan penggunaan kombinasi fitur kategorikal secara otomatis. CatBoost juga mendukung penggunaan GPU untuk mempercepat proses pelatihan, menjadikannya cocok untuk dataset besar dan kompleks. Selain itu, CatBoost secara konsisten menunjukkan performa unggul dalam banyak tugas klasifikasi, bahkan pada data dengan fitur kategorikal yang dominan.

Dalam menentukan hyperparameter yang dapat mengoptimalkan model, digunakan algoritma untuk proses _Hyperparameter Tuning_ dengan library optuna yang menerapkan algoritma _Tree-Structured Parzen Estimator_ [(Watanabe, 2023)](https://arxiv.org/abs/2304.11127). Dalam menemukan *hyperparameter* terbaik untuk algoritma CatBoost dan KNN, digunakan konfigurasi ruang pencarian yang dapat dilihat pada tabel berikut.

**Tabel 1.** Ruang Pencarian Hyperparameter untuk Algoritma CatBoost
| Hyperparameter     | Tipe Data | Range Pencarian             |
|--------------------|-----------|-----------------------------|
| depth              | Integer   | [4, 10]                 |
| learning_rate      | Float     | [0.01, 0.3]             |
| iterations         | Integer   | [100, 1000]             |
| border_count       | Integer   | [32, 255]               |
| bagging_temperature| Float     | [0, 1]                  |
| l2_leaf_reg        | Float     | [0.001, 10]             |

**Tabel 2.** Ruang Pencarian Hyperparameter untuk Algoritma KNN

| Hyperparameter | Tipe Data     | Range Pencarian                  |
|----------------|---------------|----------------------------------|
| n_neighbors    | Integer        | [3, 10]                     |
| weights        | Kategori       | {'uniform', 'distance'}           |
| p              | Integer        | [1, 2]                     |

Berdasarkan pemodelan yang dilakukan serta proses hyperparameter tuning yang dibatasi dengan trial sebanyak 10 untuk mengefisienkan pencarian diperoleh kedua algoritma optimal pada konfigurasi berikut.

1. Best KNN Parameters: {'n_neighbors': 3, 'weights': 'distance', 'p': 2}
2. Best CatBoost Parameters: {'depth': 9, 'learning_rate': 0.23318511591740917, 'iterations': 926, 'border_count': 99, 'bagging_temperature': 0.5500581986623337, 'l2_leaf_reg': 6.570372884152365}

### Penerapan Teknik Oversampling : SMOTE
Dalam mengatasi kondisi ketidakseimbangan data, dilakukan juga skenario untuk menambahkan treatment berupa Oversampling dengan **SMOTE** [(Chawla et al., 2002)](https://arxiv.org/pdf/1106.1813).  SMOTE berguna untuk menambah distribusi data pada class minoritas berdasarkan hyperparameter bernama **Sampling Strategy**. Kondisi ini berarti jika **Sampling Strategy** diatur menjadi 0.5, maka class minoritas akan ditambah hingga mencapai 0.5 dari jumlah class mayoritas. Berdasarkan teknik oversampling yang diterapkan pada dtaa latih, model kembali dilatih dan dilakukan proses hyperparameter tuning sehingga diperoleh pada skenario dengan SMOTE, ditemukan model optimal dengan konfigurasi berikut.

1. Best KNN Parameters: {'n_neighbors': 9, 'weights': 'distance', 'p': 2}
2. Best CatBoost Parameters: {'depth': 4, 'learning_rate': 0.29937529311170064, 'iterations': 879, 'border_count': 220, 'bagging_temperature': 0.32996767322674414, 'l2_leaf_reg': 2.2683208191054964}

Diperoleh konfigurasi berbeda yang menunjukkan bahwa model optimal dipengaruhi oleh teknik oversampling yang diterapkan.

## Evaluation

<figure>
  <img src="Visualisasi/Confusion Matrix Gabungan.png" alt="Visualisasi Confusion Matrix" width="600"/>
  <figcaption><b>Gambar 4.</b> Sebaran Distribusi Variabel Prediktor</figcaption>
</figure>

Ukuran evaluasi yang digunakan untuk menilai kebaikan model dan skenario yang didefinisikan adalah sebagai berikut.

### Rumus Ukuran Evaluasi Model

#### 1. Balanced Accuracy

Balanced Accuracy digunakan untuk mengatasi ketidakseimbangan kelas, dan didefinisikan sebagai rata-rata dari sensitivitas (recall) dan spesifisitas:

$$
\text{Balanced Accuracy} = \frac{1}{2} \left( \frac{TP}{TP + FN} + \frac{TN}{TN + FP} \right)
$$

#### 2. F1 Score (Macro)

F1 Score menggabungkan precision dan recall. Pendekatan "macro" menghitung skor F1 untuk setiap kelas dan mengambil rata-rata tanpa mempertimbangkan proporsi kelas:

$$
\text{F1 Score (Macro)} = \frac{1}{N} \sum_{i=1}^{N} \frac{2 \cdot \text{Precision}_i \cdot \text{Recall}_i}{\text{Precision}_i + \text{Recall}_i}
$$

#### 3. Accuracy

Akurasi mengukur proporsi prediksi yang benar terhadap total data:

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

#### 4. AUC (Area Under Curve)

AUC mengukur kemampuan model dalam membedakan antara kelas positif dan negatif, biasanya merujuk pada luas di bawah kurva ROC (Receiver Operating Characteristic). Nilainya berada di antara 0 dan 1, semakin tinggi semakin baik.

$$
\text{AUC} = \int_0^1 TPR(FPR^{-1}(x)) \, dx
$$

Keterangan:

- **TP**: True Positive
- **TN**: True Negative
- **FP**: False Positive
- **FN**: False Negative
- **TPR**: True Positive Rate
- **FPR**: False Positive Rate

Ukuran evaluasi yang dihitung pada setiap model akan mengacu pada confusion matrix yang dapat dilihat pada Gambar 4. Komponen True Positive, True Negative, False Positive, dan False Negative digunakan untuk menghitung keempat ukuran evaluasi yang digunakan. Namun, hal ini sudah dapat dilakukan menggunakan library dari scikit-learn. Hasil evaluasi pada keempat skenario yang digunakan dapat dilihat pada Tabel 1.

### Tabel 1. Evaluasi Model pada Skenario Analisis

| No  | Skenario             | Balanced Accuracy | F1 Score (Macro) | Accuracy | AUC  |
| --- | -------------------- | ----------------- | ---------------- | -------- | ---- |
| 0   | Model KNN            | 0.78              | 0.82             | 0.95     | 0.86 |
| 1   | Model CatBoost       | 0.85              | 0.89             | 0.97     | 0.97 |
| 2   | Model KNN-SMOTE      | 0.84              | 0.79             | 0.93     | 0.92 |
| 3   | Model CatBoost-SMOTE | 0.85              | 0.89             | 0.97     | 0.98 |

**Penjelasan :**

Berdasarkan hasil evaluasi pada keempat skenario yang diujikan terlihat bahwa penerapan teknik oversampling pada algoritma KNN berhasil meningkatkan balanced accuracy dan AUC pada model KNN namun menurunkan ukuran evaluasi F1-Score Macro dan Accuracy pada moodel tersebut. Hal ini dapat dilihat pada nilai Balanced Accuracy yang sebelumnya 0.78, setelah menerapkan SMOTE menjadi 0.83. Perlakuan tersebut memberikan penurunan pada F-1 Score Macro yang awalnya bernilai 0.82 menjadi 0.79. Hal ini menunjukkan ketidakstabilan hasil model.

Namun, kondisi berbeda ditunjukkan oleh algoritma CatBoost, dimana terdapat peningkatan hasil setelah SMOTE pada ukuran evaluasi F-1 Score Macro, Accuracy, dan AUC walaupun tidak begitu signifikan. Hal ini menunjukkan penerapan teknik Oversampling terbukti efektif dalam meningkatkan informasi yang diperoleh model terkait sebaran data. Pada balanced accuracy terdapat penurunan yang kecil sebesar 0,002 setelah diberikan treatment oversampling, namun hal tersebut masih bisa dipertahankan karena ukuran lainnya menunjukkan peningkatan.

Berdasarkan analisis yang dilakukan dapat diperoleh kesimpulan sebagai berikut.

1. Model terbaik yang dapat digunakan untuk mengklasifikasikan kerentanan diabetes adalah model yang dilatih menggunakan Algoritma Catboost serta menerapkan Teknik Oversampling SMOTE. Hal ini ditunjukkan dengan ukuran evaluasi yang lebih baik dibanding skenario lainnya.
2. Teknik oversampling SMOTE terbukti efektif untuk meningkatakn performa model klasifikasi yang diukur melalui peningkatan performa berdasarkan evaluasi tanpa implementasi SMOTE.
3. Model yang dibangun sudah cukup mengklasifikasikan data dengan baik, sehingga dapat diimplementasikan dalam pembuatan model prediksi diabetes.
4. Dashboard penelitian dapat diakses melalui tautan berikut : [Dashboard Prediksi Kerentanan Diabetes](https://projekregresi-laskarai-kerentanan-diabetes.streamlit.app/)

## Deployment

Model yang dibentuk dideploy menggunakan streamlit yang dapat diakses pada tautan berikut.

[Dashboard Prediksi Kerentanan Diabetes](https://projekregresi-laskarai-kerentanan-diabetes.streamlit.app/)

User akan diminta untuk mengisi form berupa isian terkait kondisi kesehatan seperti Jenis Kelamin, Usia, Riwayat Hipertensi, Riwayat Penyakit Hati, Berat Badan, Tinggi Badan, HbA1c Level, Kadar Gula Darah, dan Riwayat Merokok. Berat badan dan tinggi badan digunakan untuk menghitung Body Mass Index (BMI). Hasil prediksi akan berupa keterangan kerentanan diabetes seseorang.

<figure>
  <img src="Visualisasi/Preview Hasil.png" alt="Hasil Prediksi pada Dashboard" width="600"/>
  <figcaption><b>Gambar 5.</b> Preview Laman Dashboard</figcaption>
</figure>

## Reference

Referensi ditulis menggunakan format APA.

1. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. Journal of Artificial Intelligence Research, 16, 321–357. https://doi.org/10.1613/jair.953

2. Chang, W., Wang, X., Yang, J., & Qin, T. (2023). An improved CatBoost-based classification model for ecological suitability of blueberries. Sensors, 23(4), 1811. https://doi.org/10.3390/s23041811

3. Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. IEEE Transactions on Information Theory, 13(1), 21–27. https://doi.org/10.1109/TIT.1967.1053964

4. International Diabetes Federation. (n.d.). Diabetes facts & figures. https://idf.org/about-diabetes/diabetes-facts-figures/

5. m2s6a8. (2024). diabetes_prediction_dataset [Dataset]. Hugging Face. https://huggingface.co/datasets/m2s6a8/diabetes_prediction_dataset

6. Olisah, C. C., Smith, L., & Smith, M. (2022). Diabetes mellitus prediction and diagnosis from a data preprocessing and machine learning perspective. Computer Methods and Programs in Biomedicine, 220, 106773. https://doi.org/10.1016/j.cmpb.2022.106773

7. Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: unbiased boosting with categorical features. Advances in Neural Information Processing Systems, 31. doi.org/10.48550/arXiv.1706.09516

8. Rokom. (2024, Januari 10). Saatnya mengatur si manis. Sehat Negeriku – Kementerian Kesehatan Republik Indonesia. https://sehatnegeriku.kemkes.go.id/baca/blog/20240110/5344736/saatnya-mengatur-si-manis/

9. Suyal, M., & Goyal, P. (2022). A review on analysis of K-nearest neighbor classification machine learning algorithms based on supervised learning. International Journal of Engineering Trends and Technology, 70(7), 43–48. https://doi.org/10.14445/22315381/IJETT-V70I7P205

10. Watanabe, S. (2023). Tree-structured parzen estimator: Understanding its algorithm components and their roles for better empirical performance. arXiv preprint arXiv:2304.11127. https://arxiv.org/abs/2304.11127

11. Wu, Y., Zhang, Q., Hu, Y., Sun-Woo, K., Zhang, X., Zhu, H., Jie, L., & Li, S. (2022). Novel binary logistic regression model based on feature transformation of XGBoost for type 2 diabetes mellitus prediction in healthcare systems. Future Generation Computer Systems, 129, 1–12. https://doi.org/10.1016/j.future.2021.11.003

12. Wu, Y.-T., Zhang, C.-J., Mol, B. W., Kawai, A., Li, C., Chen, L., Wang, Y., Sheng, J.-Z., Fan, J.-X., Shi, Y., & Huang, H.-F. (2021). Early prediction of gestational diabetes mellitus in the Chinese population via advanced machine learning. The Journal of Clinical Endocrinology & Metabolism, 106(3), e1191–e1205. https://doi.org/10.1210/clinem/dgaa899
