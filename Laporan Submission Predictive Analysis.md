# Laporan Proyek Machine Learning - Nama Anda

## Domain Proyek

Diabetes melitus merupakan salah satu penyakit tidak menular yang paling mematikan dan membebani kondisi kesehatan global. Berdasarkan data dari International Diabetes Federation, pada tahun 2025 dilaporkan 11.1% populasi penduduk dunia atau 1 dari 9 orang dewasa berusia 20 - 79 tahun hidup dengan diabetes, di mana hampir 4 dari 10 orang tidak peduli terhadap kondisi yang mereka hadapi [1](https://idf.org/about-diabetes/diabetes-facts-figures/). Menurut IDF, Indonesia menduduki peringkat kelima negara dengan jumlah diabetes terbanyak dengan 19,5 juta penderita di tahun 2021 dan diprediksi akan menjadi 28,6 juta pada 2045 [2](https://sehatnegeriku.kemkes.go.id/baca/blog/20240110/5344736/saatnya-mengatur-si-manis/#:~:text=Angka%20ini%20diprediksi%20akan%20terus%20meningkat%20mencapai,diabetes%20melitus%20merupakan%20ibu%20dari%20segala%20penyakit.). Perkiraan peningkatan tersebut menunjukkan peluang penyakit tersebut untuk berkembang tentu cukup besar.

Penyakit diabetes, terutama tipe 2, berkembang secara perlahan dan seringkali tidak terdiagnosis sampai terjadi komplikasi serius. Komplikasi tersebut dapat berupa penyakit jantung, stroke, gagal ginjal, kebutaan, bahkan amputasi anggota tubuh. Beban ekonomi yang ditimbulkan pun sangat besar, baik dari segi biaya pengobatan jangka panjang maupun hilangnya produktivitas. Salah satu upaya penting dalam menangani masalah ini adalah dengan melakukan deteksi dini terhadap individu yang berisiko tinggi terkena diabetes. Mengacu pada kondisi tersebut, pengembangan model prediksi untuk mengukur kerentanan seseorang dapat terkena diabetes dapat menjadi sangat relevan. Hal ini dikarenakan metode tersebut memungkinkan identifikasi dini terhadap individu dengan kerentanan tinggi, sehingga intervensi medis dan perubahan gaya hidup bisa dilakukan lebih awal.

Dalam membuat model prediksi khususnya pada lingkup medis tentu dibutuhkan metode yang dapat menangkap hubungan kompleks dari beberapa variabel prediktor untuk memperkirakan kondisi medis secara akurat. Dalam mencapai kondisi tersebut, algoritma Machine Learning sering digunakan karena memiliki performa yang baik dalam menangkap hubungan kompleks terutama pada data dengan jumlah besar [3](https://www.sciencedirect.com/science/article/pii/S0169260722001596). Selain itu, implementasi machine learning dalam membentuk model prediksi untuk kasus medis terbukti memiliki performa yang baik dalam menangkap hubungan antar variabel [4](https://www.sciencedirect.com/science/article/abs/pii/S0167739X21004325)[5](https://academic.oup.com/jcem/article/106/3/e1191/6031346).

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
Dataset yang digunakan dalam penelitian ini diperoleh dari Hugging Face Hub terkait Dataset Diabetes [Dataset Diabetes](https://huggingface.co/datasets/m2s6a8/diabetes_prediction_dataset). Dataset tersebut terdiri dari 9 variabel dengan 100.000 observasi. Variabel penyusun dataset dapat dilihat pada keterangan berikut.
- gender : menyatakan jenis kelamin dari responden, terdapat 3 isian yang diberikan pada data yaitu Female, Male, dan Other.
- age : menyatakan usia dari responden sehingga menerima inputan berupa nilai numerik positif.
- hypertension : menyatakan riwayat hipertensi dari responden sehingga isian akan berupa jawaban 0 (Tidak) atau 1 (Ya).
- heart_disease : menyatakan riwayat penyakit hati dari responden sehingga isian akan berupa jawaban 0 (Tidak) atau 1 (Ya).
- smoking_history : menyatakan riwayat merokok dari responden. Terdapat 6 opsi yang dapat dipilih yaitu No Info, Never, Former, Current, Not Current, dan Ever.
- bmi : menyatakan body mass index dari responden.
- HbA1c_level : menyatakan kadar hemoglobin terglikasi dalam darah yang umum digunakan untuk mengevaluasi kadar gula darah rata-rata seseorang pada periode tertentu.
- blood_glucose_level : menyatakan kadar gula darah dalam tubuh seseorang.
- diabetes **(variabel target)** : menyatakan klasifikasi seseorang dikategorikan diabetes (1) atau tidak terkena diabetes (0).


Paragraf awal bagian ini menjelaskan informasi mengenai data yang Anda gunakan dalam proyek. Sertakan juga sumber atau tautan untuk mengunduh dataset. Contoh: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data).

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
- cuisine : merupakan jenis masakan yang disajikan pada restoran.
- dst

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

