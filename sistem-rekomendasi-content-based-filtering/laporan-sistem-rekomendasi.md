# Laporan Proyek Machine Learning - Alfin Muhammad Ilmi

## Project Overview

Meningkatnya jumlah pecinta film selaras dengan banyaknya jumlah film yang diproduksi. Banyak film dengan genre atau aliran yang serupa maupun berbeda meramaikan industri perfilman di dalam negeri dan luar negeri. Saking banyaknya film yang diproduksi membuat calon penonton bingung dan kesulitan dalam menentukan film yang akan ditonton selanjutnya [[1]](http://j-ptiik.ub.ac.id/index.php/j-ptiik/article/download/9163/4159). Hal tersebut dapat menghabiskan banyak waktu dalam mencari film yang cocok untuk calon penonton. Agar calon penonton dapat mencari film sesuai dengan genre yang disukai, maka perlu sistem yang dapat merekomendasikan berbagai film sesuai dengan genre yang diinginkan.

## Business Understanding

### Problem Statements

- Bagaimana cara membuat sistem rekomendasi dengan metode terbaik untuk merekomendasi judul film berdasarkan genre atau aliran?

### Goals

- Mengetahui cara membuat sistem rekomendasi dengan metode terbaik untuk merekomendasikan judul film berdasarkan genre atau alirannya.

### Solution statements
- Menawarkan solusi sistem rekomendasi dengan teknik *content based filtering* yang menghasilkan rekomendasi judul film berdasarkan genre atau alirannya. Untuk mendapatkan solusi terbaik, akan digunakan dua metode yang berbeda yaitu *Cosine Similarity* dan *Euclidean Similarity*. Lalu untuk mengukur kinerja model akan digunakan metrik *Precision* dan lama waktu komputasi setiap modelnya.

## Data Understanding

Tabel 1. Informasi Dataset

| | Keterangan |
|---|---|
| Sumber | [Kaggle - Netflix Movies and TV Shows Dataset](https://www.kaggle.com/datasets/shivamb/netflix-shows) |
| Jumlah Data | 8807 |
| *Usability* | 10.00 |
| Lisensi | CC0: Public Domain |
| Jenis dan Ukuran Berkas | CSV (3.4 MB) |

### Deskripsi Variabel

Sesuai dengan informasi dari [Kaggle](https://www.kaggle.com/datasets/shivamb/netflix-shows), Variabel-variabel pada *Netflix Movies and TV Shows* Dataset adalah sebagai berikut:

* `show_id` adalah ID untuk setiap *Movie* / *TV Show*
* `type` adalah sebagai pengenal apakah termasuk pada kategori *Movie* / *TV Show*
* `title` adalah judul dari *Movie* / *TV Show*
* `director` adalah nama dari sutradara film (*Movie*)
* `cast` adalah aktor yang terlibat dalam *Movie* / *TV Show*
* `country` adalah negara tempat *Movie* / *TV Show* itu diproduksi
* `date_added` adalah tanggal *Movie* / *TV Show* ditambahkan ke Netflix
* `release_year` adalah tahun rilis sebenarnya dari *Movie* / *TV Show*
* `rating` adalah jenis-jenis rating *Movie* / *TV Show*
* `duration` adalah durasi total (dalam menit atau jumlah *season*)
* `listed_in` adalah genre atau aliran *Movie* / *TV Show*
* `description` adalah deskripsi ringkasan

### Menentukan Fitur yang Akan Digunakan

Dalam kasus ini akan direkomendasikan film berdasarkan genre atau alirannya saja. Sehingga pada kasus ini hanya membutuhkan kolom (fitur) `show_id`, `title`, dan `listed_in`. 

Selain itu, pada kasus ini hanya dipilih data yang bertipe `Movie`. Sehingga data yang bertipe `TV Show` tidak dibutuhkan.

Tabel 2. Lima contoh Dataset setelah Pemilihan Fitur

|index|show_id|title|listed_in|
|---|---|---|---|
|0 |s1|Dick Johnson Is Dead|Documentaries|
|6 |s7|My Little Pony: A New Generation	|Children & Family Movies|
|7 |s8|Sankofa|Dramas, Independent Movies, International Movies|
|9 |s10|The Starling	|Comedies, Dramas|
|12|s13|Je Suis Karl|Dramas, International Movies|

## Data Preparation
### Mengatasi Missing Value

Untuk mendeteksi *missing value* digunakan fungsi isnull().sum() dan untuk mendeteksi nilai NAN digunakan isna().sum().

Tabel 3. Hasil Deteksi *Missing Value*

| Fitur | Jumlah *Missing Value* |
|---|:---:|
| show_id   | 0 |
| title  | 0 |
| listed_in  | 0 |

Tabel 4. Hasil Deteksi Nilai NAN

| Fitur | Jumlah Nilai NAN |
|---|:---:|
| show_id   | 0 |
| title  | 0 |
| listed_in  | 0 |

Dari Tabel 3. dan Tabel 4. terlihat bahwa setiap fitur tidak memiliki Missing Value (NULL) maupun Nilai NAN sehingga dapat dilanjutkan ke tahapan selanjutnya yaitu menghilangkan data duplikat.

### Menghilangkan data Duplikat

Sebelum menghilangkan data duplikat, perlu melakukan pengecekan terlebih dahulu pada dataset.

Tabel 5. Perbandingan Banyak Data `title` yang *Unique* dengan Jumlah Seluruh Data

| | Unique title | Total Data |
|:---:|:---:|:---:|
| Jumlah Film | 6131 | 6131 |

Dapat dilihat bahwa jumlah `title` yang *unique* sama dengan total data. Ini menunjukkan bahwa setiap baris memiliki judul film yang berbeda-beda.

## Modeling

Di tahap ini, sistem rekomendasi dibuat menggunakan model dengan metode *Cosine Similarity* dan *Euclidean Similarity*. Tetapi sebelumnya akan dilakukan perubahan tipe data dari kategorikal menjadi data numerik menggunakan metode `TF-IDF Vectorizer`.

### TF-IDF Vectorizer

Di tahap ini akan dilakukan ekstraksi fitur sekaligus menyeleksi fitur yang sering muncul dan jarang muncul. Berikut daftar fitur hasil `TF-IDF Vectorizer`.

Tabel 6. Daftar Fitur Hasil `TF-IDF Vectorizer`

| Nama Fitur |
|---|
| 'action' |
| 'adventure' |
| 'anime' |
| 'children' |
| 'classic' |
| 'comedies' |
| 'comedy' |
| 'cult' |
| 'documentaries' |
| 'dramas' |
| 'faith' |
| 'family' |
| 'fantasy' |
| 'features' |
| 'fi' |
| 'horror' |
| 'independent' |
| 'international' |
| 'lgbtq' |
| 'movies' |
| 'music' |
| 'musicals' |
| 'romantic' |
| 'sci' |
| 'spirituality' |
| 'sports' |
| 'stand' |
| 'thrillers' |
| 'up' |

 Setelah itu, dilakukan *one-hot-encoding* pada setiap `title` terhadap fitur-fitur yang dihasilkan oleh `TF-IDF Vectorizer`. Berikut matrik TF-IDF yang dihasilkan.

 Tabel 7. Matrik TF-IDF dengan 10 Sampel pada Kolom dan 10 Sampel pada Baris

|title|sci|family|spirituality|children|features|stand|thrillers|movie|comedies|faith|
|---|---|---|---|---|---|---|---|---|---|---|
|Norm of the North: Keys to the Kingdom|0.0|0.680177|0.0|0.680177|0.0|0.0|0.0|0.273345|0.0|0.0|
|Andhaghaaram|0.0|0.0|0.0|0.0|0.0|0.0|0.559144|0.435371|0.0|0.0|
|Familiye|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.541509|0.0|0.0|
|DreamWorks Home: For the Holidays|0.0|0.431517|0.0|0.431517|0.0|0.0|0.0|0.173415|0.304451|0.0|
|Apache Warrior|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|
|Mr. Virgin|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.350455|0.615263|0.0|
|Have a Good Trip: Adventures in Psychedelics|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|
|Never Back Down|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.202607|0.0|0.0|
|Executive Decision|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.0|
|Five Elements Ninjas|0.0|0.0|0.0|0.0|0.0|0.0|0.0|0.275727|0.0|0.0|

Pada Tabel 7. di atas, matrik TF-IDF memiliki ukuran 6131 x 6131. Artinya adalah ada 6131 judul film yang berbeda.

### Cosine Similarity

Kelebihan dari metode *cosine similarity* adalah tidak bergantung pada besarnya vektor. Tetapi kelebihan tersebut dapat menjadi kekurangan jika pada kasus tertentu, makna frekuensi kemunculan fitur menjadi penting. Sedangkan pada kasus ini, *Cosine Similarity* aman digunakan karena sudah dilakukan tahap *one-hot-encoding* pada matrik tf-idf. Sehingga frekuensi tiap kategori pada produk mempunyai bobot yang sama yaitu 0 (tidak ada) atau 1 (ada).

Untuk implementasinya menggunakan fungsi `cosine_similarity()` dari *library* sklearn dengan lama waktu komputasinya sebagai berikut.
```
Execution Time Cosine Similarity (Seconds) : 1.0046014785766602
```

Dengan hasil matrik *Cosine Similarity* nya sebagai berikut.

Tabel 8. Matrik *Cosine Similarity* dengan 5 Sampel pada Kolom dan 10 Sampel pada Baris

| title| Chal Mere Bhai | A Futile and Stupid Gesture | Air Force One | Paying Guests | Lowriders |
|--------------------------|---------------:|----------------------------:|--------------:|--------------:|-----------|
|     War Against Women    | 0.352047       | 0.0                         | 0.0           | 0.418899      | 0.119662  |
|        God Calling       |       0.155752 | 0.0                         | 0.0           | 0.185328      | 0.167643  |
| Lupt                     |       0.327769 | 0.0                         | 0.0           | 0.346705      | 0.147201  |
|      The Grandmaster     | 0.254135       | 0.0                         | 0.818365      | 0.302394      | 0.273538  |
|         Catch.er         | 0.291335       | 0.0                         | 0.0           | 0.346658      | 0.313578  |
|       The Roommate       | 0.0            | 0.0                         | 0.0           | 0.0           | 0.0       |
|     American Hangman     | 0.0            | 0.0                         | 0.0           | 0.0           | 0.0       |
|   Le serment des Hitler  | 0.352047       | 0.0                         | 0.0           | 0.418899      | 0.119662  |
|        Eve's Apple       |       0.352047 | 0.0                         | 0.0           | 0.418899      | 0.119662  |
| What Makes a Psychopath? | 0.0            | 0.0                         | 0.0           | 0.0           | 0.0       |

### Euclidean Similarity

Kelebihan Euclidean adalah dapat memperoleh nilai perbedaan antara dua vektor yang sama arahnya namun beda besarannya. Sedangkan kekurangan algoritma ini adalah fitur dengan frekuensi kemunculan paling banyak akan mendominasi fitur lain dalam hasil komputasi jarak euclideannya.

Untuk implementasinya menggunakan fungsi `euclidean_distances()` dari *library* sklearn dengan lama waktu komputasinya sebagai berikut.

```
Execution Time Euclidean Similarity (Seconds) : 2.6405744552612305
```

Dengan hasil matrik *Euclidean Similarity* nya sebagai berikut.

Tabel 9. Matrik *Euclidean Similarity* dengan 5 Sampel pada Kolom dan 10 Sampel pada Baris

| title                   | Gerald's Game | Krish Trish and Baltiboy: The Greatest Trick | Secret Obsession | Little Singham: Legend of Dugabakka | An Ordinary Man |
|-------------------------|--------------:|---------------------------------------------:|-----------------:|------------------------------------:|----------------:|
|     Alex Strangelove    | 0.423335      | 0.424294                                     | 0.414214         | 0.456715                            | 0.443505        |
|        Cappuccino       |      0.425304 | 0.426481                                     | 0.414214         | 0.467742                            | 0.450564        |
|        The Worthy       |      0.457271 | 0.419533                                     | 0.477729         | 0.418998                            | 0.467089        |
|           Her           | 0.419036      | 0.419532                                     | 0.414214         | 0.418997                            | 0.429057        |
|    Bo Burnham: Inside   | 0.414214      | 0.414214                                     | 0.414214         | 0.414214                            | 0.414214        |
|       Bombay Rose       | 0.431840      | 0.433772                                     | 0.414214         | 0.431691                            | 0.441234        |
|       Skater Girl       | 0.428400      | 0.661924                                     | 0.414214         | 0.603386                            | 0.435785        |
| Manorama Six Feet Under | 0.513919      | 0.424372                                     | 0.586907         | 0.423331                            | 0.659798        |
|        Ainu Mosir       |      0.428476 | 0.430014                                     | 0.414214         | 0.428357                            | 0.462622        |
|          Romina         | 0.520550      | 0.436058                                     | 0.414214         | 0.433713                            | 0.414214        |

### Mendapatkan Rekomendasi

Tahap ini merupakan tahap pengujian hasil top-10 rekomendasi judul film. Sampel yang dipilih adalah sebagai berikut.

Tabel 10. Sampel Judul Film yang Digunakan 

|index|show_id|title|listed_in|
|---|---|---|---|
|826|s827|Bo Burnham: Inside|Stand-Up Comedy|

#### Rekomendasi dengan Cosine Similarity

Tabel 11. Top-10 Rekomendasi dengan Cosine Similarity

|index|show_id|title|listed_in|
|---|---|---|---|
|0|Bo Burnham: what.|s4792|Stand-Up Comedy|
|1|Joe Mande’s Award-Winning Comedy Special|s5370|Stand-Up Comedy|
|2|Aditi Mittal: Things They Wouldn't Let Me Say|s5372|Stand-Up Comedy|
|3|Alan Saldaña: Locked Up|s767|Stand-Up Comedy|
|4|Tom Papa: You're Doing Great!|s2953|Stand-Up Comedy|
|5|D.L. Hughley: Clear|s5379|Stand-Up Comedy|
|6|Zach Galifianakis: Live at the Purple Onion|s4081|Stand-Up Comedy|
|7|Oh, Hello On Broadway|s5433|Stand-Up Comedy|
|8|Chris D'Elia: Man on Fire|s5417|Stand-Up Comedy|
|9|Tom Segura: Completely Normal|s5380|Stand-Up Comedy|

Pada Tabel 11. di atas, terlihat bahwa untuk 10 hasil rekomendasi dengan *Cosine Similarity* mempunyai genre yang sama dengan sampel yaitu `Stand-Up Comedy` sehingga top-10 rekomendasi di atas adalah relevan dengan sampel.

#### Rekomendasi dengan Euclidean Similarity

Tabel 12. Top-10 Rekomendasi dengan Euclidean Similarity

|index|show_id|title|listed_in|
|---|---|---|---|
|0|Bo Burnham: what.|s4792|Stand-Up Comedy|
|1|Joe Mande’s Award-Winning Comedy Special|s5370|Stand-Up Comedy|
|2|Aditi Mittal: Things They Wouldn't Let Me Say|s5372|Stand-Up Comedy|
|3|Alan Saldaña: Locked Up|s767|Stand-Up Comedy|
|4|Tom Papa: You're Doing Great!|s2953|Stand-Up Comedy|
|5|D.L. Hughley: Clear|s5379|Stand-Up Comedy|
|6|Zach Galifianakis: Live at the Purple Onion|s4081|Stand-Up Comedy|
|7|Oh, Hello On Broadway|s5433|Stand-Up Comedy|
|8|Chris D'Elia: Man on Fire|s5417|Stand-Up Comedy|
|9|Tom Segura: Completely Normal|s5380|Stand-Up Comedy|

Pada Tabel 12. di atas, terlihat bahwa untuk 10 hasil rekomendasi dengan *Euclidean Similarity* mempunyai genre yang sama dengan sampel yaitu `Stand-Up Comedy` sehingga top-10 rekomendasi di atas adalah relevan dengan sampel.

## Evaluation

$$\text{Recommender system precision (P)} = \frac{\text{\#of our recommendation that relevant}}{\text{\#of item we recommend}}\times 100% $$

Dari hasil rekomendasi di atas diketahui bahwa `Bo Burnham: Inside` termasuk ke dalam genre atau aliran `Stand-Up Comedy`. Dari 10 judul film yang direkomendasikan, berikut nilai *precision* pada model *cosine similarity* dan *euclidean distance*.

Tabel 13. Komparasi Metrik *Precision*

|Model | Sesuai | Tidak Sesuai |Total| Precision |
|---|---|---|---|---|
|_Cosine Similarity_|10|0|10|100%|
|_Euclidean Similarity_|10|0|10|100%|
 
Dapat dilihat pada tabel di atas bahwa model *Cosine Similiarity* dan *Euclidean Distance* memiliki nilai *precision* yang sama pada top-10 rekomendasi di atas.

Selain dari nilai *precision*, lama komputasi dari setiap metode perlu dipertimbangkan juga. Berikut perbandingan waktu komputasi antara *Cosine Similiarity* dan *Euclidean Distance* :

Tabel 14. Perbandingan Waktu Komputasi

||Cosine Similarity|Euclidean Similarity|
|---|---|---|
|Time (Seconds)|1.004601|2.640574|

Berdasarkan output di atas, waktu komputasi pada metode Cosine Similarity (1.004601 detik) lebih cepat dibandingkan Euclidean Similarity (2.640574 detik).

Dari hasil di atas dapat disimpulkan bahwa model terbaik untuk sistem rekomendasi film berdasarkan genre adalah model dengan metode *Cosine Similarity*.