# Laporan Proyek Machine Learning

### Nama : Ahmad Fauzan Nabil

### Nim : 211351002

### Kelas : Malam B

## Domain Proyek

Web app ini bertujuan untuk mendalami dan menganalisis hubungan yang kompleks antara berbagai faktor yang berpengaruh terhadap tingkat kebahagiaan suatu negara. Faktor-faktor yang menjadi fokus utama dalam web app ini meliputi tingkat kemakmuran ekonomi, kekuatan hubungan sosial dalam masyarakat, serta persepsi masyarakat terhadap tingkat kepercayaan kepada pemerintah, yang sering diindikasikan oleh tingkat korupsi yang dirasakan.

## Business Understanding

Memahami tentang faktor apa saja yang dapat mempengaruhi kebahagiaan suatu negara, Dengan pemahaman ini memungkin juga memberikan rekomendasi kebijakan atau arahan untuk perbaikan yang dapat membantu pemerintah atau lembaga lain dalam meningkatkan skor kebahagiaan negara.

### Problem Statements

-   Bagaimana cara mengukur tingkat kebahagiaan suatu negara?
-   Bagaimana cara mengetahui faktor yang sangat berpengaruh pada kebahagiaan suatu negara?

### Solution statements

-   Kita dapat mengukur kebahagiaan suatu negara berdasarkan faktor-faktor seperti ekonomi, hubungan sosial, dan persepsi terhadap pemerintah.
-   Kita dapat mengetahuinya dengan mencari korelasi skor kebahagiaan dengan faktor-faktornya.

### Goals

Memberikan pemahaman tentang faktor apa saja yang dapat mempengaruhi kebahagiaan suatu negara.

## Data Understanding

Dataset ini adalah data dari Gallup World Poll. Ini adalah survei global yang dilakukan oleh lembaga Gallup untuk mengukur tingkat kebahagiaan di berbagai negara. Data yang digunakan dalam laporan ini berasal dari survei ini, yang diadakan setiap tahun. Dataset ini berisikan 156 baris dan 9 kolom

-   Link dataset [World Happiness Report](https://www.kaggle.com/datasets/unsdsn/world-happiness/)

### Variabel-variabel pada Dataset World Happiness Report adalah sebagai berikut:

-   **Overall rank** : Peringkat dari skor kebahagian suatu negara.
-   **Country or region** : Nama negara atau wilayah.
-   **GDP per capita** : PDB per kapita adalah ukuran produksi ekonomi suatu negara yang memperhitungkan jumlah penduduknya.
-   **Social support** : Dukungan sosial berarti memiliki teman dan orang lain, termasuk keluarga, untuk diandalkan saat membutuhkan atau dalam situasi krisis, memberikan fokus yang lebih luas dan citra diri yang positif.
-   **Healthy life expectancy** : Tingkat harapan hidup yang sehat.
-   **Freedom to make life choices** : Kebebasan untuk memilih menggambarkan kesempatan dan otonomi seseorang untuk melakukan tindakan yang dipilih dari setidaknya dua pilihan yang tersedia, tanpa dibatasi oleh pihak eksternal.
-   **Generosity** : Kualitas dari sikap baik dan murah hati.
-   **Perceptions of corruption** : Ukuran seberapa banyak percayaan masyarakat terhadap tingkat kejujuran dan keadilan dalam pemerintahan dan bisnis di negara mereka.

Semua variabel atau faktor yang akan digunakan adalah berupa angka bertipe data `float64`, angka tersebut mempresentasikan skor dari masing-masing faktornya

## Data Preparation

### Import Dataset

Pertama kita upload dulu file kaggle.json yang kita dapatkan dari website kaggle dan membuat folder untuk menyimpan filenya

```py
from google.colab import files
files.upload()
```

```py
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle
```

Selanjutnya mendownload datasetnya

```py
!kaggle datasets download -d unsdsn/world-happiness
```

Ekstrak hasil download tadi yaitu `world-happines.zip`

```python
!unzip world-happiness.zip -d world-happiness
!ls world-happiness
```

### Import library yang akan digunakan

```py
import pandas as pd
import numpy as np
from math import pi

import matplotlib.pyplot as plt
import seaborn as sns
import pylab
import plotly.express as px

from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
```

### Data discovery

Kita buat variable yang bernama df untuk menyimpan dataframe dari dataset

```
df = pd.read_csv('/content/world-happiness/2019.csv')
```

Lalu kita Lihat dimensi dataframe, dan lihat 5 data teratas

```python
print("Dimensi dataframe", df.shape)
df.head()
```

Dimensi dataframe (156, 9)

Kita lihat informasi dari data kita, disini kita memiliki datatype object tapi tidak apa karena tidak akan kita digunakan pada proses modeling

```py
df.info()
```

Disini kita lihat ringkasan statistikk tiap data seperti jumlah, rata-rata, min dan max

```py
df.describe()
```

Pada data ini kita tidak menemukan data null dan itu bagus

```py
df.isnull().sum()
```

Tidak ada data duplikat semua data unik pada dataset ini

```python
df.duplicated().sum()
```

### Visulaisasi

Heatmap antar kolom, kita lihat GDP per capita, social support dan healt life expceptancy memiliki korelasi paling tinggi pada score happines dari sebuah negara

```py
sns.heatmap(df.corr(), annot=True)
```

![heatmap](https://github.com/fauzann221/world-happy-clustering/assets/149223860/a75d05f4-32d2-44b5-b86d-72fafc34f155)


Menampilkan 10 negara dengan rasio bahagia tertinggi, bisa kita lihat Finland pada urutan pertama lalu diikuti oleh denmark

```py
region_happiness = df.groupby('Country or region')['Score'].mean().sort_values(ascending=False)

top_10_regions = region_happiness.head(10)

plt.figure(figsize=(12,10))
sns.barplot(x=top_10_regions.index, y=top_10_regions.values, palette=sns.cubehelix_palette(10))
plt.xticks(rotation=90)
plt.xlabel('Country or Region')
plt.ylabel('Region Happiness Ratio')
plt.title('Top 10 Countries with Highest Happiness Ratio')
plt.show()

```

![10happines](https://github.com/fauzann221/world-happy-clustering/assets/149223860/f71da5ed-450f-4117-90a2-52a3c8a9719e)


Menampilkan 10 negara dengan GDP percapita tertinggi, Qatar adalah negara dengan GDP tertinggi

```py
highest_gdp = df.groupby('Country or region')['GDP per capita'].mean().sort_values(ascending=False)

top_10_regions = highest_gdp.head(10)

plt.figure(figsize=(12,10))
sns.barplot(x=top_10_regions.index, y=top_10_regions.values, palette=sns.cubehelix_palette(10))
plt.xticks(rotation=90)
plt.xlabel('Country or Region')
plt.ylabel('Region Highest GDP per capita')
plt.title('Top 10 Countries with Highest GDP')
plt.show()
```

![10gdp](https://github.com/fauzann221/world-happy-clustering/assets/149223860/940e5377-3071-4d55-9c3f-6d276466aedc)


Hubungan GDP pre capita dengan Happiness Score, semakin GDP per capita tinggi semakin tinggi kebahagiaan sebuah negara

```py
dataframe2 = pd.pivot_table(df, index='Country or region', values=["Score", "GDP per capita"])

dataframe2["Score"] = dataframe2["Score"] / max(dataframe2["Score"])
dataframe2["GDP per capita"] = dataframe2["GDP per capita"] / max(dataframe2["GDP per capita"])

# Menggunakan regplot untuk menggambarkan hubungan antara "GDP per capita" dan "Score"
sns.lmplot(x="GDP per capita", y="Score", data=dataframe2, fit_reg=False)
plt.xlabel('GDP per capita (Normalized)')
plt.ylabel('Happiness Score (Normalized)')
plt.title('Relationship between GDP per capita and Happiness Score')
plt.show()
```

![gdpvsscore](https://github.com/fauzann221/world-happy-clustering/assets/149223860/4d85b3fd-c151-4528-9d1c-5650f97099bf)

Persepsi korupsi atau kepercayaan masyarakat terhadap pemerintah memiliki efek negatif yang cukup signifikan

```py
dataframe2 = pd.pivot_table(df, index='Country or region', values=["Score", "Perceptions of corruption"])

dataframe2["Score"] = dataframe2["Score"] / max(dataframe2["Score"])
dataframe2["Perceptions of corruption"] = dataframe2["Perceptions of corruption"] / max(dataframe2["Perceptions of corruption"])

# Menggunakan regplot untuk menggambarkan hubungan antara "Perceptions of corruption" dan "Score"
sns.lmplot(x="Perceptions of corruption", y="Score", data=dataframe2, fit_reg=False)
plt.xlabel('Perceptions of corruption (Normalized)')
plt.ylabel('Happiness Score (Normalized)')
plt.title('Relationship between Perceptions of corruption and Happiness Score')
plt.show()
```

![corupvsscore](https://github.com/fauzann221/world-happy-clustering/assets/149223860/55d192e6-9bb0-495b-b076-496517507a7a)

### Preprocessing

Pertama kita tidak mau menggunakan semua kolom pada dataframe jadi kita akan membuat dataframe baru untuk modelingnya, lalu kita lihat isi dataframenya

```python
x = df.loc[:, ["Score", "GDP per capita", "Social support", "Perceptions of corruption"]]
x
```

## Modeling

Proses preprocessing selesai selanjutnya kita akan mencari elbow untuk dijadikan angka cluster, bisa kita lihat elbow yang memungkinkan adalah 4 dan 5

```py
k = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i).fit(x)
    k.append(kmeans.inertia_)

fig, ax = plt.subplots(figsize=(20,8))
sns.lineplot(x=list(range(1,11)), y=k, ax=ax, marker='^')
ax.set_title("Mencari elbow")
ax.set_xlabel("clusters")
ax.set_ylabel("inertia")

ax.annotate("Possible elbow point", xy=(4,40), xytext=(3.5, 150), xycoords="data",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="red", lw=2))

ax.annotate("Possible elbow point", xy=(5, 25), xytext=(6, 100), xycoords="data",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="red", lw=2))
```

![elbow](https://github.com/fauzann221/world-happy-clustering/assets/149223860/a0d8a46a-f21e-4eff-8525-b3e41841060d)

Bisa kita lihat possible elbownya pointnya berada dari 3 sampai 5 kita akan ambil 4 karena 4 mewakili keduanya

```py
n_clust = 4
kmean = KMeans(n_clusters=n_clust).fit(x)
x["Labels"] = kmean.labels_
```

## Visualiasis hasil algoritma

Model kita sudah di kelompokan menjadi 4 cluster, lalu kita lihat cluster berdasarkan GDP per capita dengan happiness score

```python
plt.figure (figsize=(10,8))
sns.scatterplot(x='GDP per capita', y='Score', hue='Labels', size='Labels', palette=sns.color_palette('hls', n_clust), data=x, markers=True)

for label in x['Labels']:
  plt.annotate (label,
  (x[x['Labels']==label]['GDP per capita'].mean(),
   x[x['Labels'] ==label]['Score'].mean()),
   horizontalalignment = 'center',
   verticalalignment = 'center',
   size = 20, weight='bold',
   color = 'black')
```

![clusgdp](https://github.com/fauzann221/world-happy-clustering/assets/149223860/a7744c0d-0687-42a2-9b15-8e38da492e30)


Lalu social support dengan happiness score

![clustsocial](https://github.com/fauzann221/world-happy-clustering/assets/149223860/706aecfd-a57f-4e62-9b4e-ac3cb9cf72a7)

Selanjutnya kita lihat nilai rata-rata dari setiap label

```py
km_mean = x.groupby(['Labels']).mean().round(2)
km_mean
```

```

            Score	GDP per capita	Social support Perceptions of corruption
Labels
0	         7.15	       1.36     	1.50    	0.24
1	         4.70	       0.67     	1.06    	0.09
2	         5.88	       1.09     	1.36    	0.08
3	         3.57	       0.36     	0.74    	0.11
```

## Deployment
