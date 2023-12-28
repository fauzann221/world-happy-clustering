import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Preparation model
df = pd.read_csv("2019.csv")
x = df.loc[:, ["Score", "GDP per capita", "Social support", "Perceptions of corruption"]]

# Sidebar
st.sidebar.subheader("Nilai jumlah K")
clust = st.sidebar.slider("Tentukan Jumlah Cluster : ", 2,10,3,1)
st.sidebar.divider()
st.sidebar.write('''
Social support adalah dukungan sosial yang dirasakan oleh individu dalam masyarakat. Ini mencakup jaringan dukungan sosial dan perasaan keamanan dalam hubungan sosial.
''')

# Menampilkan elbow
st.write("### Mencari Elbow : ")
try:
    k = []
    for i in range(1,11):
        kmeans = KMeans(n_clusters=i).fit(x)
        k.append(kmeans.inertia_)
    fig, ax = plt.subplots(figsize=(10,8))
    sns.lineplot(x=list(range(1,11)), y=k, ax=ax, marker='^')
    ax.set_title("Mencari elbow")
    ax.set_xlabel("clusters")
    ax.set_ylabel("inertia")

    ax.annotate("Possible elbow point", xy=(3,50), xytext=(3, 150), xycoords="data",
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="red", lw=2))
    ax.annotate("Possible elbow point", xy=(5, 28), xytext=(5, 150), xycoords="data",
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="red", lw=2))
    st.set_option("deprecation.showPyplotGlobalUse", False)
    elbo_plot = st.pyplot()

except Exception as e:
    print(f"An exception occurred: {str(e)}")

def k_means(n_clust):
    kmean = KMeans(n_clusters=n_clust).fit(x)
    x["Labels"] = kmean.labels_
    plt.figure (figsize=(10,8))
    plt.title(f"Cluster hubungan antara Happiness Score dan Social Support")
    sns.scatterplot(x="Social support", y='Score', hue='Labels', size='Labels', palette=sns.color_palette('hls', n_clust), data=x, markers=True)
    for label in x['Labels']:
        plt.annotate (label,
            (x[x['Labels'] == label]["Social support"].mean(),
            x[x['Labels'] == label]['Score'].mean()),
            horizontalalignment = 'center',
            verticalalignment = 'center',
            size = 20, weight='bold',
            color = 'black')            
    st.write("### Cluster Plot Happiness Score & Social Support : ")
    plot = plt.gcf()  # Mengambil current figure
    st.pyplot(plot)   # Menampilkan plot di Streamlit
    return x

st.divider()
try:
    df_clust = k_means(clust)
except Exception as e:
    print(f"An exception occurred: {str(e)}")
st.divider()
dataframe2 = pd.pivot_table(df, index='Country or region', values=["Score", "Social support"])
dataframe2["Score"] = dataframe2["Score"] / max(dataframe2["Score"])
dataframe2["Social support"] = dataframe2["Social support"] / max(dataframe2["Social support"])

st.write("### Hubungan antara Social Support dengan happiness score")
# Menggunakan regplot untuk menggambarkan hubungan antara "Social support" dan "Score"
plt.figure(figsize=(10,8))
sns.lmplot(x="Social support", y="Score", data=dataframe2, fit_reg=False)
plt.xlabel('Social support (Normalized)')
plt.ylabel('Happiness Social support (Normalized)')
plt.title('Hubungan antara Social support and Happiness Score')
st.pyplot()

st.write('''
    Dari grafik diatas dapat kita simpulkan hubungan antara happiness score dengan social support terlihat lebih kuat daripada GDP per kapita. Hal ini menunjukkan bahwa hubungan sosial memainkan peran penting dalam kebahagiaan.
''')

st.divider()
st.write("### Data setelah clustering : ")
st.dataframe(df_clust, use_container_width=True)

st.divider()
x_mean = x.groupby(['Labels']).mean().round(2)
st.write("### Mean dari label dari masing-masing label")
st.dataframe(x_mean, use_container_width=True)