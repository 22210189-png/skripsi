import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Topic Modeling", page_icon="🧠", layout="wide")

# =========================================================
# PATH
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_csv(file_name):
    file_path = DATA_DIR / file_name

    if not file_path.exists():
        return pd.DataFrame()

    encodings = ["utf-8", "utf-8-sig", "latin1"]
    for enc in encodings:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except Exception:
            continue

    return pd.DataFrame()


hasil_topik_df = load_csv("hasil_topik_per_dokumen.csv")
keyword_df = load_csv("Keyword_logistik_pemilu_2024.csv")

# =========================================================
# FUNGSI BANTU
# =========================================================
def get_topic_column(df):
    if df.empty:
        return None

    possible_topic_cols = [
        col for col in df.columns
        if "topik" in col.lower() or "topic" in col.lower()
    ]

    if possible_topic_cols:
        return possible_topic_cols[0]

    return None


def show_basic_info(df, nama_file):
    col1, col2, col3 = st.columns(3)
    col1.metric("Jumlah Baris", df.shape[0] if not df.empty else 0)
    col2.metric("Jumlah Kolom", df.shape[1] if not df.empty else 0)
    col3.metric("Nama File", nama_file)


def search_dataframe(df, label):
    if df.empty:
        return

    keyword = st.text_input(f"Cari data pada {label}")
    if keyword:
        filtered_df = df[
            df.astype(str).apply(
                lambda row: row.str.contains(keyword, case=False, na=False).any(),
                axis=1
            )
        ]
        st.write(f"Hasil pencarian untuk: **{keyword}**")
        st.dataframe(filtered_df, use_container_width=True)


# =========================================================
# HEADER
# =========================================================
st.title("🧠 Topic Modeling LDA")
st.markdown("Halaman ini menampilkan hasil pemodelan topik menggunakan metode **Latent Dirichlet Allocation (LDA)**.")

# =========================================================
# METRICS UTAMA
# =========================================================
topic_col_name = get_topic_column(hasil_topik_df)

jumlah_dokumen = hasil_topik_df.shape[0] if not hasil_topik_df.empty else 0
jumlah_kolom_hasil = hasil_topik_df.shape[1] if not hasil_topik_df.empty else 0
jumlah_keyword = keyword_df.shape[0] if not keyword_df.empty else 0
jumlah_topik = 8

m1, m2, m3, m4 = st.columns(4)
m1.metric("Jumlah Topik", jumlah_topik)
m2.metric("Dokumen Bertopik", jumlah_dokumen)
m3.metric("Kolom Hasil Topik", jumlah_kolom_hasil)
m4.metric("Jumlah Keyword", jumlah_keyword)

st.markdown("---")

# =========================================================
# INTERPRETASI TOPIK
# =========================================================
# st.subheader("📌 Interpretasi 5 Topik Utama")
st.subheader("📌 Interpretasi 8 Topik Utama")

col_a, col_b = st.columns(2)

with col_a:
    st.info("**Topik 1**\n\nDistribusi logistik pemilu, pendistribusian, dan kelancaran logistik.")
    st.info("**Topik 2**\n\nPeran KPU dan Bawaslu dalam distribusi dan pengawasan logistik pemilu.")
    st.info("**Topik 3**\n\nPenyelenggaraan pemilu, surat suara, dan peran KPU di daerah.")

with col_b:
    st.info("**Topik 4**\n\nPenyimpanan, pengelolaan, dan pengamanan logistik pemilu di kota/daerah.")
    st.info("**Topik 5**\n\nKondisi surat suara, kerusakan logistik, dan kesiapan logistik pemilu.")

st.success(
    """
**Insight utama:**  
Topik-topik yang terbentuk menunjukkan bahwa isu logistik Pemilu 2024 banyak berkaitan dengan
distribusi logistik, koordinasi KPU dan Bawaslu, pengelolaan surat suara, penyimpanan logistik,
serta potensi kerusakan pada perlengkapan pemilu.
"""
)

st.markdown("---")

# =========================================================
# DATA HASIL TOPIK
# =========================================================
st.subheader("📊 Data Hasil Topik per Dokumen")

if not hasil_topik_df.empty:
    show_basic_info(hasil_topik_df, "hasil_topik_per_dokumen.csv")

    preview_limit = st.slider(
        "Jumlah baris ditampilkan",
        min_value=5,
        max_value=min(100, len(hasil_topik_df)),
        value=min(10, len(hasil_topik_df)),
        key="hasil_topik_slider"
    )

    st.dataframe(hasil_topik_df.head(preview_limit), use_container_width=True)

    st.markdown("### 🔍 Pencarian Data Hasil Topik")
    search_dataframe(hasil_topik_df, "hasil_topik_per_dokumen.csv")

    csv_hasil = hasil_topik_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download hasil_topik_per_dokumen.csv",
        data=csv_hasil,
        file_name="hasil_topik_per_dokumen.csv",
        mime="text/csv"
    )
else:
    st.warning("File hasil_topik_per_dokumen.csv tidak ditemukan atau kosong.")

st.markdown("---")

# =========================================================
# VISUALISASI DISTRIBUSI TOPIK
# =========================================================
st.subheader("📈 Visualisasi Distribusi Dokumen per Topik")

if not hasil_topik_df.empty and topic_col_name is not None:
    count_df = hasil_topik_df[topic_col_name].value_counts().reset_index()
    count_df.columns = ["Topik", "Jumlah Dokumen"]

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig_bar = px.bar(
            count_df,
            x="Topik",
            y="Jumlah Dokumen",
            title="Distribusi Dokumen per Topik",
            text="Jumlah Dokumen"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with chart_col2:
        fig_pie = px.pie(
            count_df,
            names="Topik",
            values="Jumlah Dokumen",
            title="Proporsi Dokumen per Topik"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.dataframe(count_df, use_container_width=True)
elif not hasil_topik_df.empty and topic_col_name is None:
    st.info("Kolom topik tidak ditemukan otomatis pada hasil_topik_per_dokumen.csv.")
else:
    st.warning("Data hasil topik belum tersedia.")

st.markdown("---")

# =========================================================
# DATA KEYWORD
# =========================================================
st.subheader("🔑 Data Keyword")

if not keyword_df.empty:
    show_basic_info(keyword_df, "Keyword_logistik_pemilu_2024.csv")

    preview_keyword = st.slider(
        "Jumlah keyword ditampilkan",
        min_value=5,
        max_value=min(100, len(keyword_df)),
        value=min(10, len(keyword_df)),
        key="keyword_slider"
    )

    st.dataframe(keyword_df.head(preview_keyword), use_container_width=True)

    st.markdown("### 🔍 Pencarian Keyword")
    search_dataframe(keyword_df, "Keyword_logistik_pemilu_2024.csv")

    st.markdown("### 📊 Visualisasi Keyword")

    # Coba deteksi kolom keyword dan frekuensi
    keyword_col = None
    freq_col = None

    for col in keyword_df.columns:
        low = col.lower()
        if keyword_col is None and ("keyword" in low or "kata" in low or "term" in low):
            keyword_col = col
        if freq_col is None and ("freq" in low or "jumlah" in low or "count" in low or "bobot" in low):
            freq_col = col

    if keyword_col and freq_col:
        chart_df = keyword_df[[keyword_col, freq_col]].dropna().head(10)

        fig_keyword = px.bar(
            chart_df,
            x=freq_col,
            y=keyword_col,
            orientation="h",
            title="Top Keyword"
        )
        st.plotly_chart(fig_keyword, use_container_width=True)
    else:
        st.info("Kolom keyword/frekuensi belum terdeteksi otomatis untuk divisualisasikan.")

    csv_keyword = keyword_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Keyword_logistik_pemilu_2024.csv",
        data=csv_keyword,
        file_name="Keyword_logistik_pemilu_2024.csv",
        mime="text/csv"
    )
else:
    st.warning("File Keyword_logistik_pemilu_2024.csv tidak ditemukan atau kosong.")

st.markdown("---")
st.caption("Halaman ini menampilkan hasil topic modeling LDA beserta distribusi topik dan keyword pendukung.")