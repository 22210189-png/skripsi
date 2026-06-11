import streamlit as st
import pandas as pd
from pathlib import Path

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Dashboard Text Mining Pemilu 2024",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# PATH FOLDER
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VIS_DIR = BASE_DIR / "visualisasi"

# =========================================================
# FUNGSI BANTU
# =========================================================
@st.cache_data
def load_csv(file_name):
    file_path = DATA_DIR / file_name

    if not file_path.exists():
        return pd.DataFrame()

    encodings_to_try = ["utf-8", "utf-8-sig", "latin1"]

    for enc in encodings_to_try:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except Exception:
            continue

    return pd.DataFrame()


def get_total_rows(df):
    return len(df) if not df.empty else 0


def get_total_columns(df):
    return len(df.columns) if not df.empty else 0


def file_status_label(df):
    return "✅ Terdeteksi" if not df.empty else "⚠️ Tidak terbaca / kosong"


# =========================================================
# LOAD DATA
# =========================================================
# dataset_df = load_csv("dataset_logistik_pemilu_dummy.csv")
dataset_df = load_csv("dataset_text_mining_fix.csv")
clean_df = load_csv("clean.csv")
data_bersih_df = load_csv("data_bersih.csv")
normalisasi_df = load_csv("Normalisasi.csv")
hasil_topik_df = load_csv("hasil_topik_per_dokumen.csv")
keyword_df = load_csv("Keyword_logistik_pemilu_2024.csv")

# =========================================================
# SIDEBAR
# =========================================================

# PATH LOGO
ASSETS_DIR = BASE_DIR / "assets"
logo_path = ASSETS_DIR / "logoooput.png"

# LOGO
if logo_path.exists():
    st.sidebar.image(str(logo_path), use_container_width=True)

# TITLE
st.sidebar.markdown("## 📊 Text Mining App")
st.sidebar.markdown("**Analisis Logistik Pemilu 2024**")

st.sidebar.info(
    "Gunakan menu di sidebar untuk membuka halaman Dataset, "
    "Preprocessing, Topic Modeling, dan Visualisasi."
)

st.sidebar.markdown("---")

# STATUS FILE
st.sidebar.subheader("📁 Status File")

files_dict = {
    # "dataset_logistik_pemilu_dummy.csv": dataset_df,
    "dataset_text_mining_fix.csv": dataset_df,
    "clean.csv": clean_df,
    "data_bersih.csv": data_bersih_df,
    "Normalisasi.csv": normalisasi_df,
    "hasil_topik_per_dokumen.csv": hasil_topik_df,
    "Keyword_logistik_pemilu_2024.csv": keyword_df,
}

for file_name, df in files_dict.items():
    st.sidebar.write(f"**{file_name}**")
    st.sidebar.caption(file_status_label(df))

st.sidebar.markdown("---")

# INFO METODE
st.sidebar.markdown(
    """
### 🧠 Metode
- Text Mining  
- NLP  
- LDA  

### 🛠️ Tools
- Python  
- Streamlit  
- Pandas  
"""
)

st.sidebar.markdown("---")

# FOOTER SIDEBAR
st.sidebar.caption("© 2025 Text Mining Dashboard")

# =========================================================
# HEADER UTAMA
# =========================================================
st.title("🗳️ Dashboard Text Mining Logistik Pemilu 2024")
st.markdown(
    """
Dashboard ini menampilkan hasil penelitian **Identifikasi Isu Logistik Pemilu 2024 di Indonesia**
menggunakan pendekatan **Text Mining** dan **Topic Modeling Latent Dirichlet Allocation (LDA)**.
"""
)

# =========================================================
# METRICS
# =========================================================
st.markdown("## 📌 Ringkasan Data")

col1, col2, col3, col4 = st.columns(4)

jumlah_dokumen = get_total_rows(dataset_df)
# jumlah_topik = 5
jumlah_topik = 8
jumlah_hasil_topik = get_total_rows(hasil_topik_df)
jumlah_keyword = get_total_rows(keyword_df)

col1.metric("Jumlah Dokumen", jumlah_dokumen)
col2.metric("Jumlah Topik", jumlah_topik)
col3.metric("Data Hasil Topik", jumlah_hasil_topik)
col4.metric("Jumlah Keyword", jumlah_keyword)

st.markdown("---")

# =========================================================
# RINGKASAN PENELITIAN
# =========================================================
left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("📖 Ringkasan Penelitian")
    st.write(
        """
Penelitian ini bertujuan untuk mengidentifikasi isu-isu logistik Pemilu 2024 di Indonesia
berdasarkan data berita online menggunakan teknik **text mining**.

Tahapan utama analisis:
1. Pengumpulan data berita
2. Preprocessing teks
3. Pembentukan model LDA
4. Visualisasi hasil topic modeling
"""
    )

with right_col:
    st.subheader("🎯 Fokus Penelitian")
    st.success(
        """
- Isu logistik pemilu  
- Distribusi logistik  
- Peran KPU dan Bawaslu  
- Surat suara dan kerusakan logistik  
- Topic modeling berbasis LDA
"""
    )

st.markdown("---")

# =========================================================
# TAHAPAN ANALISIS
# =========================================================
st.subheader("⚙️ Tahapan Analisis")

step1, step2, step3, step4 = st.columns(4)

with step1:
    st.info("**1. Data Collection**\n\nPengumpulan data berita terkait logistik Pemilu 2024.")

with step2:
    st.info("**2. Preprocessing**\n\nCleaning, tokenizing, stopwords removal, stemming, punctuation removal.")

with step3:
    st.info("**3. LDA Modeling**\n\nPembentukan topik laten dari kumpulan dokumen.")

with step4:
    st.info("**4. Visualization**\n\nInterpretasi topik dan visualisasi hasil model.")

st.markdown("---")

# =========================================================
# INTERPRETASI TOPIK
# =========================================================
# st.subheader("🧠 Interpretasi 5 Topik Utama")
st.subheader("🧠 Interpretasi 8 Topik Utama")

topic_col1, topic_col2 = st.columns(2)

with topic_col1:
    st.info(
        """
**Topik 1**  
Distribusi logistik pemilu, pendistribusian, dan kelancaran logistik.
"""
    )

    st.info(
        """
**Topik 2**  
Peran KPU dan Bawaslu dalam distribusi dan pengawasan logistik pemilu.
"""
    )

    st.info(
        """
**Topik 3**  
Penyelenggaraan pemilu, surat suara, dan peran KPU di daerah.
"""
    )

with topic_col2:
    st.info(
        """
**Topik 4**  
Penyimpanan, pengelolaan, dan pengamanan logistik pemilu di kota/daerah.
"""
    )

    st.info(
        """
**Topik 5**  
Kondisi surat suara, kerusakan logistik, dan kesiapan logistik pemilu.
"""
    )

st.markdown("---")

# =========================================================
# PREVIEW DATA
# =========================================================
preview_col1, preview_col2 = st.columns(2)

with preview_col1:
    st.subheader("📂 Preview Dataset Utama")

    if not dataset_df.empty:
        st.caption(
            f"{get_total_rows(dataset_df)} baris • {get_total_columns(dataset_df)} kolom"
        )
        st.dataframe(dataset_df.head(10), use_container_width=True)
    else:
        st.warning("File dataset_logistik_pemilu_dummy.csv belum terbaca atau kosong.")

with preview_col2:
    st.subheader("📊 Preview Hasil Topic Modeling")

    if not hasil_topik_df.empty:
        st.caption(
            f"{get_total_rows(hasil_topik_df)} baris • {get_total_columns(hasil_topik_df)} kolom"
        )
        st.dataframe(hasil_topik_df.head(10), use_container_width=True)
    else:
        st.warning("File hasil_topik_per_dokumen.csv belum terbaca atau kosong.")

st.markdown("---")

# =========================================================
# STATUS FILE PREPROCESSING
# =========================================================
st.subheader("🧹 Status Data Preprocessing")

prep1, prep2, prep3 = st.columns(3)

with prep1:
    if not clean_df.empty:
        st.success("clean.csv siap digunakan")
    else:
        st.warning("clean.csv belum terbaca")

with prep2:
    if not data_bersih_df.empty:
        st.success("data_bersih.csv siap digunakan")
    else:
        st.warning("data_bersih.csv belum terbaca")

with prep3:
    if not normalisasi_df.empty:
        st.success("Normalisasi.csv siap digunakan")
    else:
        st.warning("Normalisasi.csv belum terbaca")

st.markdown("---")

# =========================================================
# STATUS VISUALISASI HTML
# =========================================================
st.subheader("🌐 Status Visualisasi LDA")

html_files = list(VIS_DIR.glob("*.html"))

if html_files:
    st.success(f"File visualisasi ditemukan: {html_files[0].name}")
    st.caption("Visualisasi interaktif dapat dibuka pada halaman Visualisasi.")
else:
    st.warning("Belum ada file HTML visualisasi di folder 'visualisasi'.")

st.markdown("---")

# =========================================================
# FOOTER
# =========================================================
st.caption("Dashboard dibuat dengan Streamlit menggunakan data hasil text mining dan LDA.")