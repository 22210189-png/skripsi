import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Preprocessing", page_icon="🧹", layout="wide")

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


clean_df = load_csv("clean.csv")
data_bersih_df = load_csv("data_bersih.csv")
normalisasi_df = load_csv("Normalisasi.csv")

# =========================================================
# FUNGSI BANTU
# =========================================================
def show_basic_info(df, nama_file):
    col1, col2, col3 = st.columns(3)
    col1.metric("Jumlah Baris", df.shape[0] if not df.empty else 0)
    col2.metric("Jumlah Kolom", df.shape[1] if not df.empty else 0)
    col3.metric("Nama File", nama_file)


def show_column_info(df):
    if df.empty:
        st.warning("Data kosong.")
        return

    info_df = pd.DataFrame({
        "Nama Kolom": df.columns,
        "Tipe Data": [str(df[col].dtype) for col in df.columns],
        "Non-Null": [df[col].notnull().sum() for col in df.columns],
        "Missing": [df[col].isnull().sum() for col in df.columns]
    })

    st.dataframe(info_df, use_container_width=True)


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
st.title("🧹 Preprocessing Text")
st.markdown("Halaman ini menampilkan hasil tahapan preprocessing pada data teks.")

# =========================================================
# RINGKASAN
# =========================================================
st.subheader("📌 Ringkasan Tahapan Preprocessing")
st.info(
    """
Tahapan preprocessing yang digunakan dalam penelitian ini meliputi:
- Cleaning data
- Tokenizing
- Removing stopwords
- Stemming
- Removing punctuation
- Normalisasi
"""
)

summary_col1, summary_col2, summary_col3 = st.columns(3)

summary_col1.metric("clean.csv", clean_df.shape[0] if not clean_df.empty else 0)
summary_col2.metric("data_bersih.csv", data_bersih_df.shape[0] if not data_bersih_df.empty else 0)
summary_col3.metric("Normalisasi.csv", normalisasi_df.shape[0] if not normalisasi_df.empty else 0)

st.markdown("---")

# =========================================================
# TAB PREPROCESSING
# =========================================================
tab1, tab2, tab3 = st.tabs(["Cleaning", "Data Bersih", "Normalisasi"])

# =========================================================
# TAB 1 - CLEANING
# =========================================================
with tab1:
    st.subheader("🧼 Hasil Cleaning")

    if not clean_df.empty:
        show_basic_info(clean_df, "clean.csv")

        limit_clean = st.slider(
            "Jumlah baris ditampilkan (Cleaning)",
            min_value=5,
            max_value=min(100, len(clean_df)),
            value=min(10, len(clean_df)),
            key="clean_slider"
        )
        st.dataframe(clean_df.head(limit_clean), use_container_width=True)

        st.markdown("### Informasi Kolom")
        show_column_info(clean_df)

        st.markdown("### Pencarian Data")
        search_dataframe(clean_df, "clean.csv")

        csv_clean = clean_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download clean.csv",
            data=csv_clean,
            file_name="clean.csv",
            mime="text/csv"
        )
    else:
        st.warning("File clean.csv tidak ditemukan atau kosong.")

# =========================================================
# TAB 2 - DATA BERSIH
# =========================================================
with tab2:
    st.subheader("🧽 Hasil Data Bersih")

    if not data_bersih_df.empty:
        show_basic_info(data_bersih_df, "data_bersih.csv")

        limit_bersih = st.slider(
            "Jumlah baris ditampilkan (Data Bersih)",
            min_value=5,
            max_value=min(100, len(data_bersih_df)),
            value=min(10, len(data_bersih_df)),
            key="bersih_slider"
        )
        st.dataframe(data_bersih_df.head(limit_bersih), use_container_width=True)

        st.markdown("### Informasi Kolom")
        show_column_info(data_bersih_df)

        st.markdown("### Pencarian Data")
        search_dataframe(data_bersih_df, "data_bersih.csv")

        csv_bersih = data_bersih_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download data_bersih.csv",
            data=csv_bersih,
            file_name="data_bersih.csv",
            mime="text/csv"
        )
    else:
        st.warning("File data_bersih.csv tidak ditemukan atau kosong.")

# =========================================================
# TAB 3 - NORMALISASI
# =========================================================
with tab3:
    st.subheader("🔤 Hasil Normalisasi")

    if not normalisasi_df.empty:
        show_basic_info(normalisasi_df, "Normalisasi.csv")

        limit_norm = st.slider(
            "Jumlah baris ditampilkan (Normalisasi)",
            min_value=5,
            max_value=min(100, len(normalisasi_df)),
            value=min(10, len(normalisasi_df)),
            key="normalisasi_slider"
        )
        st.dataframe(normalisasi_df.head(limit_norm), use_container_width=True)

        st.markdown("### Informasi Kolom")
        show_column_info(normalisasi_df)

        st.markdown("### Pencarian Data")
        search_dataframe(normalisasi_df, "Normalisasi.csv")

        csv_norm = normalisasi_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Normalisasi.csv",
            data=csv_norm,
            file_name="Normalisasi.csv",
            mime="text/csv"
        )
    else:
        st.warning("File Normalisasi.csv tidak ditemukan atau kosong.")

st.markdown("---")

# =========================================================
# PERBANDINGAN FILE
# =========================================================
st.subheader("📊 Perbandingan Dataset Preprocessing")

comparison_df = pd.DataFrame({
    "File": ["clean.csv", "data_bersih.csv", "Normalisasi.csv"],
    "Jumlah Baris": [
        clean_df.shape[0] if not clean_df.empty else 0,
        data_bersih_df.shape[0] if not data_bersih_df.empty else 0,
        normalisasi_df.shape[0] if not normalisasi_df.empty else 0,
    ],
    "Jumlah Kolom": [
        clean_df.shape[1] if not clean_df.empty else 0,
        data_bersih_df.shape[1] if not data_bersih_df.empty else 0,
        normalisasi_df.shape[1] if not normalisasi_df.empty else 0,
    ]
})

st.dataframe(comparison_df, use_container_width=True)

st.caption("Halaman ini menampilkan hasil preprocessing yang digunakan sebelum proses topic modeling LDA.")