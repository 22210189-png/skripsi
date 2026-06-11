import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Dataset", page_icon="📂", layout="wide")

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
        except:
            continue

    return pd.DataFrame()

dataset_df = load_csv("dataset_logistik_pemilu_dummy.csv")

# =========================================================
# HEADER
# =========================================================
st.title("📂 Dataset Logistik Pemilu 2024")
st.markdown("Halaman ini menampilkan dataset utama beserta analisis awal.")

# =========================================================
# CEK DATA
# =========================================================
if dataset_df.empty:
    # st.warning("File dataset_logistik_pemilu_dummy.csv tidak ditemukan atau kosong.")
    st.warning("File dataset_text_mining_fix.csv tidak ditemukan atau kosong.")
    st.stop()

# =========================================================
# METRICS
# =========================================================
col1, col2, col3 = st.columns(3)

col1.metric("Jumlah Baris", dataset_df.shape[0])
col2.metric("Jumlah Kolom", dataset_df.shape[1])
col3.metric("Missing Value", dataset_df.isnull().sum().sum())

st.markdown("---")

# =========================================================
# PREVIEW DATA
# =========================================================
st.subheader("📊 Preview Dataset")

limit = st.slider("Jumlah data ditampilkan", 5, 100, 10)
st.dataframe(dataset_df.head(limit), use_container_width=True)

st.markdown("---")

# =========================================================
# INFO KOLOM
# =========================================================
st.subheader("📋 Informasi Kolom")

info_df = pd.DataFrame({
    "Nama Kolom": dataset_df.columns,
    "Tipe Data": [str(dataset_df[col].dtype) for col in dataset_df.columns],
    "Non-Null": [dataset_df[col].notnull().sum() for col in dataset_df.columns],
    "Missing": [dataset_df[col].isnull().sum() for col in dataset_df.columns]
})

st.dataframe(info_df, use_container_width=True)

st.markdown("---")

# =========================================================
# VISUALISASI DATA
# =========================================================
st.subheader("📈 Visualisasi Data")

numeric_cols = dataset_df.select_dtypes(include='number').columns.tolist()

if numeric_cols:
    selected_col = st.selectbox("Pilih kolom numerik", numeric_cols)

    colA, colB = st.columns(2)

    with colA:
        fig = px.histogram(
            dataset_df,
            x=selected_col,
            title=f"Distribusi {selected_col}"
        )
        st.plotly_chart(fig, use_container_width=True)

    with colB:
        fig2 = px.box(
            dataset_df,
            y=selected_col,
            title=f"Boxplot {selected_col}"
        )
        st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("Tidak ada kolom numerik untuk divisualisasikan.")

st.markdown("---")

# =========================================================
# FREKUENSI (KATEGORIK)
# =========================================================
st.subheader("📊 Distribusi Data Kategorikal")

cat_cols = dataset_df.select_dtypes(include='object').columns.tolist()

if cat_cols:
    selected_cat = st.selectbox("Pilih kolom kategori", cat_cols)

    value_counts = dataset_df[selected_cat].value_counts().head(10)

    fig3 = px.bar(
        x=value_counts.values,
        y=value_counts.index,
        orientation='h',
        title=f"Top 10 {selected_cat}"
    )
    st.plotly_chart(fig3, use_container_width=True)

else:
    st.info("Tidak ada kolom kategorikal.")

st.markdown("---")

# =========================================================
# FILTER DATA
# =========================================================
st.subheader("🔎 Filter Data")

selected_column = st.selectbox("Pilih kolom untuk filter", dataset_df.columns)

unique_values = dataset_df[selected_column].dropna().unique()

if len(unique_values) < 50:
    selected_value = st.selectbox("Pilih nilai", unique_values)

    filtered_df = dataset_df[dataset_df[selected_column] == selected_value]

    st.write(f"Hasil filter: **{selected_value}**")
    st.dataframe(filtered_df, use_container_width=True)
else:
    st.info("Kolom memiliki terlalu banyak nilai unik untuk filter dropdown.")

st.markdown("---")

# =========================================================
# SEARCH DATA
# =========================================================
st.subheader("🔍 Pencarian Data")

keyword = st.text_input("Masukkan kata kunci")

if keyword:
    filtered_df = dataset_df[
        dataset_df.astype(str).apply(
            lambda row: row.str.contains(keyword, case=False, na=False).any(),
            axis=1
        )
    ]

    st.write(f"Hasil pencarian untuk: **{keyword}**")
    st.dataframe(filtered_df, use_container_width=True)

# =========================================================
# DOWNLOAD DATA
# =========================================================
st.subheader("⬇️ Download Dataset")

csv = dataset_df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="Download CSV",
    data=csv,
    file_name="dataset_logistik_pemilu.csv",
    mime="text/csv"
)