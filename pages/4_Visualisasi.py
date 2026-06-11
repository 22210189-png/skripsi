import streamlit as st
from pathlib import Path
import streamlit.components.v1 as components

st.set_page_config(page_title="Visualisasi", page_icon="🌐", layout="wide")

# =========================================================
# PATH
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent
VIS_DIR = BASE_DIR / "visualisasi"

# =========================================================
# FUNGSI BANTU
# =========================================================
def load_html_file(file_path):
    encodings = ["utf-8", "utf-8-sig", "latin1"]

    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                return f.read()
        except Exception:
            continue

    return None


# =========================================================
# HEADER
# =========================================================
st.title("🌐 Visualisasi Interaktif LDA")
st.markdown(
    """
Halaman ini menampilkan visualisasi interaktif hasil **topic modeling LDA**
dari file HTML yang dihasilkan, misalnya dari **pyLDAvis**.
"""
)

# =========================================================
# CEK FILE HTML
# =========================================================
html_files = sorted(list(VIS_DIR.glob("*.html")))

if not html_files:
    st.warning("Tidak ada file HTML di folder visualisasi.")
    st.stop()

# =========================================================
# INFO RINGKAS
# =========================================================
col1, col2 = st.columns(2)
col1.metric("Jumlah File HTML", len(html_files))
col2.metric("Folder Visualisasi", "visualisasi")

st.markdown("---")

# =========================================================
# PILIH FILE
# =========================================================
file_names = [file.name for file in html_files]

selected_file = st.selectbox(
    "Pilih file visualisasi HTML",
    file_names
)

selected_path = VIS_DIR / selected_file

# =========================================================
# DETAIL FILE
# =========================================================
file_size_kb = selected_path.stat().st_size / 1024

detail_col1, detail_col2, detail_col3 = st.columns(3)
detail_col1.metric("Nama File", selected_file)
detail_col2.metric("Ukuran File (KB)", f"{file_size_kb:.2f}")
detail_col3.metric("Status", "Siap ditampilkan")

st.markdown("---")

# =========================================================
# PENGATURAN TAMPILAN
# =========================================================
st.subheader("⚙️ Pengaturan Tampilan")

height_value = st.slider(
    "Tinggi visualisasi",
    min_value=400,
    max_value=1200,
    value=800,
    step=50
)

show_info = st.checkbox("Tampilkan informasi penggunaan", value=True)

if show_info:
    st.info(
        """
Visualisasi ini membantu menampilkan:
- jarak antar topik
- proporsi topik dalam korpus
- kata-kata yang paling relevan pada tiap topik

Jika tampilan terlalu kecil, naikkan tinggi visualisasi dengan slider di atas.
"""
    )

st.markdown("---")

# =========================================================
# TAMPILKAN HTML
# =========================================================
st.subheader("📊 Visualisasi LDA")

html_content = load_html_file(selected_path)

if html_content:
    try:
        components.html(html_content, height=height_value, scrolling=True)
        st.success(f"Visualisasi berhasil dimuat dari file: {selected_file}")
    except Exception as e:
        st.error(f"Gagal menampilkan file HTML: {e}")
else:
    st.error("File HTML tidak dapat dibaca. Periksa encoding atau isi file.")

st.markdown("---")

# =========================================================
# DAFTAR FILE YANG TERSEDIA
# =========================================================
st.subheader("📁 Daftar File Visualisasi")

file_table = []
for file in html_files:
    file_table.append({
        "Nama File": file.name,
        "Ukuran (KB)": round(file.stat().st_size / 1024, 2),
        "Lokasi": str(file)
    })

st.dataframe(file_table, use_container_width=True)

st.caption("Halaman ini digunakan untuk menampilkan file visualisasi interaktif hasil topic modeling LDA.")