import re
import pandas as pd
from pathlib import Path
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# =========================================================
# PATH
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

INPUT_FILE = DATA_DIR / "dataset_text_mining_fix.csv"
CLEAN_FILE = DATA_DIR / "clean.csv"
DATA_BERSIH_FILE = DATA_DIR / "data_bersih.csv"
NORMALISASI_FILE = DATA_DIR / "Normalisasi.csv"

# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

# Ambil kolom teks utama
if "Text" in df.columns:
    text_col = "Text"
elif "konten" in df.columns:
    text_col = "konten"
else:
    raise ValueError("Kolom teks tidak ditemukan. Pastikan ada kolom 'Text' atau 'konten'.")

# =========================================================
# CLEANING
# =========================================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+|#\w+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean"] = df[text_col].apply(clean_text)

# =========================================================
# TOKENIZING
# =========================================================
df["Token"] = df["clean"].apply(lambda x: x.split())

# =========================================================
# NORMALISASI
# =========================================================
normalisasi_dict = {}

if NORMALISASI_FILE.exists():
    try:
        norm_df = pd.read_csv(NORMALISASI_FILE, encoding="utf-8-sig")

        if norm_df.shape[1] >= 2:
            kolom_awal = norm_df.columns[0]
            kolom_akhir = norm_df.columns[1]

            normalisasi_dict = dict(
                zip(
                    norm_df[kolom_awal].astype(str).str.lower(),
                    norm_df[kolom_akhir].astype(str).str.lower()
                )
            )
    except Exception:
        normalisasi_dict = {}

def normalisasi(tokens):
    return [normalisasi_dict.get(token, token) for token in tokens]

df["Normalisasi"] = df["Token"].apply(normalisasi)

# =========================================================
# STOPWORDS
# =========================================================
stop_factory = StopWordRemoverFactory()
stopwords = set(stop_factory.get_stop_words())

# Tambahan stopwords umum, boleh disesuaikan
tambahan_stopwords = {
    "di", "ke", "dari", "dan", "atau", "yang", "dengan", "untuk",
    "pada", "dalam", "ini", "itu", "adalah", "sebagai", "karena"
}

stopwords.update(tambahan_stopwords)

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stopwords and len(word) > 2]

df["stopwords"] = df["Normalisasi"].apply(remove_stopwords)

# =========================================================
# STEMMING
# =========================================================
stemmer = StemmerFactory().create_stemmer()

def stemming(tokens):
    text = " ".join(tokens)
    hasil = stemmer.stem(text)
    return hasil.split()

df["stemmer"] = df["stopwords"].apply(stemming)

# Kolom final untuk LDA
df["Berita"] = df["stemmer"].apply(lambda tokens: " ".join(tokens))

# =========================================================
# SIMPAN FILE
# =========================================================
output_df = df.copy()

for col in ["Token", "Normalisasi", "stopwords", "stemmer"]:
    output_df[col] = output_df[col].apply(lambda tokens: " ".join(tokens) if isinstance(tokens, list) else tokens)

# clean.csv
clean_cols = list(df.columns[:]) 
clean_output = output_df.drop(columns=["stemmer", "Berita"], errors="ignore")
clean_output.to_csv(CLEAN_FILE, index=False, encoding="utf-8-sig")

# data_bersih.csv
output_df.to_csv(DATA_BERSIH_FILE, index=False, encoding="utf-8-sig")

print("Preprocessing selesai.")
print(f"clean.csv dibuat: {CLEAN_FILE}")
print(f"data_bersih.csv dibuat: {DATA_BERSIH_FILE}")
print(f"Jumlah data: {len(df)} baris")