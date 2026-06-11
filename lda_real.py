import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import pyLDAvis
import pyLDAvis.lda_model


# =========================================================
# PATH
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VIS_DIR = BASE_DIR / "visualisasi"

DATA_DIR.mkdir(exist_ok=True)
VIS_DIR.mkdir(exist_ok=True)

INPUT_FILE = DATA_DIR / "data_bersih.csv"
HASIL_TOPIK_FILE = DATA_DIR / "hasil_topik_per_dokumen.csv"
KEYWORD_FILE = DATA_DIR / "Keyword_logistik_pemilu_2024.csv"
HTML_FILE = VIS_DIR / "lda_visualisasi.html"


# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

# Pilih kolom teks hasil preprocessing
if "Berita" in df.columns:
    text_col = "Berita"
elif "stemmer" in df.columns:
    text_col = "stemmer"
elif "stopwords" in df.columns:
    text_col = "stopwords"
else:
    raise ValueError("Kolom teks tidak ditemukan. Pastikan ada kolom 'Berita', 'stemmer', atau 'stopwords'.")

df[text_col] = df[text_col].fillna("").astype(str)
df = df[df[text_col].str.strip() != ""].reset_index(drop=True)

print(f"Jumlah dokumen untuk LDA: {len(df)}")


# =========================================================
# VECTORIZE
# =========================================================
vectorizer = CountVectorizer(
    max_df=0.95,
    min_df=2,
    max_features=1000
)

dtm = vectorizer.fit_transform(df[text_col])

feature_names = vectorizer.get_feature_names_out()


# =========================================================
# LDA MODEL
# =========================================================
jumlah_topik = 8

lda_model = LatentDirichletAllocation(
    n_components=jumlah_topik,
    random_state=42,
    learning_method="batch",
    max_iter=20
)

lda_output = lda_model.fit_transform(dtm)


# =========================================================
# HASIL TOPIK PER DOKUMEN
# =========================================================
dominant_topic = lda_output.argmax(axis=1) + 1
topic_probability = lda_output.max(axis=1)

hasil_df = df.copy()
hasil_df["Topik_Dominan"] = ["Topik " + str(t) for t in dominant_topic]
hasil_df["Probabilitas_Topik"] = topic_probability

hasil_df.to_csv(HASIL_TOPIK_FILE, index=False, encoding="utf-8-sig")


# =========================================================
# KEYWORD PER TOPIK
# =========================================================
top_n_words = 25
keyword_rows = []

for topic_idx, topic in enumerate(lda_model.components_):
    top_indices = topic.argsort()[:-top_n_words - 1:-1]

    for rank, word_idx in enumerate(top_indices, start=1):
        keyword_rows.append({
            "Topik": f"Topik {topic_idx + 1}",
            "Ranking": rank,
            "Keyword": feature_names[word_idx],
            "Bobot": topic[word_idx]
        })

keyword_df = pd.DataFrame(keyword_rows)
keyword_df.to_csv(KEYWORD_FILE, index=False, encoding="utf-8-sig")


# =========================================================
# VISUALISASI pyLDAvis
# =========================================================
vis_data = pyLDAvis.lda_model.prepare(
    lda_model,
    dtm,
    vectorizer,
    sort_topics=False
)

pyLDAvis.save_html(vis_data, str(HTML_FILE))


# =========================================================
# INFO SELESAI
# =========================================================
print("LDA selesai dijalankan.")
print(f"Hasil topik disimpan ke: {HASIL_TOPIK_FILE}")
print(f"Keyword disimpan ke: {KEYWORD_FILE}")
print(f"Visualisasi disimpan ke: {HTML_FILE}")
print(f"Jumlah topik: {jumlah_topik}")
print(f"Jumlah dokumen hasil topik: {len(hasil_df)}")
print(f"Jumlah keyword: {len(keyword_df)}")