# src/preprocessing.py
import ssl
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 1) Gerekli nltk paketlerini bulamazsa indir
for pkg in ["punkt", "wordnet", "stopwords"]:
    try:
        nltk.data.find(f"{'corpora' if pkg != 'punkt' else 'tokenizers'}/{pkg}")
    except LookupError:
        try:  # proxy / self-signed sertifika sorunlarına karşı
            _ctx = ssl._create_unverified_context
            ssl._create_default_https_context = _ctx
        except Exception:
            pass
        nltk.download(pkg, quiet=True)

# 2) Ön hazırlık
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)          # HTML etiketlerini sil
    text = re.sub(r"[^a-z\s]", "", text)       # harf ve boşluk dışındakileri sil
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w)          # kök bul
              for w in tokens
              if w not in stop_words]          # stop-word filtrele
    return " ".join(tokens)

def apply_preprocessing(df, column: str = "review"):
    df["clean_review"] = df[column].apply(preprocess_text)
    return df
