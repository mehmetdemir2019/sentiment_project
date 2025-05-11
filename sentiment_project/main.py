from src.data_loader import load_imdb_data
from src.preprocessing import apply_preprocessing
from src.vectorizer import build_text_clf
from src.evaluate import full_report
from utils import save

train_df, test_df = load_imdb_data("data/aclImdb")
train_df = apply_preprocessing(train_df)
test_df  = apply_preprocessing(test_df)

clf = build_text_clf()
clf.fit(train_df["clean_review"], train_df["label"])

save(clf, "models/svm_tfidf.joblib")

full_report(clf, test_df["clean_review"], test_df["label"], name="SVM + TF-IDF")
