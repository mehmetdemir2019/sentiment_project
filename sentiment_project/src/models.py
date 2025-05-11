from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

def train_naive_bayes(X_train, y_train):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train):
    model = LinearSVC()
    model.fit(X_train, y_train)
    return model
