import os
import pandas as pd

def load_imdb_data(data_dir):
    def read_reviews(path, sentiment):
        data = []
        for file in os.listdir(path):
            with open(os.path.join(path, file), encoding="utf8") as f:
                data.append((f.read(), sentiment))
        return data

    train_pos = read_reviews(os.path.join(data_dir, 'train/pos'), 1)
    train_neg = read_reviews(os.path.join(data_dir, 'train/neg'), 0)
    test_pos = read_reviews(os.path.join(data_dir, 'test/pos'), 1)
    test_neg = read_reviews(os.path.join(data_dir, 'test/neg'), 0)

    train_df = pd.DataFrame(train_pos + train_neg, columns=['review', 'label'])
    test_df = pd.DataFrame(test_pos + test_neg, columns=['review', 'label'])

    return train_df, test_df

# Veri yolu
data_dir = 'aclImdb'
train_df, test_df = load_imdb_data(data_dir)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

train_df.head()

