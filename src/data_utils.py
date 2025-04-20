import pandas as pd

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.sample(1000, random_state=42).reset_index(drop=True)
    texts = df["text"].tolist()
    labels = df["label"].tolist()
    return texts, labels