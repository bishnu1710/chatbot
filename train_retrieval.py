# train_retrieval.py
import os
import json
import random
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

print("Downloading EmpatheticDialogues dataset from Hugging Face...")
#ds = load_dataset("facebook/empathetic_dialogues", name="train", split="train")
 # HF dataset
import pandas as pd

# Direct download from Hugging Face repo
url = "https://huggingface.co/datasets/facebook/empathetic_dialogues/resolve/main/train.csv"
ds = pd.read_csv(url)

# Now we can iterate over rows like before
for _, ex in ds.iterrows():
    tag = str(ex.get("context", "general")).strip() or "general"
    prompt = str(ex.get("prompt", "")).strip()
    utterance = str(ex.get("utterance", "")).strip()
    

print("Dataset loaded. Preparing intents...")

# Build simple intents by grouping conversations by `context` (emotion-like field),
# using `prompt` as the user pattern and `utterance` as the response.
intents = {}
patterns = []
tags = []

for ex in ds:
    tag = ex.get("context", "general").strip() or "general"
    prompt = ex.get("prompt", "").strip()
    utterance = ex.get("utterance", "").strip()
    if not prompt or not utterance:
        continue
    # Collect for intents JSON
    if tag not in intents:
        intents[tag] = {"patterns": [], "responses": []}
    # keep several patterns/responses per tag
    intents[tag]["patterns"].append(prompt)
    intents[tag]["responses"].append(utterance)
    # for classifier training
    patterns.append(prompt)
    tags.append(tag)

# Save intents.json (reduce responses per tag to at most 20 to keep it small)
intents_list = []
for tag, entry in intents.items():
    patterns_sample = entry["patterns"][:30]
    responses_sample = entry["responses"][:30]
    intents_list.append({"tag": tag, "patterns": patterns_sample, "responses": responses_sample})

with open("data/intents.json", "w", encoding="utf-8") as f:
    json.dump({"intents": intents_list}, f, indent=2, ensure_ascii=False)

print(f"Saved data/intents.json with {len(intents_list)} tags")

# Train TF-IDF + classifier
print("Vectorizing patterns and training classifier...")
vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1,2))
X = vectorizer.fit_transform(patterns).toarray()

le = LabelEncoder()
y = le.fit_transform(tags)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

clf = LogisticRegression(max_iter=2000)
clf.fit(X_train, y_train)

print("Train accuracy:", clf.score(X_train, y_train))
print("Validation accuracy:", clf.score(X_test, y_test))

# Save artifacts
joblib.dump(clf, "models/intent_clf.joblib")
joblib.dump(vectorizer, "models/vectorizer.joblib")
joblib.dump(le, "models/label_encoder.joblib")

print("Saved models to models/ (intent_clf.joblib, vectorizer.joblib, label_encoder.joblib)")
print("Training complete.")
