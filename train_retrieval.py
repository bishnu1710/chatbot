# train_retrieval.py
import os
import json
import random
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# Make sure folders exist
os.makedirs("models", exist_ok=True)
os.makedirs("data/raw", exist_ok=True)

def load_and_format(dataset_name, split, pattern_col, response_col, tag):
    """
    Always load from Hugging Face for datasets without local copies.
    """
    print(f"📥 Loading {dataset_name}...")
    ds = load_dataset(dataset_name, split=split)
    df = pd.DataFrame(ds)
    print("📌 Columns:", df.columns.tolist())  # Debug line to see columns
    if pattern_col not in df.columns or response_col not in df.columns:
        raise KeyError(f"Expected columns '{pattern_col}' and '{response_col}' not found. Found: {df.columns.tolist()}")
    df = df.rename(columns={pattern_col: "pattern", response_col: "response"})
    df = df.dropna(subset=["pattern", "response"])
    df["tag"] = tag
    return df[["tag", "pattern", "response"]]
    # df = df.rename(columns={pattern_col: "pattern", response_col: "response"})
    # df = df.dropna(subset=["pattern", "response"])
    # df["tag"] = tag
    # return df[["tag", "pattern", "response"]]

def load_counsel_chat_local(local_path, pattern_col, response_col, tag):
    if os.path.exists(local_path):
        print(f"📄 Loading local file {local_path} instead of Hugging Face counsel-chat...")
        df = pd.read_csv(local_path)
        df = df.rename(columns={pattern_col: "pattern", response_col: "response"})
        df = df.dropna(subset=["pattern", "response"])
        df["tag"] = tag
        return df[["tag", "pattern", "response"]]
    else:
        print(f"⚠ Local file {local_path} not found. Falling back to Hugging Face counsel-chat...")
        return load_and_format("nbertagnolli/counsel-chat", "train", pattern_col, response_col, tag)

def load_amod_local(local_path, pattern_col, response_col, tag):
    if os.path.exists(local_path):
        print(f"📄 Loading local JSON {local_path} instead of Hugging Face Amod dataset...")
        records = []
        with open(local_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"⚠ Skipping bad line: {e}")
                        continue
        df = pd.DataFrame(records)
        # Make sure the expected columns exist
        if pattern_col not in df.columns or response_col not in df.columns:
            raise KeyError(f"Expected columns '{pattern_col}' and '{response_col}' not found in local file")
        df = df.rename(columns={pattern_col: "pattern", response_col: "response"})
        df = df.dropna(subset=["pattern", "response"])
        df["tag"] = tag
        return df[["tag", "pattern", "response"]]
    else:
        print(f"⚠ Local file {local_path} not found. Falling back to Hugging Face Amod dataset...")
        return load_and_format("Amod/mental_health_counseling_conversations", "train", pattern_col, response_col, tag)
def load_helios_local(file_path, text_col, tag):
    print(f"📄 Loading local Parquet {file_path} instead of Hugging Face helios dataset...")
    df = pd.read_parquet(file_path)
    if text_col not in df.columns:
        raise KeyError(f"Expected column '{text_col}' not found. Found: {df.columns.tolist()}")
    
    # Since there's only a text column, we'll use it for both pattern and response
    df = df.rename(columns={text_col: "pattern"})
    df["response"] = df["pattern"]
    df = df.dropna(subset=["pattern", "response"])
    df["tag"] = tag
    return df[["tag", "pattern", "response"]]
import pandas as pd

def load_tywei08_local(file1, file2, question_col, answer_col, tag):
    print(f"📄 Loading local CSVs {file1} and {file2} instead of Hugging Face Rajaram1996 dataset...")
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    for df, name in [(df1, file1), (df2, file2)]:
        if question_col not in df.columns or answer_col not in df.columns:
            raise KeyError(f"Expected columns '{question_col}' and '{answer_col}' not found in {name}. Found: {df.columns.tolist()}")

    df1 = df1[[question_col, answer_col]].rename(columns={question_col: "pattern", answer_col: "response"})
    df2 = df2[[question_col, answer_col]].rename(columns={question_col: "pattern", answer_col: "response"})

    df = pd.concat([df1, df2], ignore_index=True)
    df.dropna(subset=["pattern", "response"], inplace=True)

    df["tag"] = tag
    return df


# ------------------------------------
# 1. Load datasets
# ------------------------------------
datasets_list = []

# Empathetic dialogues
datasets_list.append(load_and_format(
    "facebook/empathetic_dialogues", "train",
    "prompt", "utterance", "empathetic"
))

# CounselChat (local CSV)
datasets_list.append(load_counsel_chat_local(
    "data/counsel_chat.csv",
    "questionText", "answerText", "counseling"
))

# Amod (local JSON)
datasets_list.append(load_amod_local("data/combined_dataset.json", "Context", "Response", "counseling"))

# Mental health chatbot dataset
datasets_list.append(load_helios_local(
    "data/helios_chatbot.parquet", "text","counseling"
))

# Mental health QA
datasets_list.append(load_tywei08_local(
    "data/Interview_Data_6K.csv",
    "data/Synthetic_Data_10K.csv",
    "input",  # CSV column name for questions
    "output",    # CSV column name for answers
    "counseling"
))


# # Suicide prevention / crisis conversations
# datasets_list.append(load_and_format(
#     "NickyNicky/nlp-mental-health-conversations", "train",
#     "question", "answer", "crisis"
# ))

# # Suicide-specific dataset
# datasets_list.append(load_and_format(
#     "shreya142/mental-health-suicide-dataset", "train",
#     "text", "label", "crisis"
# ))

# ------------------------------------
# 2. Combine datasets
# ------------------------------------
print("🔄 Combining datasets...")
df_all = pd.concat(datasets_list, ignore_index=True)

# ------------------------------------
# 3. Group into intents.json format
# ------------------------------------
intents = {}
for _, row in df_all.iterrows():
    tag = str(row["tag"]).strip().lower() or "general"
    pattern = str(row["pattern"]).strip()
    response = str(row["response"]).strip()
    if not pattern or not response:
        continue
    if tag not in intents:
        intents[tag] = {"patterns": [], "responses": []}
    intents[tag]["patterns"].append(pattern)
    intents[tag]["responses"].append(response)

# Add manual crisis boosters
extra_crisis_patterns = [
    "help me", "please help", "i am not okay", "i feel hopeless",
    "i can't go on", "i want to end my life", "i'm thinking of suicide",
    "i feel like dying", "i don't want to live", "i need help urgently"
]
if "crisis" in intents:
    intents["crisis"]["patterns"].extend(extra_crisis_patterns)
    intents["crisis"]["responses"].extend(intents["crisis"]["responses"][:5])

# Limit examples per intent
intents_list = []
for tag, entry in intents.items():
    patterns_sample = random.sample(entry["patterns"], min(30, len(entry["patterns"])))
    responses_sample = random.sample(entry["responses"], min(30, len(entry["responses"])))
    intents_list.append({
        "tag": tag,
        "patterns": patterns_sample,
        "responses": responses_sample
    })

# Save intents.json
with open("data/intents.json", "w", encoding="utf-8") as f:
    json.dump({"intents": intents_list}, f, indent=2, ensure_ascii=False)

print(f"✅ Saved data/intents.json with {len(intents_list)} tags")

# ------------------------------------
# 4. Train TF-IDF + Logistic Regression
# ------------------------------------
patterns = []
tags = []
for intent in intents_list:
    for p in intent["patterns"]:
        patterns.append(p)
        tags.append(intent["tag"])

print("🤖 Vectorizing patterns and training classifier...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = vectorizer.fit_transform(patterns).toarray()

le = LabelEncoder()
y = le.fit_transform(tags)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

clf = LogisticRegression(max_iter=3000)
clf.fit(X_train, y_train)

print(f"📊 Train accuracy: {clf.score(X_train, y_train):.4f}")
print(f"📊 Validation accuracy: {clf.score(X_test, y_test):.4f}")

# ------------------------------------
# 5. Save model artifacts
# ------------------------------------
joblib.dump(clf, "models/intent_clf.joblib")
joblib.dump(vectorizer, "models/vectorizer.joblib")
joblib.dump(le, "models/label_encoder.joblib")

print("✅ Saved models to 'models/'")
print("🎯 Training complete.")
