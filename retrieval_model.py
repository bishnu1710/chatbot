# retrieval_model.py
import json
import random
import joblib
import os
import numpy as np
import re
from spellchecker import SpellChecker

MODEL_DIR = "models"
INTENTS_PATH = "data/intents.json"

class RetrievalModel:
    def __init__(self, model_dir=MODEL_DIR, intents_path=INTENTS_PATH, threshold=0.2):
        self.model_dir = model_dir
        self.intents_path = intents_path
        self.threshold = threshold
        self._loaded = False
        self.spell = SpellChecker()
        self._load()

    def _load(self):
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError("models/ directory not found. Run train_retrieval.py first.")
        self.clf = joblib.load(os.path.join(self.model_dir, "intent_clf.joblib"))
        self.vectorizer = joblib.load(os.path.join(self.model_dir, "vectorizer.joblib"))
        self.le = joblib.load(os.path.join(self.model_dir, "label_encoder.joblib"))
        with open(self.intents_path, "r", encoding="utf-8") as f:
            self.intents = json.load(f)["intents"]
        # map tag -> responses
        self.responses_by_tag = {item["tag"]: item["responses"] for item in self.intents}
        self._loaded = True
    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        # Correct spelling word by word
        words = text.split()
        corrected_words = [self.spell.correction(w) or w for w in words]
        return " ".join(corrected_words).strip()
    
    def predict_tag(self, text):
        text = self.preprocess(text)
        vec = self.vectorizer.transform([text]).toarray()
        probs = self.clf.predict_proba(vec)[0]
        idx = int(np.argmax(probs))
        prob = float(probs[idx])
        tag = self.le.inverse_transform([idx])[0]
        if prob >= self.threshold:
            return tag, prob
        return None, prob

    def get_response(self, text):
        tag, prob = self.predict_tag(text)
        if tag:
            choices = self.responses_by_tag.get(tag, ["I'm here to listen; tell me more."])
            return random.choice(choices), tag, prob
        return None, None, prob
