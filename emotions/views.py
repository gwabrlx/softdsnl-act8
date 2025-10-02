# emotions/views.py
import os, json, pickle
import numpy as np
import tensorflow as tf
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE = settings.BASE_DIR
MODEL_PATH = os.path.join(BASE, "model_artifacts", "emotion_model.h5")
TOKENIZER_PATH = os.path.join(BASE, "model_artifacts", "tokenizer.pkl")
LABEL_PATH = os.path.join(BASE, "model_artifacts", "label_map.json")

# load once
model = tf.keras.models.load_model(MODEL_PATH)
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)
with open(LABEL_PATH, "r") as f:
    cfg = json.load(f)
EMOTIONS = cfg["emotions"]
MAX_LEN = cfg["max_len"]

class PredictEmotion(APIView):
    def post(self, request):
        text = request.data.get("text", "")
        if not text:
            return Response({"detail": "No text provided."}, status=status.HTTP_400_BAD_REQUEST)
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=MAX_LEN)
        preds = model.predict(padded)[0]
        top_idx = int(np.argmax(preds))
        top_emotion = EMOTIONS[top_idx]
        scores = {EMOTIONS[i]: float(preds[i]) for i in range(len(EMOTIONS))}
        return Response({"text": text, "predicted_emotion": top_emotion, "scores": scores})
