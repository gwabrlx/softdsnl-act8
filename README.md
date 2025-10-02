# 🧠 Emotion Classification with TensorFlow + Django

## 📌 Overview
In this activity, you will build a **text classifier that detects emotions** (e.g., *joy, sadness, anger, fear, surprise, neutral*) instead of just simple sentiment (positive/negative).  
You will use the **GoEmotions dataset (by Google Research)** and connect your trained model to a **Django REST API**, which can be tested using **Postman**.

This project builds on your previous sentiment analysis activity and introduces **multi-class classification** for more fine-grained predictions.

---

## 🎯 Learning Objectives
By completing this activity, you will be able to:
- Understand the difference between **sentiment analysis** and **emotion classification**.
- Preprocess textual data for multi-class classification.
- Train and evaluate a deep learning model using **TensorFlow/Keras**.
- Integrate the trained model into a **Django REST API**.
- Use **Postman** to test real-time predictions.

---

## 📂 Deliverables
Submit the following:

1. ✅ A **PDF report** named:  
   ```
   SOFTDSNL_Emotions_Surnames.pdf
   ```
   containing:
   - Training results (accuracy & loss screenshots).
   - Postman test results (at least 2 examples for each emotion category).
   - Link to your **GitHub repository fork**.

2. ✅ Your **GitHub repo** must include:
   - `train_emotions.py` → for training the model.  
   - `requirements.txt` → with all dependencies.  
   - A **Django project** with a REST API endpoint (`/predict-emotion`) that accepts text input and returns the predicted emotion.  
   - `README.md` (this file).  

---

## 🛠 Requirements
- Python 3.9+  
- TensorFlow 2.x  
- Django + Django REST Framework  
- Numpy, Pandas, Scikit-learn  
- Hugging Face `datasets` (to load GoEmotions)

Install dependencies:
```bash
pip install -r requirements.txt
```

Example `requirements.txt`:
```
tensorflow
django
djangorestframework
numpy
pandas
scikit-learn
datasets
```

---

## 🚀 Steps

### 1️⃣ Load and Preprocess Dataset
We will use **GoEmotions** (a dataset of Reddit comments labeled with 27 emotions). For simplicity, you may select a subset like:  
`[joy, sadness, anger, fear, surprise, neutral]`.

```python
from datasets import load_dataset
import pandas as pd

# Load GoEmotions dataset
dataset = load_dataset("go_emotions")

# Use 'train' split for training, 'test' for evaluation
train_data = dataset["train"]
test_data = dataset["test"]

print(train_data[0])  # sample data
```

---

### 2️⃣ Train the Emotion Classifier
- Convert text into numerical features (use `Tokenizer` from Keras).
- Train a simple **LSTM/GRU** or a **Dense Neural Network** for multi-class classification.

Example template (`train_emotions.py`):

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models
import numpy as np
from datasets import load_from_disk
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset from disk
dataset = load_from_disk("go_emotions_dataset")  # Assuming the dataset is saved locally

# Accessing the splits
train_data = dataset["train"]
validation_data = dataset["validation"]
test_data = dataset["test"]

# Prepare data (You can choose to work with any split - here we use "train")
texts = train_data["text"]
labels = [l[0] if len(l) > 0 else 27 for l in train_data["labels"]]  # handle multi-label

# Encode labels
encoder = LabelEncoder()
labels_encoded = encoder.fit_transform(labels)

# Tokenize text
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=50)

# Build model
model = models.Sequential([
    layers.Embedding(10000, 64, input_length=50),
    layers.Bidirectional(layers.LSTM(64)),
    layers.Dense(64, activation="relu"),
    layers.Dense(len(set(labels_encoded)), activation="softmax")
])

# Compile
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Train the model
model.fit(padded, labels_encoded, epochs=5, validation_split=0.2)

# Save model + tokenizer + encoder
model.save("emotion_model.h5")

# Save tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Save encoder
with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)
```

---

### 3️⃣ Create Django REST API
Inside your Django app (e.g., `emotions`), create a `views.py`:

```python
from django.http import JsonResponse
import tensorflow as tf
import numpy as np
import pickle

# Load model and preprocessing tools
model = tf.keras.models.load_model("emotion_model.h5")
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

def predict_emotion(request):
    if request.method == "POST":
        import json
        body = json.loads(request.body)
        text = body.get("text", "")

        seq = tokenizer.texts_to_sequences([text])
        padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=50)
        prediction = model.predict(padded)
        emotion = encoder.inverse_transform([np.argmax(prediction)])[0]

        return JsonResponse({"text": text, "predicted_emotion": emotion})
```

Add this to `urls.py`:
```python
from django.urls import path
from . import views

urlpatterns = [
    path("predict-emotion", views.predict_emotion, name="predict_emotion"),
]
```

---

### 4️⃣ Test with Postman
- Run Django server:  
  ```bash
  python manage.py runserver
  ```
- In **Postman**, send a **POST request** to:  
  `http://127.0.0.1:8000/predict-emotion`  
  with body:
  ```json
  {
    "text": "I am so excited for this trip!"
  }
  ```

Expected response:
```json
{
  "text": "I am so excited for this trip!",
  "predicted_emotion": "joy"
}
```

---

## 📝 Grading Criteria
| Criteria | Points |
|----------|---------|
| Model training (accuracy/loss screenshots) | 20 |
| Django API implementation | 20 |
| Postman tests (2 per category) | 30 |
| Report (clarity, screenshots, repo link) | 20 |
| Code quality & documentation | 10 |
| **Total** | **100** |

---

## ✅ Submission
- Upload your **PDF report** and submit the link to your **GitHub repository**.
- Ensure your Django API is working and well-documented.

---
