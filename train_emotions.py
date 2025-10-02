# train_emotions.py
import os, json, pickle
import numpy as np
from datasets import load_dataset
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models

# -------- CONFIG --------
EMOTIONS = ['joy','sadness','anger','fear','surprise','neutral']  # chosen subset
NUM_WORDS = 20000
MAX_LEN = 128
EMBED_DIM = 128
BATCH_SIZE = 64
EPOCHS = 4
OUT_DIR = "model_artifacts"
os.makedirs(OUT_DIR, exist_ok=True)
# ------------------------

print("Loading GoEmotions...")
dataset = load_dataset("go_emotions")
label_names = dataset['train'].features['labels'].feature.names
subset_indices = [label_names.index(e) for e in EMOTIONS]

def multi_hot_for_sample(label_list):
    mh = np.zeros(len(EMOTIONS), dtype=int)
    for lbl in label_list:
        if lbl in subset_indices:
            pos = subset_indices.index(lbl)
            mh[pos] = 1
    return mh

def prepare_split(split):
    texts = []
    labels = []
    for ex in dataset[split]:
        mh = multi_hot_for_sample(ex['labels'])
        if mh.sum() == 0:
            continue
        texts.append(ex['text'])
        labels.append(mh)
    return texts, np.array(labels)

print("Preparing data...")
train_texts, y_train = prepare_split('train')
val_texts, y_val = prepare_split('validation')
test_texts, y_test = prepare_split('test')

print(f"Samples -> train: {len(train_texts)} val: {len(val_texts)} test: {len(test_texts)}")

print("Tokenizing...")
tokenizer = Tokenizer(num_words=NUM_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(train_texts)
X_train = pad_sequences(tokenizer.texts_to_sequences(train_texts), maxlen=MAX_LEN)
X_val = pad_sequences(tokenizer.texts_to_sequences(val_texts), maxlen=MAX_LEN)
X_test = pad_sequences(tokenizer.texts_to_sequences(test_texts), maxlen=MAX_LEN)

print("Building model...")
model = models.Sequential([
    layers.Embedding(NUM_WORDS, EMBED_DIM, input_length=MAX_LEN),
    layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
    layers.GlobalMaxPool1D(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(EMOTIONS), activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

print("Training...")
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                    validation_data=(X_val, y_val))

print("Evaluating...")
preds = model.predict(X_test)
preds_bin = (preds >= 0.5).astype(int)

print("Classification report:")
print(classification_report(y_test, preds_bin, target_names=EMOTIONS, zero_division=0))

# Save artifacts
model.save(os.path.join(OUT_DIR, "emotion_model.h5"))
with open(os.path.join(OUT_DIR, "tokenizer.pkl"), "wb") as f:
    pickle.dump(tokenizer, f)
with open(os.path.join(OUT_DIR, "label_map.json"), "w") as f:
    json.dump({"emotions": EMOTIONS, "max_len": MAX_LEN, "num_words": NUM_WORDS}, f)

# Save training curves
import matplotlib.pyplot as plt
plt.figure()
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.title('Accuracy')
plt.savefig(os.path.join(OUT_DIR, 'accuracy.png'))

plt.figure()
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.title('Loss')
plt.savefig(os.path.join(OUT_DIR, 'loss.png'))

print("Saved model and artifacts to", OUT_DIR)
