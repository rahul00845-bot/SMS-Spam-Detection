# Required Libraries
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

print("Starting script execution...", flush=True)

# Load and Clean Dataset
data = pd.read_csv("spam.csv", encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['category', 'message']

# Label Encoding (ham: 0, spam: 1)
label_enc = LabelEncoder()
data['category'] = label_enc.fit_transform(data['category'])

# Split Features and Labels
texts = data['message'].values
targets = data['category'].values

# Text Tokenization and Padding
vocab_size = 5000
oov_tok = "<OOV>"

text_tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
text_tokenizer.fit_on_texts(texts)
text_seq = text_tokenizer.texts_to_sequences(texts)
text_pad = pad_sequences(text_seq, padding='post')

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    text_pad, targets, test_size=0.2, random_state=42
)

# Neural Network Architecture
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=16, input_length=text_pad.shape[1]),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile Model
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print("Model training in progress...", flush=True)

# Training
training_history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    verbose=2
)

# Final Evaluation
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Set Accuracy: {test_accuracy:.4f}", flush=True)

# Plotting Training Progress
def visualize_training(history_obj):
    metrics = history_obj.history
    epochs_range = range(len(metrics['accuracy']))

    plt.figure(figsize=(12, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, metrics['accuracy'], label='Train Accuracy')
    plt.plot(epochs_range, metrics['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, metrics['loss'], label='Train Loss')
    plt.plot(epochs_range, metrics['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

visualize_training(training_history)
