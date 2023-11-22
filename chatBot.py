import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Sample conversation data
conversations = [
    ('Hello', 'Hi there!'),
    ('How are you?', 'I\'m good, thank you. How about you?'),
    ('What do you do?', 'I\'m a chatbot.'),
    ('Bye', 'Goodbye!'),
]

# Create vocabulary
all_words = set()
for conv in conversations:
    all_words.update(conv[0].split())
    all_words.update(conv[1].split())

word2idx = {word: idx + 1 for idx, word in enumerate(all_words)}
idx2word = {idx + 1: word for idx, word in enumerate(all_words)}

# Convert conversation data to sequences of indices
X = []
y = []
for conv in conversations:
    X.append([word2idx[word] for word in conv[0].split()])
    y.append([word2idx[word] for word in conv[1].split()])

# Calculate max_len
max_len_X = max(len(seq) for seq in X)
max_len_y = max(len(seq) for seq in y)
max_len = max(max_len_X, max_len_y)

# Pad sequences
X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=max_len, padding='post')
y = tf.keras.preprocessing.sequence.pad_sequences(y, maxlen=max_len, padding='post')

# Build the model
model = Sequential()
model.add(Embedding(input_dim=len(all_words) + 1, output_dim=64, input_length=max_len))
model.add(LSTM(128))
model.add(Dense(len(all_words) + 1, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['categorical_accuracy'])

# Print model summary
model.summary()
print("Shape of y:", y.reshape(-1, max_len).shape)
print("Shape of X:", X.shape)

# Train the model
model.fit(X, y.reshape(-1, max_len), epochs=100, batch_size=1)

# model.fit(X, y.reshape(-1, max_len, 1), epochs=100, batch_size=1)

# Function to generate a response
def generate_response(input_text):
    input_seq = [word2idx[word] for word in input_text.split()]
    input_seq = tf.keras.preprocessing.sequence.pad_sequences([input_seq], maxlen=max_len, padding='post')
    predicted_idx = np.argmax(model.predict(input_seq), axis=-1)
    return ' '.join(idx2word[idx] for idx in predicted_idx[0] if idx != 0)

# Test the chatbot
user_input = 'Hello'
while user_input.lower() != 'bye':
    response = generate_response(user_input)
    print(f'User: {user_input}')
    print(f'Chatbot: {response}\n')
    user_input = input('User: ')
