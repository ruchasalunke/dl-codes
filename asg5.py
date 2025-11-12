import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Lambda
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
# Example: To load your own text file with sentences, uncomment and modify the following lines:
# with open('your_text_file.txt', 'r', encoding='utf-8') as file:
#     text = file.readlines()
#     # Optionally strip whitespace/newlines
#     text = [line.strip() for line in text if line.strip()]
# If no file is loaded, fallback sample sentences:
text = [
 "Machine learning models can learn word embeddings",
  "Continuous Bag of Words is one Word2Vec model",
   "Neural networks are powerful tools for NLP tasks"
]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)
vocab_size = len(tokenizer.word_index) + 1
sequences = tokenizer.texts_to_sequences(text)

window_size = 2
embedding_dim = 10

contexts = []
targets = []

for sentence in sequences:
    for i, word in enumerate(sentence):
        start = max(0, i - window_size)
        end = i + window_size + 1
        context_words = [sentence[j] for j in range(start, end) if j != i and j < len(sentence)]
        if len(context_words) < window_size * 2:
            context_words = [0]*(window_size*2 - len(context_words)) + context_words  # pad if needed
        contexts.append(context_words)
        targets.append(word)


X = np.array(contexts)
y = to_categorical(targets, num_classes=vocab_size)

model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=window_size * 2),
    Lambda(lambda x: K.mean(x, axis=1)),
    Dense(vocab_size, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam')

model.summary()
model.fit(X, y, epochs=10, batch_size=16)

weights = model.get_weights()[0]
with open("vectors_simple.txt", "w") as f:
    f.write(f"{vocab_size} {embedding_dim}\n")
    for word, i in tokenizer.word_index.items():
        vector = weights[i]
        vector_str = ' '.join(map(str, vector))
        f.write(f"{word} {vector_str}\n")

print("Training complete and embeddings saved to vectors_simple.txt")
