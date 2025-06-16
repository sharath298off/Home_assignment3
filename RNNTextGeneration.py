import tensorflow as tf
import numpy as np
import os

# Load txt data
path_to_file = "shakespeare.txt"  # Replace with your path
text = open(path_to_file, 'r', encoding='utf-8').read()
print("Total characters loaded:", len(text))

# Create character vocabulary
vocab = sorted(set(text))
print("Vocabulary size:", len(vocab))

char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text])

# Prepare dataset
seq_length = 30
examples_per_epoch = len(text) // (seq_length + 1)

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

def split_input_target(chunk):
    return chunk[:-1], chunk[1:]

dataset = sequences.map(split_input_target)
BATCH_SIZE = 16
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE).repeat()

print("Batches per epoch:", examples_per_epoch // BATCH_SIZE)

# Model parameters
vocab_size = len(vocab)
embedding_dim = 64
rnn_units = 256

def build_model(batch_size):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(None,), batch_size=batch_size),
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])

model = build_model(BATCH_SIZE)

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model.compile(optimizer='adam', loss=loss)

# Checkpoints
checkpoint_dir = './training_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, "model.weights.h5")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    verbose=1
)

EPOCHS = 3
steps_per_epoch = examples_per_epoch // BATCH_SIZE

model.fit(dataset, epochs=EPOCHS, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint_callback])

# Reload model for generation
model = build_model(1)
model.load_weights(checkpoint_path)
model.build(tf.TensorShape([1, None]))

def generate_text(model, start_string, num_generate=300, temperature=1.0):
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []

    # Correctly reset LSTM states here:
    model.layers[1].reset_states()

    for _ in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])

    return start_string + ''.join(text_generated)

print("\nGenerated Text:\n")
print(generate_text(model, start_string="To be, or not to be, ", temperature=0.7))
