import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from gtts import gTTS
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force CPU, skip GPU init
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # quieter TF logs


# =====================
# Model definitions
# =====================

class Encoder(tf.keras.Model):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.dense = tf.keras.layers.Dense(embed_dim)

    def call(self, features):
        x = self.dense(features)
        return tf.keras.activations.relu(x)


class AttentionModel(tf.keras.Model):
    def __init__(self, units: int):
        super().__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features shape: (batch, 64, embed_dim)
        # hidden shape:   (batch, units)
        hidden_with_time_axis = hidden[:, tf.newaxis, :]  # (batch, 1, units)
        score = tf.keras.activations.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tf.keras.activations.softmax(self.V(score), axis=1)  # (batch, 64, 1)
        context_vector = attention_weights * features                              # (batch, 64, embed_dim)
        context_vector = tf.reduce_sum(context_vector, axis=1)                     # (batch, embed_dim)
        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, embed_dim: int, units: int, vocab_size: int):
        super().__init__()
        self.units = units
        self.attention = AttentionModel(units)
        self.embed = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )
        self.d1 = tf.keras.layers.Dense(units)
        self.d2 = tf.keras.layers.Dense(vocab_size)

    def call(self, x, features, hidden):
        context_vector, attention_weights = self.attention(features, hidden)
        x = self.embed(x)  # (batch, 1, embed_dim)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)  # (batch, 1, embed_dim+embed_dim)
        output, state = self.gru(x)                                     # output: (batch, 1, units)
        output = self.d1(output)                                        # (batch, 1, units)
        output = tf.reshape(output, (-1, output.shape[2]))              # (batch, units)
        logits = self.d2(output)                                        # (batch, vocab_size)
        return logits, state, attention_weights

    def init_state(self, batch_size: int):
        return tf.zeros((batch_size, self.units))


# =====================
# Constants
# =====================

EMBED_DIM = 256
UNITS = 512
VOCAB_SIZE = 5001
MAX_LENGTH = 31
ATTN_FEATURES = 64       # 8*8 from InceptionV3 feature map
FEATURE_DEPTH = 2048     # <— important: depth from InceptionV3
IMAGE_SHAPE = (299, 299)


# =====================
# Utilities
# =====================

@st.cache_resource
def load_image_backbone():
    image_model = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet")
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    return tf.keras.Model(new_input, hidden_layer)


@st.cache_resource
# ===== FIX load_caption_models() =====
@st.cache_resource(show_spinner=False)  # you can add ttl or a version string if needed
def load_caption_models(cache_bump: int = 1):
    encoder = Encoder(EMBED_DIM)

    # Build with correct shape so Dense kernel is (2048, 256),
    # which matches your saved weights.
    _ = encoder(tf.zeros((1, ATTN_FEATURES, FEATURE_DEPTH)))  # <— 2048 here

    decoder = Decoder(embed_dim=EMBED_DIM, units=UNITS, vocab_size=VOCAB_SIZE)
    _ = decoder(
        x=tf.random.uniform((1, 1), maxval=VOCAB_SIZE, dtype=tf.int32),
        features=tf.random.uniform((1, ATTN_FEATURES, EMBED_DIM)),
        hidden=tf.zeros((1, UNITS)),
    )

    encoder.load_weights("outputs_encoder.h5")
    decoder.load_weights("outputs_decoder.h5")
    return encoder, decoder



@st.cache_resource
def load_tokenizer(tokenizer_path: str = "tokenizer.json"):
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        tokenizer_json = json.load(f)
    return tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)


def preprocess_image(image_path: str):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SHAPE)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img


def plot_attention_map(caption_tokens, weights, image_path):
    # Build a figure Grid and return it for Streamlit
    fig = plt.figure(figsize=(10, 10))
    temp_img = np.array(Image.open(image_path))

    cap_len = len(caption_tokens)
    grid = int(np.ceil(np.sqrt(cap_len)))
    for i in range(cap_len):
        ax = fig.add_subplot(grid, grid, i + 1)
        ax.set_title(caption_tokens[i], fontsize=8, color="red")
        ax.imshow(temp_img)
        w = np.reshape(weights[i], (8, 8))
        ax.imshow(w, cmap="gist_heat", alpha=0.6, extent=[0, temp_img.shape[1], temp_img.shape[0], 0])
        ax.axis("off")

    fig.tight_layout()
    return fig


def evaluate(image_path, encoder, decoder, tokenizer, image_features_extract_model):
    attention_plot = np.zeros((MAX_LENGTH, ATTN_FEATURES), dtype=np.float32)
    hidden = decoder.init_state(batch_size=1)

    temp_input = tf.expand_dims(preprocess_image(image_path), 0)
    img_tensor = image_features_extract_model(temp_input)          # (1, 8, 8, 2048)
    img_tensor = tf.reshape(img_tensor, (img_tensor.shape[0], -1, img_tensor.shape[3]))  # (1, 64, 2048)

    features = encoder(img_tensor)  # (1, 64, 256)

    # Prepare start token
    if "<start>" not in tokenizer.word_index:
        raise ValueError("Tokenizer is missing the '<start>' token.")
    dec_input = tf.expand_dims([tokenizer.word_index["<start>"]], 0)

    result_tokens = []

    for i in range(MAX_LENGTH):
        logits, hidden, attn = decoder(dec_input, features, hidden)
        attention_plot[i] = tf.reshape(attn, (-1,)).numpy()

        predicted_id = int(tf.argmax(logits[0]).numpy())
        predicted_token = tokenizer.index_word.get(predicted_id, "<unk>")
        result_tokens.append(predicted_token)

        if predicted_token == "<end>":
            break

        dec_input = tf.expand_dims([predicted_id], 0)

    # Trim plot to actual length
    attention_plot = attention_plot[: len(result_tokens), :]
    return result_tokens, attention_plot


def synthesize_audio(text: str, out_path: str = "voice.mp3") -> str:
    tts = gTTS(text, lang="en", slow=False)
    tts.save(out_path)
    return out_path


# =====================
# Streamlit App
# =====================

def main():
    st.title("Image Captioning with Voice Output")

    # Load resources once
    tokenizer = load_tokenizer("tokenizer.json")
    encoder, decoder = load_caption_models(cache_bump=2)
    feat_model = load_image_backbone()

    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Save file
        tmp_path = "temp_image.jpg"
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        st.image(Image.open(uploaded_file), caption="Uploaded Image", use_column_width=True)

        if st.button("Predict Caption"):
            with st.spinner("Generating caption..."):
                tokens, attn = evaluate(tmp_path, encoder, decoder, tokenizer, feat_model)

                # Build sentence and strip final <end> if present
                if tokens and tokens[-1] == "<end>":
                    tokens = tokens[:-1]
                caption = " ".join(tokens)

                st.write(f"**Predicted Caption:** {caption}")

                # Attention map
                fig = plot_attention_map(tokens, attn, tmp_path)
                st.pyplot(fig)

                # Audio
                audio_path = synthesize_audio(f"Predicted Caption: {caption}")
                with open(audio_path, "rb") as f:
                    st.audio(f.read(), format="audio/mp3")

if __name__ == "__main__":
    main()
