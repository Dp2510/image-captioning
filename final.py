import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from gtts import gTTS
from io import BytesIO
import base64
import IPython.display as display
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Embedding, GRU
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import json
import IPython.display as display
import matplotlib.pyplot as plt
import cv2

tf.compat.v1.reset_default_graph()

class Encoder(Model):
    def __init__(self, embed_dim):
        super(Encoder, self).__init__()
        self.dense = tf.keras.layers.Dense(embed_dim)

    def call(self, features):
        features = self.dense(features)
        features = tf.keras.activations.relu(features)
        return features

class Attention_model(Model):
    def __init__(self, units):
        super(Attention_model, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
        self.units=units

    def call(self, features, hidden):
        hidden_with_time_axis = hidden[:, tf.newaxis]
        score = tf.keras.activations.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tf.keras.activations.softmax(self.V(score), axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights
    
class Decoder(Model):
   def __init__(self, embed_dim, units, vocab_size):
        super(Decoder, self).__init__()
        self.units=units
        self.attention = Attention_model(self.units) #iniitalise your Attention model with units
        self.embed = tf.keras.layers.Embedding(vocab_size, embed_dim) #build your Embedding layer
        self.gru = tf.keras.layers.GRU(self.units,return_sequences=True,return_state=True,recurrent_initializer='glorot_uniform')
        self.d1 = tf.keras.layers.Dense(self.units) #build your Dense layer
        self.d2 = tf.keras.layers.Dense(vocab_size) #build your Dense layer


   def call(self,x,features, hidden):
        context_vector, attention_weights = self.attention(features, hidden) #create your context vector & attention weights from attention model
        embed = self.embed(x) # embed your input to shape: (batch_size, 1, embedding_dim)
        embed = tf.concat([tf.expand_dims(context_vector, 1), embed], axis = -1) # Concatenate your input with the context vector from attention layer. Shape: (batch_size, 1, embedding_dim + embedding_dim)
        output,state = self.gru(embed) # Extract the output & hidden state from GRU layer. Output shape : (batch_size, max_length, hidden_size)
        output = self.d1(output)
        output = tf.reshape(output, (-1, output.shape[2])) # shape : (batch_size * max_length, hidden_size)
        output = self.d2(output) # shape : (batch_size * max_length, vocab_size)

        return output, state, attention_weights

   def init_state(self, batch_size):
        return tf.zeros((batch_size, self.units))
   

embedding_dim = 256 
units = 512
vocab_size = 5001
train_num_steps = 32364
test_num_steps = 8091
max_length = 31
feature_shape =2048
attention_feature_shape = 64


encoder = Encoder(embedding_dim)
_ = encoder(tf.zeros((1, 2048))) 
decoder = Decoder(embed_dim=256, units=512, vocab_size=5001)  
dummy_x = tf.random.uniform((1, 1)) 
dummy_features = tf.random.uniform((1, 64, 256))  
dummy_hidden = tf.zeros((1, 512))  
_, _, _ = decoder(dummy_x, dummy_features, dummy_hidden)

# Load weights
encoder.load_weights('outputs_encoder.h5')
decoder.load_weights('outputs_decoder.h5')


# Function to load the tokenizer
def load_tokenizer(tokenizer_path):
    with open('tokenizer.json', 'r', encoding='utf-8') as f:
      tokenizer_json = json.load(f)
      tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)
    return tokenizer

tokenizer = load_tokenizer('tokenizer.json')

# Load the image model
image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

IMAGE_SHAPE = (299, 299)
def load_images(image_path) :
  img = tf.io.read_file(image_path, name = None)
  img = tf.image.decode_jpeg(img, channels=0)
  img = tf.image.resize(img, IMAGE_SHAPE)
  img = tf.keras.applications.inception_v3.preprocess_input(img)
  return img, image_path


def evaluate(image):
    attention_plot = np.zeros((max_length, attention_feature_shape))

    hidden = decoder.init_state(batch_size=1)

    temp_input = tf.expand_dims(load_images(image)[0], 0) 
    img_tensor_val = image_features_extract_model(temp_input) 
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder (img_tensor_val) 
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden) 
        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.argmax(predictions[0]).numpy() 
        result.append (tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot,predictions

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot,predictions


def plot_attention_map (caption, weights, image) :

  fig = plt.figure(figsize = (20, 20))
  temp_img = np.array(Image.open(image))

  cap_len = len(caption)
  for cap in range(cap_len) :
    weights_img = np.reshape(weights[cap], (8,8))
    wweights_img = np.array(Image.fromarray(weights_img).resize((224,224), Image.LANCZOS))

    ax = fig.add_subplot(cap_len//2, cap_len//2, cap+1)
    ax.set_title(caption[cap], fontsize = 14, color = 'red')

    img = ax.imshow(temp_img)

    ax.imshow(weights_img, cmap='gist_heat', alpha=0.6, extent=img.get_extent())
    ax.axis('off')
  plt.subplots_adjust(hspace=0.2, wspace=0.2)
  plt.show()


def pred_caption_audio(image_test, autoplay=False, weights=(0.5, 0.5, 0, 0)) :

    test_image = image_test
    result, attention_plot, pred_test = evaluate(test_image)
    pred_caption=' '.join(result).rsplit(' ', 1)[0]
    candidate = pred_caption.split()
    print ('Prediction Caption:', pred_caption)
    plot_attention_map(result, attention_plot, test_image)
    speech = gTTS('Predicted Caption : ' + pred_caption, lang = 'en', slow = False)
    speech.save('voice.mp3')
    audio_file = 'voice.mp3'


    return pred_caption,audio_file



# Assuming the Decoder and other necessary functions are defined similarly
# Define Streamlit app logic
def app():
    st.title('Image Captioning with Voice Output')
    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        # Save the uploaded image to a temporary file
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.getvalue())
        image = Image.open(uploaded_file)  
        st.image(image, caption='Uploaded Image', use_column_width=True)
        if st.button('Predict Caption'):
            with st.spinner('Generating caption...'):
                pred_caption, audio_file = pred_caption_audio("temp_image.jpg")
            st.write(f'Predicted Caption: {pred_caption}')
            audio_bytes = open(audio_file, 'rb').read()
            st.audio(audio_bytes, format='audio/mp3', start_time=0)

app()