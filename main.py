import string
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load mô hình TensorFlow và tokenizer
model = load_model('./model/model.h5')
with open('./model/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

max_sequence_len = 20  # Chiều dài tối đa của chuỗi

def predict_next_words(seed_text, next_words=5):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predicted_probs = model.predict(token_list)
        predicted_word = tokenizer.index_word[np.argmax(predicted_probs)]
        seed_text += " " + predicted_word
    return seed_text

def get_all_predictions(input_text, top_clean=5):
    print("Input text:", input_text)
    predictions = predict_next_words(input_text, top_clean)
    
    # Trả về kết quả chỉ cho mô hình TensorFlow
    return {
        'bert': predictions,
        'xlnet': "",
        'xlm': "",
        'bart': "",
        'electra': "",
        'roberta': ""
    }
