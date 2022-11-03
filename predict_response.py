import nltk
import numpy as np
import random
import json
import pickle

ignore_words = ['?', '!',',','.', "'s", "'m"]
import tensorflow 
from data_preprocessing import get_stem_words

model = tensorflow.keras.models.load_model('./chatbot_model.h5')
intents = json.loads(open('C:/Users/tomas/Proyectos Byjus School/clase120/intents.json').read())
words = pickle.load(open('C:/Users/tomas/Proyectos Byjus School/clase120/words.pkl', 'rb'))
classes = pickle.load(open('C:/Users/tomas/Proyectos Byjus School/clase120/classes.pkl', 'rb'))

def preprocess_users_input(users_input):
    tokenizer = nltk.word_tokenize(users_input)
    stemming = get_stem_words(tokenizer, ignore_words)
    list_word = sorted(list(stemming))

    bag = []
    bigBag = []

    for recorrer in words:
        if recorrer in list_word:
            bag.append(1)
        else:
            bag.append(0)

    bigBag.append(bag)

    return np.array(bigBag)

def predict_text(users_input):
    input_preprocess = preprocess_users_input(users_input)
    prediction = model.predict(input_preprocess)
    prediction_value = np.argmax(prediction[0])

    return prediction_value

def bot_response(users_input):
    prediction_class = predict_text(users_input)
    predicton_response = classes[prediction_class]

    for response in intents["intents"]:
        if response["tag"] == predicton_response:
            save_bot_response = random.choice(response["responses"])
            return save_bot_response

print("Hola soy bot tu asistente personal")
while True:
    users_input = input("Mensaje del usuario:")
    print("Usuario:",users_input)
    bot_response_show = bot_response(users_input)
    print("Bot:", bot_response_show)