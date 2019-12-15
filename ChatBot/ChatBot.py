import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import playsound
from gtts import gTTS
import speech_recognition as sr
import numpy
import tflearn
import tensorflow
import random
import json
import pickle
import os 

#let's deffine our speaking and listening functions 

num = 1
def assitant_speaks(output):
    global num
    num +=1
    sound = gTTS(text=output,lang='en',slow=False)
    file = str(num)+".mp3"
    sound.save(file)
    playsound.playsound(file)
    os.remove(file)

def get_audio():
    rObject = sr.Recognizer()
    audio = ''
    rObject.energy_threshold = 50
    rObject.dynamic_energy_threshold = False
    with sr.Microphone() as source : 
        print('Speak :...')
        rObject.adjust_for_ambient_noise(source)
        audio = rObject.listen(source)
    print('Stop.')
    try:
        text = sr.recognize_google(audio,key=None,language='en-US',show_all=False)
        print('\n You said  : ',text)
        return text
    except:
        assitant_speaks('Sorry i didn\'t get that ...Speak again')
        get_audio()


with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

if os.path.exists("./model.tflearn.meta"):
    model.load("./model.tflearn")
else:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)


def chat():
    print("Start talking with the bot (say quit to stop)!")
    while True:
        inp = input("Wrtie : ")
        if inp.lower() == "quit":
            break
        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        assitant_speaks(random.choice(responses))

chat()