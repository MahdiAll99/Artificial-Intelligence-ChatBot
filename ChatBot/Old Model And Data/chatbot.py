import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer  = LancasterStemmer()

import numpy
import tflearn
import tensorflow as tf
import json
import random
import numpy as np
import pickle
import os
from tensorflow.keras.callbacks import TensorBoard
import datetime


log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
'''log_dir = os.path.join(
    "logs",
    "fit",
    datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
)'''
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)


with open("intents.json") as file :
	data = json.load(file)

#print(data)
#we are gonna try to open up some saved data (so we dont create the date eveytime the program runs,it runs one time then it's saved for life) and if that doesnt work we are gonna generate the data
try :
	with open("data.pickle","rb") as f :
		words,labels,training,output = pickle.load(f)

except:
	print('\nCreating Data......\n')
	words=[]
	labels=[]
	#every doc_x (pattern) has its doc_y (tag) so now we know every pattern and the tag that it belongs to
	docs_x=[]
	docs_y=[]

	for intent in data["intents"]:
		for pattern in intent["patterns"]:
		#stemming take each word in our patternand bring it down to the root word 
		#what's ip ? change it just to what to get the root of the worl .we care about the meaning of the world
			wrds = nltk.word_tokenize(pattern) #return a list of all the words in the input sentence
			#add all those wrds to words
			words.extend(wrds) 
			docs_x.append(wrds)
			docs_y.append(intent["tag"])
		if intent["tag"] not in labels:
			#fill the labels with out tags 
			labels.append(intent["tag"])
	#lets stem every word and remove duplicates
	#A word stem is part of a word. It is sort of a normalization idea, but linguistic. For example, the stem of the word waiting is wait
	
	words = [stemmer.stem(w.lower()) for w in words]
	words = sorted(list(set(words)))

	labels = sorted(labels)

	training = []
	output = []

	out_empty=[0 for _ in range(len(labels))]

	for x,doc in enumerate(docs_x):
		bag = [] #bag of words 
		wrds = [stemmer.stem(w) for w in doc] #we are gonna stem all of the words that are in our patterns_we didnt stem them we added them to words

	#now we are gonna go through all the different words that are in our stemmed list and we are gonna put either 1 or 0 into our bag of words depending if it's in the main word list(words)or not
		for w in words:
			if w in words:
				bag.append(1)
			else :
				bag.append(0)

		output_row = out_empty[:]
		output_row[labels.index(docs_y[x])] = 1 #look through the labels list and see where the tag is in that list and set that value to 1 in out output row 
											   #b darija ghadi n9lbo 3la fin kayn had tag dialna f labels then an7to 1 f output_row f index fach l9inah 
		training.append(bag)
		output.append(output_row)

	training = np.array(training)
	otuput = np.array(output)	

	#with open("data.pickle","wb") as f :
	#	pickle.dump((words,labels,training,output),f)

tf.reset_default_graph()

net = tflearn.input_data(shape=[None,len(training[0])])
net=tflearn.fully_connected(net,8) #add this fully connected layer to our neural network and we are gonna have  4 neurons
net=tflearn.fully_connected(net,8)
net=tflearn.fully_connected(net,len(output[0]),activation='softmax') #softmax is gonna go and give us a probability for each neuron in this layer and that will be the output for the network
net=tflearn.regression(net) #minimize the provided loss #layer implements the linear or logistic regression

model = tflearn.DNN(net)

#if the model already exists so no need to train it again :)	

if os.path.exists("model.tflearn.meta"): 

	model.load("./model.tflearn")

else:	
	#passing all our data to the model
	model.fit(training,output,n_epoch=10000,batch_size=8,show_metric=True) #Trains the model for a fixed number of epochs (iterations on a dataset)
	#model.fit(training,output,n_epoch=1000,validation_data=(training, output),callbacks=[tensorboard_callback])
	#batch_size: Integer or None. Number of samples per gradient update
	#epochs: Integer. Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided
	model.save("model.tflearn") #save the model obiously to not train it everytime

def bag_of_words(s,words):
	bag=[0 for _ in range(len(words))]

	s_words = nltk.word_tokenize(s)
	s_words = [stemmer.stem(word.lower()) for word in s_words]

	for se in s_words:
		for i,w in enumerate(words):
			if w==se:
				bag[i] = 1

	return np.array(bag)			


def chat():
    print("ALL bot under your commad type to talk,and type quit then i will fuck off!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))

chat()
