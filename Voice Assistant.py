import speech_recognition as sr
import os
from selenium import webdriver
from gtts import gTTS
import playsound

#NUM TO RENAME THE SOUND FILE AND AVOID SAME NAMED FILES
num = 1

def assistant_speaks(output):
	global num
	#increase num every time the function is called
	num += 1
	print('person : ',output)

	toSpeak = gTTS(text=output,lang='en',slow=False)
	#saving the audio file given by google to text speech to be played as our assitant
	file = str(num)+".mp3"
	toSpeak.save(file)

	#we'll use playsound to play our file aka assitant's voice
	playsound.playsound(file,True)

	#delete file after playing it obvoiusly
	os.remove(file)
def get_name():
	rObject = sr.Recognizer()
	audio  = ''
	with sr.Microphone() as source :
		print('Speak ...')
		rObject.adjust_for_ambient_noise(source)
		audio = rObject.listen(source,phrase_time_limit=3)
	print('Stop.')
	text = rObject.recognize_google(audio,key = None,language='en-US',show_all=False)
	return text
def get_audio():
	rObject=sr.Recognizer()
	audio=''
	with sr.Microphone() as source:
		print('Speak .... ')
		rObject.adjust_for_ambient_noise(source)
		audio = rObject.listen(source)
	print('Stop ...')
	try:

		text = rObject.recognize_google(audio)
		print('YOU SAID :', text)
		return text
	except :
		assistant_speaks('Could not understand what you said, Please repeat it again !')
		return '0'	

def process_text(input):
	try:
		if 'search' in str(input) or 'play' in str(input) :
			search_web(input)
			return 
		elif 'who are you' in str(input) or 'define yourself' in str(input) :
			speech = 'Hello, I am your vocal assitant AI, I am here to make your life much easier.'
			assistant_speaks(speech)	
		elif 'who created you' in str(input) or 'who made you' in str(input) :
			assistant_speaks('I was made by my boyfriend Mahdi')
		elif 'open' in str(input):
			open_application(input.lower())
		else :
			assistant_speaks('I will search this on web for you , Do you want me to continue')
			answer = get_audio()
			if 'yes' in str(answer) or 'yeah' in str(answer) or ' yep '	in str(answer) :
				search_web(input)
			else :
				return
	except: 
		assistant_speaks('Could not understand what you said, Please repeat it again !')
		return 

def search_web(input):
	driver = webdriver.Chrome('./Chromedriver.exe')
	driver.implicity_wait(1)
	driver.maximize_window()
	if ' youtube' in str(input.lower()) :
		assistant_speaks('Openning Youtube')
		index = input.lower().split().index('youtube')
		query=input.split()[index+1 :]
		driver.get('http://www.youtube.com/results?sear_query='+'+'.join(query))
		return
#Driver code ........	
if __name__=='__main__':
	#assistant_speaks("hey, what's your name human ?")
	#name = get_name()
	assistant_speaks('hello Mahdi.')
	while True : 
		assistant_speaks('what can i do for you ?')
		text = get_audio().lower()
		if text == '0' : 
			continue
		if  "exit" in str(text) or "leave" in str(text)  or "bye" in str(text) or "go" in str(text) or "sleep" in str(text) :
			assistant_speaks('good bye dear , '+ name+'.')
			break
		process_text(text)

