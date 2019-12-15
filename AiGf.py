import speech_recognition as sr 
from gtts import gTTS
import os
import playsound
num=1
def assistant_speaks(output):
	global num
	num+=1
	toSpeak = gTTS(text=output, lang ='en', slow = False) 
	    # saving the audio file given by google text to speech 
	file = str(num)+".mp3"  
	toSpeak.save(file) 
	      
	    # playsound package is used to play the same file. 
	playsound.playsound(file, True)  
	os.remove(file) 
def get_audio():
	rObject = sr.Recognizer()
	audio=''
	with sr.Microphone() as source :
		print('speak ...')

		audio = rObject.listen(source,phrase_time_limit=5)
	print('Stop.')
	
	try:
		text=rObject.recognize_google(audio,language = 'en-US')
		print("You : ",text)
		return text
	except :
	#	assistant_speaks("could not understand your audio,Please try again !")
		return 0
if __name__=='__main__':
	assistant_speaks('   Hey babe')
	assistant_speaks('how was your day ?')
	answer = get_audio()			
	while(1):
		assistant_speaks('I really missed you ! what is our plan today ?')
		text=get_audio().lower()
		if text == 0 :
			continue
		if 'exit' in str(text) or 'bye' in str(text) or 'sleep' in str(text):
			assistant_speaks('Ok by . take care of yourself baby.')
			break
