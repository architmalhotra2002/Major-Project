import pyttsx3

def Say(Text):
    engine = pyttsx3.init("sapi5")
    voices = engine.getProperty('voices')
    
    # Adjust voice selection based on available voices
    engine.setProperty('voice', voices[0].id)  # You can change [0] to select a different voice
    engine.setProperty('rate', 170)  # Adjust the speech rate as needed
    
    # Print the spoken text
    print("A.I:", Text)
    
    # Speak the text
    engine.say(Text)
    engine.runAndWait()
