import datetime
from speak import Say



def Time():
    time = datetime.datetime.now().strftime("%H:%M")
    Say(time)

def Date():
    date = datetime.date.today()
    Say(date)


def NonInputExecution(query):
    query = str(query)
    if "time" in query:
        Time()
    elif "date" in query:
        Date()
    
def Input_Execution (tag,query):
    if "wikipedia" in tag :
        name = str(query).replace("who is","").replace("about","").replace("what is","").replace("wikipedia","")
        import wikipedia
        result = wikipedia.summary(name)
        Say(result)
    elif "google" in tag:
        query = str(query).replace("google","")
        query = query.replace("search","")
        import wikipedia as googleScrap
        import pywhatkit
        
        result = googleScrap.summary(query)
        Say(result)
