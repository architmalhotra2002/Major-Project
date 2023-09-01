# Audio file formats
# .mp3
# .flac
# .wav
import wave
#Audio signal Parameters
# - number of channels
# - sample width 
# - framerate/sample_rate: 44,100 Hz
# - number of frames 
# - value of frames
obj = wave.open("Archit.wav","rb") 
print("Number of Channels",obj.getnchannels())
print("Samplewidth",obj.getsampwidth())
print("Framerate",obj.getframerate())
print("Number of Frames",obj.getnframes())
print("Parameters",obj.getparams())
t_audio = obj.getnframes()/obj.getframerate()
print("Time of audio : ",t_audio)

frames = obj.readframes(-1)
print(type(frames),type(frames[0]))
print(len(frames))

obj.close()
obj_new = wave.open("Archit_new.wav","wb") 
obj_new.setnchannels(1)
obj_new.setsampwidth(2)
obj_new.setframerate(16000.00)
obj_new.writeframes(frames)
obj_new.close()