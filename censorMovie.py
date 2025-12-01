import subprocess
import json
import os
import whisperx
import soundfile as sf 
import numpy as np 

#TODO:
# - Write a wrapper script that calls on this script to batch run 
# - Error file that appends to include files that failed
# - MUCH LATER: have a flag to indicate not to take the first audio file


#.....
# - Majority match for profanity (e.g. dont want to throw out 'embarassing')
# - Summarized list of profanity omitted 
# - Setup script to save files in appropriate directories (scratch and such)

#Left off: 

# - Whitelist ("hello")
# - If nothing is removed, skip file but note its skipped


# /root/VideoCleanUp/whisper.cpp/models/ggml-base.bin
# ./build/bin/whisper-cli -m models/ggml-base.bin -f samples/jfk.wav --max-len 1 -of "test2" -oj
# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------

# PROFANITY_LIST = {"fuck", "shit", "bitch", "asshole", "fucker", "motherfucker", "ass", "porn", "hell","fucking","dam","twat","dick","cunt"}  # customize
TIME_PADDING = .2 #20%, e.g. tStart - (duration*.2) : tEnd + (duration*.2)
PROFANITY_LIST_FILE_NAME = "PROFLIST.json"
# ----------------------------------------------------------

def loadProfanityList():
    with open(PROFANITY_LIST_FILE_NAME,'r') as f:
        l = json.load(f)
    return l 


def runWhisperX(audio_path, output_json = "transcript.json"):
    # model = whisperx.load_model("base",device="cpu", compute_type = "int8")
    model = whisperx.load_model("base",device="cpu", compute_type = "int8", vad_method = "silero")
    model_a, metadata = whisperx.load_align_model(language_code="en",device="cpu")
    audio = whisperx.load_audio(audio_path.replace(".wav","_mono.wav"))
    
    #Extracting results
    results = model.transcribe(audio, batch_size = 16) #All text
    aligned_result = whisperx.align(results["segments"], model_a, metadata, audio, device = "cpu") #Time stamps by word
    
    #Saving it to a json file
    with open(output_json,'w') as f:
        json.dump(aligned_result,f,indent = 4) #writing data to the file
        
    return aligned_result, results 


def extract_audio(mkv_path, wav_path):
    """
    Extracts audio from MKV to WAV using ffmpeg.
    """
    
    wav_path = wav_path.replace(".wav","")
    
    #Extracting mono 
    cmd = ["ffmpeg", "-y", "-i", mkv_path, "-ac", "1", wav_path + "_mono.wav"]
    subprocess.run(cmd, check=True)
    
    #Extracting full audio 
    cmd = ["ffmpeg", "-y", "-i", mkv_path, "-map", "0:a:0", "-c:a", "pcm_s16le", wav_path + "_full.wav"]
    subprocess.run(cmd, check=True)
    

def mux_new_audio(original_mkv, censored_wav, output_mkv):
    """
    Replaces/adds the censored audio track into a new MKV.
    You can choose to add it as a second track instead of replacing.
    """

    cmd = [
        "ffmpeg",
        "-y",
        "-i", original_mkv,
        "-i", censored_wav,
        "-map", "0:v",          # keep original video
        "-map", "0:a:0",          # keep original audio
        "-map", "1:a:0",          # add censored audio
        "-c:v", "copy",
        "-c:a:0", "copy",
        "-c:a:1","flac",
        # "-metadata:s:a:1", "title=Censored Audio",
        output_mkv
    ]
    subprocess.run(cmd, check=True)

def inList(text, tarList):
    '''
    text - search text
    tarList - this is the reference list it is being compared against
    
    '''
    
    PERCENT_MATCH = 0.6 #Percentage match, expressed as decimal
    
    textCount = len(text) #Number of characters in search text
    for l in tarList:
        if l in text:
            fullLen = len(l)
            if fullLen/textCount >= PERCENT_MATCH:
                return True 
    
    return False 

def removedContentSummary(languageRemoved):
    print('-'*46)
    print('\n')
    print(f'Word          : Count')
    totalCount = 0
    for key in languageRemoved.keys():
        print(f'{key:14}: {languageRemoved[key]:5}')
        totalCount += languageRemoved[key]
    print('-'*21)
    print(f'Words blocked: {totalCount:4}')


        

def censor_audio(original_wav, transcript, PROFANITY_LIST):
    output_wav = original_wav.replace(".wav", "_censored.wav")
    original_wav = original_wav.replace(".wav","") + "_full.wav"
    data, samplerate = sf.read(original_wav)   # data.shape = (samples, channels)
    languageRemoved = {}
    
    if data.ndim == 1:
        data = data[:, None]
        
    for segment in transcript["word_segments"]:
        text = segment["word"].lower().replace(" ","")
        contains_profanity = inList(text, PROFANITY_LIST)
        if not contains_profanity:
            continue
        
        if text in languageRemoved.keys():
            languageRemoved[text] += 1 
        else:
            languageRemoved[text] = 1
        
        start_sec = segment["start"] 
        end_sec = segment["end"]
        duration = end_sec - start_sec
        
        #Padding around word to ensure it isn't missed
        start_sec -= TIME_PADDING*duration 
        end_sec += TIME_PADDING*duration 

        #Converting to frames
        startFrame = int(start_sec * samplerate)
        endFrame = int(end_sec * samplerate)
        
        #Silence all channels for the interval
        data[startFrame:endFrame, :] = 0.0
    
    #Writing file now 
    sf.write(output_wav, data, samplerate, subtype = 'PCM_16')
    
    # formatted_json = json.dumps(languageRemoved, indent=4)
    # print(formatted_json)
    
    return output_wav, languageRemoved

def logRemovedLanguage(input_mkv, baseDir, languageRemoved):
    '''
    input_mkv - Name of the log file (.json)
    baseDirectory that files will be saved to. Note, will create replicate folder structure here
    
    
    
    input_mkv - name of the mkv file (original) 
    baseDir - This is the top level directory where the log files will be stored (sub folders made to mirror structure contianing the input_mkv)
    languageRemoved - Dictionary containing words/counts of removed files 
    
    !!!! LEFT OFF HERE
    '''
    
    input_mkv = os.path.abspath(input_mkv) #Getting full directory
    baseDir = os.path.abspath(baseDir) #Getting full directory 
    
    mkvFileName = input_mkv.split('/')[-1]
    subFolders = input_mkv.split('/')[-3:-1] #Getting 2 levels up from file of interest
    subFolders = [item for item in subFolders if item != "Movies"]#Dropping "Movies" if applicable
    
    if len(subFolders) > 1:
        logName =  mkvFileName.replace(".mkv","_DroppedLanguage.json")
        logName = subFolders[1] + "_" + logName 
        subFolders = subFolders[0] 
    else:
        logName =  mkvFileName.replace(".mkv","_DroppedLanguage.json")
        subFolders = subFolders[0]
    
    #Building log directory
    logDir = os.getcwd() 
    logDir = os.path.join(logDir,subFolders)
    if not os.path.exists(logDir):
        os.mkdir(logDir) 
    
    logFileDir = os.path.join(logDir,logName)

    with open(logFileDir,'w') as f:
        json.dump(languageRemoved,f,indent=4)
    
    # pass 

    # #os.path.abspath -> this will get the full directory to the file
    # cwd = os.getcwd() #Current working directory
    # mkvFile = os.path.abspath(os.path.join(cwd,input_mkv)) #Full directory of mkvfile
    # logFile = input_mkv.replace(".mkv","_DroppedLanguage.json")
    # logFile = os.path.abspath(logFile).split("/")[-1] #Getting the actual file name 
    
    # #Building the file directory for storing log file 
    

    # # os.path.exists()
    # # os.path.join()
    # # a = os.getcwd().split("/") 
    # # a[-1] & a[-2]
    # # os.mkdir() 
    # # os.path.abspath(c)
    # with open(name, "w") as f:
    #     json.dump(languageRemoved,f)


# ----------------------------------------------------------
# MAIN WORKFLOW
# ----------------------------------------------------------
def censor_mkv(input_mkv, output_mkv="output_censored.mkv",logsDir = os.getcwd()):
    temp_wav = "temp_audio.wav"
    
    PROFANITY_LIST = loadProfanityList() 

    print("[1] Extracting audio...")
    extract_audio(input_mkv, temp_wav)

    print("[2] Running WhisperX for transcription...")
    transcript, _ = runWhisperX(temp_wav)

    print("[3] Creating censored audio track...")
    censored_wav, languageRemoved = censor_audio(temp_wav, transcript, PROFANITY_LIST)

    print("[4] Muxing new MKV with additional censored audio track...")
    mux_new_audio(input_mkv, censored_wav, output_mkv)
    
    removedContentSummary(languageRemoved = languageRemoved) #Printing summary of content removed
    # removedContent(languageRemoved = languageRemoved, fileName = input_mkv.replace(".mkv","_DroppedLanguage.json"))
        # with open("output.json", "w") as json_file:
        # json.dump(data, json_file)
        
    #Saving information about language dropped
    logRemovedLanguage(input_mkv,baseDir = logsDir, languageRemoved = languageRemoved)

    print("Done!")
    print(f"New file created: {output_mkv}")
    

# ----------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_mkv", help="Path to input MKV file")
    parser.add_argument("--logs", default = os.getcwd(), help = "Specify the directory where the logs are stored for profanity that was filtered")
    parser.add_argument("--out", default="censored_output.mkv", help="Output MKV")
    args = parser.parse_args()

    censor_mkv(args.input_mkv, args.out, args.logs)
    
    
#Cant find the audio file in .mkv after it supposedly uploaded
#Beeping doesnt null out actual audio, just plays ontop of it




#TODO:
# Update code to take positional arguments for beep, silence, or both
# Some kind of print function that translates all of the text and bolds profanity?
# Appears to have delay (see end of video ~ 20:00. or earlier around 7:16)




# import soundfile as sf
# import numpy as np

# def silence_intervals_multichannel(input_wav, output_wav, intervals):
#     # Load audio data and sample rate
#     data, samplerate = sf.read(input_wav)   # data.shape = (samples, channels)

#     # Ensure 2D (mono would be shape (samples,))
#     if data.ndim == 1:
#         data = data[:, None]

#     # Convert time -> sample index
#     for start_sec, end_sec in intervals:
#         start_frame = int(start_sec * samplerate)
#         end_frame = int(end_sec * samplerate)

#         # Silence all channels in this range
#         data[start_frame:end_frame, :] = 0.0

#     # Save back with same channels
#     sf.write(output_wav, data, samplerate, subtype='PCM_16')

# # Example usage:
# intervals = [(12.3, 12.8), (55.1, 56.0)]
# silence_intervals_multichannel("full_audio.wav", "clean_audio.wav", intervals)


#To extract mono version:
# ffmpeg -i full_audio.wav -ac 1 whisper_input.wav

#Auto-extract original
# ffmpeg -i input.mkv -map 0:a:0 -c:a pcm_s16le full_audio.wav

#To remux:
# ffmpeg -i input.mkv -i clean_audio.wav \
#   -map 0:v -map 0:a:0 -map 1:a:0 \
#   -c:v copy \
#   -c:a:0 copy \
#   -c:a:1 flac \
#   output.mkv