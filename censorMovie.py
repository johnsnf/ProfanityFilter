import subprocess
import json
import os
from pydub import AudioSegment
from pydub.generators import Sine
import whisperx
import soundfile as sf 
import numpy as np 

#TODO:
# - Setup to run off of GPU
# - Setup a flag to indicate whether transcript should be written or not
# - Write a wrapper script that calls on this script to batch run 
# - Update print statements
# - Run off of soundfile to preserve the audio quality 
# - Setup catches so that it stops running if english keywords are not picked up
# - Error file that appends to include files that failed
# - A file that indicates all of the words omitted, where they were, total removed (json file)
# - Set it up so that it doesnt remove words that are far from the curse word itself (e.g. embarrising?)
# - MUCH LATER: have a flag to indicate not to take the first audio file

# /root/VideoCleanUp/whisper.cpp/models/ggml-base.bin
# ./build/bin/whisper-cli -m models/ggml-base.bin -f samples/jfk.wav --max-len 1 -of "test2" -oj
# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------
WHISPER_CPP_PATH = "/root/VideoCleanUp/whisper.cpp/build/bin/whisper-cli"          # Path to whisper.cpp executable
WHISPER_MODEL = "/root/VideoCleanUp/whisper.cpp/models/ggml-base.bin" # Local Whisper.cpp model
# WHISPER_MODEL = "/root/VideoCleanUp/whisper.cpp/models/ggml-large-v3-turbo.bin"
BEEP_FREQ = 1000                     # 1 kHz beep
BEEP_DB = -3                         # loudness of beep
PROFANITY_LIST = {"fuck", "shit", "bitch", "asshole", "fucker", "motherfucker", "ass", "porn", "hell","fucking","dam","twat","dick","cunt"}  # customize
TIME_PADDING = .2 #20%, e.g. tStart - (duration*.2) : tEnd + (duration*.2)
# ----------------------------------------------------------

def convertTimeStamp(timeString):
    d = timeString.split(",")
    d[0] = d[0].split(":") 
    time = int(d[1]) + (int(d[0][0])*60*60*1000) + (int(d[0][1])*60*1000) + (int(d[0][2])*1000)
    return time 

# def run_whisper(audio_path, output_json="transcript.json"):
#     """
#     Runs whisper.cpp on the extracted WAV file and returns parsed JSON.
#     """
#     cmd = [
#         WHISPER_CPP_PATH,
#         "-m", WHISPER_MODEL,
#         "-f", audio_path,
#         "--max-len", "1",
#         "-of", output_json.replace(".json", ""),
#         "-oj"  # output JSON
#     ]
#     print("Running whisper.cpp...")
#     subprocess.run(cmd, check=True)
#     with open(output_json, "r") as f:
#         return json.load(f)
    
def runWhisperX(audio_path, output_json = "transcript.json"):
    model = whisperx.load_model("base",device="cpu", compute_type = "int8")
    model_a, metadata = whisperx.load_align_model(language_code="en",device="cpu")
    audio = whisperx.load_audio(audio_path)
    
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
    # cmd = ["ffmpeg", "-y", "-i", mkv_path, "-vn", "-ac", "1", "-ar", "16000", wav_path + "_mono.wav"]
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
    # cmd = [
    #     "ffmpeg",
    #     "-y",
    #     "-i", original_mkv,
    #     "-i", censored_wav,
    #     "-map", "0:v",          # keep original video
    #     "-map", "0:a",          # keep original audio
    #     "-map", "1:a",          # add censored audio
    #     "-c:v", "copy",
    #     "-c:a", "aac",
    #     "-metadata:s:a:1", "title=Censored Audio",
    #     output_mkv
    # ]
    
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

def make_beep(duration_ms):
    return Sine(BEEP_FREQ).to_audio_segment(duration=duration_ms).apply_gain(BEEP_DB)

#transcription[0]["text"] - text
#transcription[0]["timestamps"]["from"] - start time
#transcription[0]["timestamps"]["to"] - end time


def censor_audio(original_wav, transcript):
    audio = AudioSegment.from_wav(original_wav)
    censored = audio[:]  # copy

    for segment in transcript["word_segments"]:
        text = segment["word"].lower().replace(" ","")
        contains_profanity = any(bad in text for bad in PROFANITY_LIST)
        if not contains_profanity:
            continue
        
        start_ms = segment["start"] * 1000 #Converting from sec to miliseconds
        end_ms = segment["end"] * 1000 #Converting from sec to miliseconds
        
        duration = end_ms - start_ms
        
        #Padding around word to ensure it isn't missed
        start_ms -= TIME_PADDING*duration 
        end_ms += TIME_PADDING*duration 
        durationNew = end_ms - start_ms 
        
        silent_segment = AudioSegment.silent(duration=durationNew, frame_rate=audio.frame_rate) #Creates a silent portion for the frames

        # beep = make_beep(duration) #Audio to replace portion with
        censored = censored[:start_ms] + silent_segment + censored[end_ms:] #Censoring exact time frame
        # censored = censored.overlay(beep, position=start_ms)

    # Save censored track
    censored_path = original_wav.replace(".wav", "_censored.wav")
    censored.set_frame_rate(audio.frame_rate)
    censored.export(censored_path, format="wav")
    return censored_path


def censor_audio2(original_wav, transcript):
    output_wav = original_wav.replace(".wav", "_censored.wav")
    original_wav = original_wav.replace(".wav","") + "_full.wav"
    data, samplerate = sf.read(original_wav)   # data.shape = (samples, channels)
    
    if data.ndim == 1:
        data = data[:, None]
        
    for segment in transcript["word_segments"]:
        text = segment["word"].lower().replace(" ","")
        contains_profanity = any(bad in text for bad in PROFANITY_LIST)
        if not contains_profanity:
            continue
        
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
    
    return output_wav

# ----------------------------------------------------------
# MAIN WORKFLOW
# ----------------------------------------------------------
def censor_mkv(input_mkv, output_mkv="output_censored.mkv"):
    temp_wav = "temp_audio.wav"

    print("[1] Extracting audio...")
    extract_audio(input_mkv, temp_wav)

    print("[2] Running Whisper.cpp for transcription...")
    transcript, _ = runWhisperX(temp_wav)

    print("[3] Creating censored audio track...")
    censored_wav = censor_audio2(temp_wav, transcript)

    print("[4] Muxing new MKV with additional censored audio track...")
    mux_new_audio(input_mkv, censored_wav, output_mkv)

    print("Done!")
    print(f"New file created: {output_mkv}")

# ----------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_mkv", help="Path to input MKV file")
    parser.add_argument("--out", default="censored_output.mkv", help="Output MKV")
    args = parser.parse_args()

    censor_mkv(args.input_mkv, args.out)
    
    
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