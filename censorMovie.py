import subprocess
import json
import os
from pydub import AudioSegment
from pydub.generators import Sine

# /root/VideoCleanUp/whisper.cpp/models/ggml-base.bin
# ./build/bin/whisper-cli -m models/ggml-base.bin -f samples/jfk.wav --max-len 1 -of "test2" -oj
# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------
WHISPER_CPP_PATH = "/root/VideoCleanUp/whisper.cpp/build/bin/whisper-cli"          # Path to whisper.cpp executable
# WHISPER_MODEL = "/root/VideoCleanUp/whisper.cpp/models/ggml-base.bin" # Local Whisper.cpp model
WHISPER_MODEL = "/root/VideoCleanUp/whisper.cpp/models/ggml-large-v3-turbo.bin"
BEEP_FREQ = 1000                     # 1 kHz beep
BEEP_DB = -3                         # loudness of beep
PROFANITY_LIST = {"fuck", "shit", "bitch", "asshole", "fucker", "motherfucker", "ass", "porn", "hell","fucking"}  # customize
# ----------------------------------------------------------

def convertTimeStamp(timeString):
    d = timeString.split(",")
    d[0] = d[0].split(":") 
    time = int(d[1]) + (int(d[0][0])*60*60*1000) + (int(d[0][1])*60*1000) + (int(d[0][2])*1000)
    return time 

def run_whisper(audio_path, output_json="transcript.json"):
    """
    Runs whisper.cpp on the extracted WAV file and returns parsed JSON.
    """
    cmd = [
        WHISPER_CPP_PATH,
        "-m", WHISPER_MODEL,
        "-f", audio_path,
        "--max-len", "1",
        "-of", output_json.replace(".json", ""),
        "-oj"  # output JSON
    ]
    print("Running whisper.cpp...")
    subprocess.run(cmd, check=True)
    with open(output_json, "r") as f:
        return json.load(f)

def extract_audio(mkv_path, wav_path):
    """
    Extracts audio from MKV to WAV using ffmpeg.
    """
    cmd = ["ffmpeg", "-y", "-i", mkv_path, "-vn", "-ac", "1", "-ar", "16000", wav_path]
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
        "-map", "0:a",          # keep original audio
        "-map", "1:a",          # add censored audio
        "-c:v", "copy",
        "-c:a", "aac",
        "-metadata:s:a:1", "title=Censored Audio",
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

    for segment in transcript["transcription"]:
        text = segment["text"].lower().replace(" ","")
        contains_profanity = any(bad in text for bad in PROFANITY_LIST)
        if not contains_profanity:
            continue

        # start_ms = int(segment["start"] * 1000)
        # end_ms = int(segment["end"] * 1000)
        
        # start_ms = convertTimeStamp(segment["timestamps"]["from"])
        # end_ms = convertTimeStamp(segment["timestamps"]["to"])
        start_ms = segment["offsets"]["from"]
        end_ms = segment["offsets"]["to"]
        
        duration = end_ms - start_ms
        # silent_segment = AudioSegment.silent(duration=silence_duration_ms, frame_rate=audio.frame_rate) #Creates a silent portion for the frames

        beep = make_beep(duration) #Audio to replace portion with
        censored = censored[:start_ms] + beep + censored[end_ms:] #Censoring exact time frame
        # censored = censored.overlay(beep, position=start_ms)

    # Save censored track
    censored_path = original_wav.replace(".wav", "_censored.wav")
    censored.export(censored_path, format="wav")
    return censored_path

# ----------------------------------------------------------
# MAIN WORKFLOW
# ----------------------------------------------------------
def censor_mkv(input_mkv, output_mkv="output_censored.mkv"):
    temp_wav = "temp_audio.wav"

    print("[1] Extracting audio...")
    extract_audio(input_mkv, temp_wav)

    print("[2] Running Whisper.cpp for transcription...")
    transcript = run_whisper(temp_wav)

    print("[3] Creating censored audio track...")
    censored_wav = censor_audio(temp_wav, transcript)

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
