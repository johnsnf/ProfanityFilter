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
WHISPER_MODEL = "/root/VideoCleanUp/whisper.cpp/models/ggml-base.bin" # Local Whisper.cpp model
# WHISPER_MODEL = "/root/VideoCleanUp/whisper.cpp/models/ggml-large-v3-turbo.bin"
BEEP_FREQ = 1000                     # 1 kHz beep
BEEP_DB = -3                         # loudness of beep
PROFANITY_LIST = {"fuck", "shit", "bitch", "asshole", "fucker", "motherfucker", "ass", "porn", "hell","fucking","dam"}  # customize
TIME_PADDING = .2 #20%, e.g. tStart - (duration*.2) : tEnd + (duration*.2)
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

        lStart = len(audio)
        # start_ms = int(segment["start"] * 1000)
        # end_ms = int(segment["end"] * 1000)
        
        # start_ms = convertTimeStamp(segment["timestamps"]["from"])
        # end_ms = convertTimeStamp(segment["timestamps"]["to"])
        start_ms = segment["offsets"]["from"]
        end_ms = segment["offsets"]["to"]
        
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




#TODO:
# Update code to take positional arguments for beep, silence, or both
# Some kind of print function that translates all of the text and bolds profanity?
# Appears to have delay (see end of video ~ 20:00. or earlier around 7:16)
# Have it store temporary files elswhere and to update/save actual files on drive
# Add some kind of flag that will stop it from muxing if there are insufficient words detected (e.g. poor/no audio)
# Add a seperate file for all profane language
# Add a counter script/log to document quantity of profane language omitted and perhaps what the words were
# Output profanity counter at the end of each run?
# Issue with sound might be the codec? See reference code below (also seems like surround sound is lost?):

# import subprocess
# import os

# def run_command(command):
#     """Helper function to run shell commands."""
#     try:
#         subprocess.run(command, check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         print(f"Command executed successfully: {command}")
#     except subprocess.CalledProcessError as e:
#         print(f"Error executing command: {command}")
#         print("STDOUT:", e.stdout.decode())
#         print("STDERR:", e.stderr.decode())
#         raise

# def get_audio_codec(input_file):
#     """Uses ffprobe to get the codec name of the first audio stream."""
#     command = f'ffprobe -v error -select_streams a:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 "{input_file}"'
#     try:
#         result = subprocess.check_output(command, shell=True, text=True).strip()
#         return result
#     except subprocess.CalledProcessError:
#         print("Could not determine audio codec using ffprobe.")
#         return None

# def extract_and_merge_audio_lossless(input_mkv, output_mkv, temp_audio_file='temp_audio'):
#     """Extracts audio and merges it back into the MKV container losslessly."""
    
#     # 1. Inspect audio codec to choose correct extension for temp file
#     audio_codec = get_audio_codec(input_mkv)
#     if not audio_codec:
#         print("Aborting.")
#         return

#     # Common extensions for common codecs
#     if audio_codec in ['aac', 'mp3', 'ac3', 'dts', 'flac', 'vorbis', 'opus']:
#         temp_audio_file += f'.{audio_codec}'
#     else:
#         # For unusual codecs, you might need to specify a generic container 
#         # or re-encode, but for simplicity, we'll try a common one or rely on ffmpeg's smart defaults.
#         print(f"Unknown codec {audio_codec}, using .mka extension.")
#         temp_audio_file += '.mka'

#     # 2. Extract audio without re-encoding
#     print(f"Extracting audio to {temp_audio_file} (codec: {audio_codec})...")
#     # -vn means no video, -acodec copy means copy the audio codec as is, -sn means no subtitles
#     run_command(f'ffmpeg -i "{input_mkv}" -vn -acodec copy -sn "{temp_audio_file}"')

#     # 3. Add the extracted audio back into a new MKV file, keeping the original video
#     print(f"Merging audio and video into {output_mkv}...")
#     # -map 0:v:0 maps the first video stream from the first input
#     # -map 1:a:0 maps the first audio stream from the second input
#     # -c copy copies all streams without re-encoding
#     run_command(f'ffmpeg -i "{input_mkv}" -i "{temp_audio_file}" -map 0:v:0 -map 1:a:0 -c copy -shortest "{output_mkv}"')

#     # 4. Clean up the temporary audio file
#     os.remove(temp_audio_file)
#     print("Process complete. Temporary file removed.")

# # Example usage:
# if __name__ == "__main__":
#     input_file = "input.mkv"
#     output_file = "output_with_new_audio.mkv"
#     extract_and_merge_audio_lossless(input_file, output_file)
