import json 

PROFANITY_LIST = {"fuck", "shit", "bitch", "asshole", "fucker", "motherfucker", "ass", "porn", "hell","fucking"}  # customize

def loadData(fileName):
    with open(fileName,'r') as f:
        data = json.load(f)
    return data["transcription"]

def languageTimes(data):
    language = {}
    for word in data:
        text = word["text"].lower().replace(" ","")
        if any(bad in text for bad in PROFANITY_LIST):
            print(f'{text:14}: [{word["timestamps"]["from"]} -> {word["timestamps"]["to"]}]')
            if not text in language.keys():
                language[text] = 1 
            else:
                language[text] += 1
    
    print('-'*46)
    print('\n')
    print(f'Word          : Count')
    totalCount = 0
    for key in language.keys():
        print(f'{key:14}: {language[key]:5}')
        totalCount += language[key]
    print('-'*21)
    print(f'Words blocked: {totalCount:4}')
    


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("transcriptJSON", help="Path to input MKV file")
    # parser.add_argument("--out", default="censored_output.mkv", help="Output MKV")
    args = parser.parse_args()

    data = loadData(args.transcriptJSON) 
    languageTimes(data)   
    