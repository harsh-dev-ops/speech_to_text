import json
import os

def ai_algorithms():
    algorithms = {}
    algorithms['speech_to_text'] = json.load(
        open(os.path.join(os.getcwd(), "app/voice/speech_to_text/config.json")))
    algorithms['noise_reduction'] = json.load(
        open(os.path.join(os.getcwd(), "app/voice/noise_reduction/config.json")))

    return algorithms
