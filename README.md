# Introduction
Flask APIs to generate speech to text. 

## Demo:
<video src='demo/demo.mov' width=1920/> </video>

## Setup:
OS:
Linux/MacOS

Python:
python=3.11

Miniconda:
```bash
conda create -n speech2text python=3.11 -y

conda activate speech2text

pip install -r requirements.txt
```

Venv:
```bash
python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt
```

## Start:
```bash
python app/main.py
```


## Routes:

### Get all Algorithms
```py
import requests
import json

url = "http://localhost:8000/"

payload = json.dumps({
  "audio": "some str"
})
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("GET", url, headers=headers, data=payload)

print(response.text)
```


### File to Base64 String
Code:
```py
import requests

url = "http://localhost:8000/fileTobase64"

payload = {}


filename = 'harvard.wav'
audio_file_path = "upload/harvard.wav"
content_type = 'audio/wav' # or 'audio/mpeg'

files=[
  ('file',(filename, open(audio_file_path,'rb'), content_type))
]

response = requests.request("POST", url, headers=headers, data=payload, files=files)

print(response.text)

````

### Upload File:

Code: 
```py
import requests

url = "http://localhost:8000/media"

payload = {}

filename = 'harvard.wav'
audio_file_path = "upload/harvard.wav"
content_type = 'audio/wav' # or 'audio/mpeg'

files=[
  ('file',(filename, open(audio_file_path,'rb'), content_type))
]

headers = {
}

response = requests.request("POST", url, headers=headers, data=payload, files=files)

print(response.text)

```

### Speech To Text - Media File
Code:
```py
import requests

url = "http://localhost:8000/speechToText/media"

payload = {'speechToText': '2',
'enableNoiseReduction': 'true',
'noiseReducer': '1'}


filename = 'harvard.wav'
audio_file_path = "upload/harvard.wav"
content_type = 'audio/wav' # or 'audio/mpeg'

files=[
  ('file',(filename, open(audio_file_path,'rb'), content_type))
]
headers = {
}

response = requests.request("POST", url, headers=headers, data=payload, files=files)

print(response.text)
```

### Speech To Text - Base64 String
Code:
```py
import requests
import json

url = "http://localhost:8000/speechToText/base64"

payload = json.dumps({
  "base64String": "",
  "speechToText": 2, # 1 to 4
  "enableNoiseReduction": True,
  "noiseReducer": 1
})
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)


```

### Postman Documentation:
Link: https://documenter.getpostman.com/view/15907649/2sAXxTapo7

## Conclusion
- **Worked well with:**
    - **Speech to Text:**
        - OpenAI Whisper: 2
        - Wav2Vec2: 3
        - SpeechT5: 4

    - **Noise Reduction:**
        - Noise Reduction: 1

- **Failed Modules:**
    - **Speech to Text:** SpeechBrain
    - **Noise Reduction:** SpeechBrain

- **Not working properly:**
    - **Speech to Text:** SpeechRecognition: 1

## Reason for Failure
- SpeechBrain doesn't work with `numpy.ndarray`. The file needs to be saved in `.wav` format and then loaded into SpeechBrain pipelines.

## Attempts to Fix the Problem
- Created custom classes for SpeechBrain using inheritance to support `ndarray`.
- Converted `ndarray` to tensors, but SpeechBrain still requires `.wav` format.

## Future Scope
- Using **GPU acceleration**, transcription can be done in near real-time.
- A client (frontend) can be built using **React** or **HTML + JavaScript** that allows users to upload audio files or capture audio from their microphone for real-time speech-to-text processing.
- For **microphone input**, the client-server can convert audio segments into base64 and use them.
- For **file uploads**, the client-server can directly use the API via form data.
- A **singleton with a metaclass** can be used to manage the four models currently in use, which will help keep the models loaded in memory as long as the REST server is running.
- **Sub-configurations** for models like Whisper, SpeechT5, and Wav2Vec2 can be created to support more sub-models (e.g., tiny, medium, large variants). An **abstract factory with a prototype design pattern** can be used for this purpose.



