from openai import OpenAI
API_KEY = open("API_KEY", "r").read()
client = OpenAI(api_key=API_KEY)

audio_file = open("output.mp3","rb")
transcript = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file,
    response_format="text"
)
print(transcript)