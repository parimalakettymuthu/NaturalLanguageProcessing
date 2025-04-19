from openai import OpenAI

API_KEY = open("API_KEY", "r").read()

client = OpenAI(api_key=API_KEY)
response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input="Hello world! This is a streaming test."
)

response.stream_to_file("output.mp3")