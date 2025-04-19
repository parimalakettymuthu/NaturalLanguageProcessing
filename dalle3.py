from openai import OpenAI
API_KEY = open("API_KEY","r").read()
client = OpenAI(api_key=API_KEY)

response = client.images.generate(
    model='dall-e-3',
    prompt="Five yellow dogs playing ball in the rain",
    n=1,
    size="1024x1024"
)
image_url = response.data[0].url
print(image_url)