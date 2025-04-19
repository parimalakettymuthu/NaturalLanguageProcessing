import google.generativeai as palm 

API_KEY = 'AIzaSyB6jpTULa6ALZ5hCo6_bYxx-mJ7K3wzeUg'
palm.configure(api_key=API_KEY)

examples = [
    ('Hello', 'Hi there mr. How can I be assistant'),
    ('I want to make a lot of money', 'You should work hard like your parents')
]

prompt = "I need help with a job interview for a data analyst job. Can you help me?"
response = palm.chat(messages=prompt, temperature=0.2, context="Speak like a CEO", examples=examples)
for message in response.messages:
    print(message['author'],message['content'])

while True:
    s = input()
    response = response.reply(s)
    print(response.last)

