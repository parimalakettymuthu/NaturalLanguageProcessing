import google.generativeai as palm

API_KEY = 'AIzaSyB6jpTULa6ALZ5hCo6_bYxx-mJ7K3wzeUg'
palm.configure(api_key=API_KEY)

model_list = [_ for _ in palm.list_models()]
for model in model_list:
    print(model.name)

# Example 1. Text Generation
model_id = 'models/text-bison-001'
prompt = '''
write a marketing proposal to sell an AI product. limit the proposal to 100 words.
'''
completion = palm.generate_text(
    model=model_id,
    prompt=prompt,
    temperature=0.99,
    max_output_tokens=800,
    candidate_count=2
)

# Access the values within the same session
#print(completion.result)
#print(completion.result)
outputs = [output['output'] for output in completion.candidates]
for output in outputs:
    print(output)
    print('-'*50)

