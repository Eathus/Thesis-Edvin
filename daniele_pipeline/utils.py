from dotenv import load_dotenv
from openai import OpenAI
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
load_dotenv()
client = OpenAI(api_key=OPENAI_API_KEY)

'''
def send_message(messages, append_response=False, verbose=False):
    f = open('logs/context_last_message.json', 'w')
    f.write(str(messages))
    comp = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages,
        response_format={ "type": "json_object" }
    )

    if append_response:
        messages.append(comp.choices[0].message)

    if verbose:
        print("Reply\n---\n" + comp.choices[0].message.content + '\n---')
    return comp.choices[0].message.content
'''

def load_prompts(directory_path):
    prompts = {}

    # Check if the directory exists
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return prompts

    # Iterate through files in the directory
    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)

        # Check if the item is a file and ends with '.txt'
        if os.path.isfile(filepath) and filename.endswith('.txt'):
            with open(filepath, 'r') as file:
                prompt_text = file.read()
                prompts[filename.split('.')[0]] = prompt_text

    return prompts


def create_message(content, role='user'):
  return {"role": role, "content": content}