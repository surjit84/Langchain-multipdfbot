# strng = "Human: What is my name? AI: ABCD Human: What is name of my father? AI: ABCD"
# strng1 = strng.replace("AI:",",AI:").replace("Human:",",Human:").split(",")
# strng1 = strng1[1:]
# for i, message in enumerate(strng1):
#     if i % 2 == 0:
#       print(message)
#     else:
#       print(message)

import google.generativeai as genai
from dotenv import load_dotenv
import os
load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

for m in genai.list_models():
  if 'embedContent' in m.supported_generation_methods:
    print(m.name)