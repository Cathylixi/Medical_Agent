import os
from dotenv import load_dotenv
from openai import OpenAI
import base64

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# The path to your image
image_path = os.path.join(os.path.dirname(__file__), '../data/input_2.jpg')

# Read and encode the image in base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Encode the image
base64_image = encode_image_to_base64(image_path)

# The text input you want to provide along with the image
text_input = "请对这个图片做OCR并返回结果"

# API call to GPT-4 Vision with both image and text
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text_input
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ],
    max_tokens=1024
)

# Print the response
print(response.choices[0].message.content)
