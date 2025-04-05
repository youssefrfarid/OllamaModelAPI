# Example: reuse your existing OpenAI setup
from openai import OpenAI
import base64
from pathlib import Path

# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Function to encode the image


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Path to your image
image_path = "street.jpg"

# Getting the Base64 string
base64_image = encode_image(image_path)

stream = client.chat.completions.create(
    model="minicpm-o-2_6",
    messages=[
        # {"role": "system", "content": "You are an assistant who perfectly describes images for visually impaired people.The image is from the person's POV so keep it as a paragraph to tell the person"},
        {"role": "system", "content": "You are an assistant who answers questions about images for visually impaired people. The image is from the person's POV so keep it as a paragraph to tell the person"},
        {
            "role": "user",
            "content": [
                # {"type": "text", "text": "Give me a description of this image for a blind person include positioning any danger any signs"},
                {"type": "text", "text": "Is there a food place near me? if so where is it?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        },
    ],
    temperature=0.2,
    stream=True
)


for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
