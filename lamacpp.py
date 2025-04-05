from llama_cpp import Llama
from llama_cpp.llama_chat_format import MiniCPMv26ChatHandler
import base64

modelPath = "/Users/yousseframy/ChinaModel/Model-7.6B-Q4_K_M.gguf"
clipPath = "/Users/yousseframy/ChinaModel/mmproj-model-f16.gguf"

chat_handler = MiniCPMv26ChatHandler(clip_model_path=clipPath)
llm = Llama(
    model_path=modelPath,
    chat_handler=chat_handler,
    n_ctx=2048,  # n_ctx should be increased to accommodate the image embedding
)


def image_to_base64_data_uri(file_path):
    with open(file_path, "rb") as img_file:
        base64_data = base64.b64encode(img_file.read()).decode('utf-8')
        return f"data:image/png;base64,{base64_data}"


# Replace 'file_path.png' with the actual path to your PNG file
file_path = 'street.jpg'
data_uri = image_to_base64_data_uri(file_path)

response = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are an assistant who perfectly describes images."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image for a blind person include positioning any danger any signs"},
                {"type": "image_url", "image_url": {
                    "url": data_uri}}
            ]
        }
    ]
)

print(response["choices"][0]["message"]["content"])
