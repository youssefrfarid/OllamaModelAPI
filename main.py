from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from ollama import chat

app = FastAPI()

# In-memory store for conversation contexts keyed by a session_id.
conversations = {}


def save_file(upload_file: UploadFile, destination: str) -> str:
    """
    Saves an uploaded file to a local destination.
    """
    with open(destination, "wb") as f:
        f.write(upload_file.file.read())
    return destination


@app.post("/chat")
async def chat_endpoint(
    message: str = Form(...),
    image: UploadFile = File(...)
):
    """
    Endpoint to start a new conversation or continue one by providing
    an image and an initial message. The conversation context is saved
    under the given session_id.
    """
    # Save the uploaded image file temporarily.
    image_path = f"temp_{image.filename}"
    save_file(image, image_path)

    messages = [{
        "role": "user",
        "content": message,
        "images": [image_path]
    }]

    response_text = ""
    # Get streaming response from Ollama.
    stream = chat(model='minicpm-v', messages=messages, stream=True)
    for chunk in stream:
        response_text += chunk['message']['content']

    return {"response": response_text}


@app.post("/continue")
async def continue_endpoint(
    image: UploadFile = File(...)
):
    """
    Endpoint to continue an existing conversation using a new image and prompt.
    The existing conversation context is retrieved using the session_id.
    """
    # Save the uploaded image file temporarily.
    image_path = f"temp_{image.filename}"
    save_file(image, image_path)

    # Build prompt content with the new prompt.
    content = "You should guide visually impaired people. Keep the scene description short and to the point include any signs or any obstacles in the scene. Describe the positoining of objects in the scene. The image is from the persons pov keep it as short as possible"

    messages = [{
        "role": "user",
        "content": content,
        "images": [image_path]
    }]

    response_text = ""
    # Stream the response from Ollama.
    stream = chat(model='minicpm-v', messages=messages, stream=True)
    for chunk in stream:
        response_text += chunk['message']['content']

    return {"response": response_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
