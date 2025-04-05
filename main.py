from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from ollama import chat

app = FastAPI()


@app.post("/chat")
async def chat_endpoint(
    message: str = Form(...),
    image: UploadFile = File(...)
):
    """
    Endpoint to start a conversation.
    Expects a 'message' field and an image file in the form data.
    The image is read in memory and its raw bytes are passed to the Ollama API.
    The response is streamed back to the client as text.
    """
    # Read image file in memory as raw bytes.
    try:
        image_bytes = await image.read()
    except Exception as e:
        raise HTTPException(
            status_code=400, detail="Error reading image file.")

    messages = [{
        "role": "user",
        "content": message,
        "images": [image_bytes]
    }]

    # Generator that streams chunks from the Ollama API.
    def generate():
        stream = chat(model='minicpm-v', messages=messages, stream=True)
        for chunk in stream:
            yield chunk['message']['content']

    return StreamingResponse(generate(), media_type="text/plain")


@app.post("/continue")
async def continue_endpoint(
    image: UploadFile = File(...)
):
    """
    Endpoint to continue an existing conversation.
    Expects an image file in the form data.
    The image is read in memory and its raw bytes are passed to the Ollama API with a predefined prompt.
    The response is streamed back to the client as text.
    """
    try:
        image_bytes = await image.read()
    except Exception as e:
        raise HTTPException(
            status_code=400, detail="Error reading image file.")

    # Predefined prompt to guide visually impaired people.
    content = (
        "You should guide visually impaired people. Keep the scene description short and to the point, "
        "including any signs or obstacles. Describe the positioning of objects in the scene. "
        "The image is from the person's POV so keep it as short as possible."
    )

    messages = [{
        "role": "user",
        "content": content,
        "images": [image_bytes]
    }]

    def generate():
        stream = chat(model='minicpm-v', messages=messages, stream=True)
        for chunk in stream:
            yield chunk['message']['content']

    return StreamingResponse(generate(), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
