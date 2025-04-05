import subprocess
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import os
import time
import uvicorn

app = FastAPI()

# Absolute paths to your model files and executable.
EXECUTABLE = "/Users/yousseframy/ChinaModel/llama.cpp/build/bin/llama-minicpmv-cli"
MODEL_PATH = "/Users/yousseframy/ChinaModel/Model-7.6B-Q4_K_M.gguf"
MMPROJ_PATH = "/Users/yousseframy/ChinaModel/mmproj-model-f16.gguf"


def run_cli_command(command, end_marker="llama_perf_context_print:", timeout=60):
    """
    Launches the CLI command as a subprocess and reads its stdout until
    a line containing the end_marker is found or until timeout.
    """
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1  # line-buffered
    )
    output = ""
    start_time = time.time()
    while True:
        line = process.stdout.readline()
        if line:
            output += line
            # Check if this line contains the end marker.
            if end_marker in line:
                break
        else:
            if process.poll() is not None:
                break
        if time.time() - start_time > timeout:
            break
    process.stdout.close()
    process.wait()
    return output


def extract_answer(output):
    """
    Extracts and returns the text that comes after <assistant> and before the
    'llama_perf_context_print:' marker.
    """
    idx_start = output.find("<assistant>")
    idx_end = output.find("llama_perf_context_print:")
    if idx_start != -1 and idx_end != -1:
        # Return text after <assistant> up to the end marker.
        return output[idx_start + len("<assistant>"):idx_end].strip()
    elif idx_start != -1:
        return output[idx_start + len("<assistant>"):].strip()
    else:
        return output.strip()


@app.post("/predict")
async def predict(
    prompt: str = Form(...),
    image: UploadFile = File(None)
):
    # Save the image temporarily if one is provided.
    image_path = None
    if image:
        try:
            contents = await image.read()
            # Consider generating a unique filename if needed.
            image_path = "temp_image.jpg"
            with open(image_path, "wb") as f:
                f.write(contents)
        except Exception as e:
            raise HTTPException(
                status_code=400, detail="Error reading image file.")

    # Build the CLI command with all arguments.
    command = [
        EXECUTABLE,
        "-m", MODEL_PATH,
        "--mmproj", MMPROJ_PATH,
        "-c", "4096",           # Context size.
        "--temp", "0.3",        # Temperature.
        "--top-p", "0.8",       # Nucleus sampling parameter.
        "--top-k", "100",       # Top-k sampling.
        "--repeat-penalty", "1.05"  # Repeat penalty.
    ]

    if image_path:
        command.extend(["--image", image_path])
    command.extend(["-p", prompt])

    try:
        # Run the CLI command and capture the full output until the end marker.
        output = run_cli_command(
            command, end_marker="llama_perf_context_print:", timeout=120)
        # Extract only the answer part (after <assistant> and before the performance log).
        answer = extract_answer(output)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Inference failed: {str(e)}")
    finally:
        # Clean up the temporary image file.
        if image_path and os.path.exists(image_path):
            os.remove(image_path)

    return JSONResponse({"output": answer})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
