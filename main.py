import torch
from transformers import pipeline
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn
import os

# specify chat server
HOST = 'localhost'
PORT = 8001
URL = f'http://{HOST}:{PORT}'

model_name = "databricks/dolly-v2-12b"

current_path = os.path.dirname(os.path.abspath(__file__))
instruct_pipeline = pipeline(model=model_name,
                             torch_dtype=torch.bfloat16,
                             trust_remote_code=True,
                             device_map="auto")


app = FastAPI()


# Remove CORS restrictions (if needed)
# from fastapi.middleware.cors import CORSMiddleware
# origins = [
#     f'http://{HOST}',
# ]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


@app.get("/chat_api")
async def chat(text: str = ""):
    reply = instruct_pipeline(text).replace('\n', '<br>')
    print(f'input:{text} reply:{reply}')

    outJson = {
        "output": [
            {
                "type": "text",
                "value": reply
            }
        ]
    }
    return outJson


app.mount("/", StaticFiles(directory="html", html=True), name="html")


def start_server():
    uvicorn.run(app, host=HOST, port=PORT)


def main():
    start_server()

    # When you want to open a browser at the same time
    # Use thread(if needed)
    # import threading
    # import webbrowser
    # threading.Thread(target=start_server).start()
    # webbrowser.open(URL, new=0, autoraise=True)


if __name__ == "__main__":
    main()




