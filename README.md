# Trying out dolly-v2

#### Installation and execution of dolly-v2

![img.png](img.png)

# Experiment environment

- OS: **Ubuntu 22.04** on **AWS**
  - g4dn.12xlarge
- GPU: **Tesla T4 x 4 (Memory 16GB x 4 = 64GB)**
- Python: **Anaconda3 environment**

## STEP 1: Create an Anaconda virtual environment

Create an Anaconda virtual environment named **env-dolly-v2** and install python 3.10.10 to try dolly-v2

**Update conda to the latest version**

```commandline
conda update -n base -c defaults conda --yes
```


**Create the env-dolly-v2 virtual environment and install python**

```commandline
conda create --yes -n env-dolly-v2
conda activate env-dolly-v2
conda install python=3.10.10 --yes
```


## STEP 2: Install required packages

Install the required packages as follows:

```commandline
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install accelerate
pip install transformers
pip install fastapi uvicorn
```


## STEP 3: Write the source code

**Write the chat server**

```python
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
   
```

**Write the chat client**

Create a directory named /html and place the following index.html file in it:


```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
    <title>Chat </title>
</head>
<body>
<script src="https://riversun.github.io/chatux/chatux.min.js"></script>
<script>
    const chatux = new ChatUx();

    // initializing param for chatux
 const initParam =
        {
            renderMode: 'auto',
            api: {
                //echo chat server
                endpoint:'/chat_api',
                method: 'GET',
                dataType: 'json'
            },
            bot: {
                botPhoto: 'https://riversun.github.io/chatbot

```


## STEP 4: Start Chat Server

```
python main.py
```

And open url,
http://localhost:8001