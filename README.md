
# LLM QnA API — SmolLM2-135M-Instruct

This project implements a **Question Answering API** powered by the lightweight **SmolLM2-135M-Instruct** model from Hugging Face.  
The system combines a **Python FastAPI backend** for inference with a **Go (Gin)** API layer for concurrency and orchestration.

---

## 📘 Objective

Build an architecture that:
- 🧩 Hosts the **SmolLM2-135M-Instruct** model for inference (Python),
- ⚙️ Exposes a **Go (Gin)** REST API for users,
- 🧠 Supports **single** and **batched concurrent** requests,
- 💡 Demonstrates separation of inference and API layers.

---

## 🏗️ System Architecture

```text
+----------------------------+          +----------------------------------+
|  Go API Server (Gin)       |  <---->  |  Python Model Host (FastAPI)     |
|  - /chat                   |   HTTP   | - /infer                         |
|  - /chat/batched           |  calls   | - SmolLM2-135M-Instruct model    |
|  - Concurrent goroutines   |          | - CPU-only inference(HF Space)   |
+----------------------------+          +----------------------------------+
````

---

## 📁 Directory Structure

```
/
├── hosting/
│   ├── app.py              # FastAPI model host
│   ├── requirements.txt    # Python dependencies
│   └── Dockerfile          # Hugging Face Space build file
│
├── api_server/
│   ├── main.go             # Go API (Gin) for routing and batching
│   └── go.mod              # Go module dependencies
│
└── README.md               # Documentation
```

---

## ⚙️ Components

### 🧩 Model Host — FastAPI (Python)

**Model:** `HuggingFaceTB/SmolLM2-135M-Instruct`
**Framework:** FastAPI
**Environment:** CPU-only Hugging Face Space (Docker backend)

#### 🔹 Endpoints

| Method | Endpoint | Description              |
| :----- | :------- | :----------------------- |
| `GET`  | `/`      | Health check             |
| `POST` | `/infer` | Model inference endpoint |

#### 🔹 Example Request

```bash
curl -X POST "https://trinitysoul-infer-tifin.hf.space/infer" \
  -H "Content-Type: application/json" \
  -d '{"system_prompt":"You are helpful.","user_prompt":"Explain AI."}'
```

#### 🔹 Example Response

```json
{
  "response": "Artificial intelligence is the science of creating machines that can perform reasoning and learning tasks similar to humans."
}
```

#### 🔹 `app.py`

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI, Request
import torch

app = FastAPI()

model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

@app.get("/")
async def hello():
    return {"Hello"}

@app.post("/infer")
async def infer(request: Request):
    data = await request.json()
    system_prompt = data.get("system_prompt", "")
    user_prompt = data.get("user_prompt", "")
    prompt = f"System-Prompt: {system_prompt}\nUser-Prompt: {user_prompt}\nAssistant-Answer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=128)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}
```

---

### ⚙️ Go API Server — Gin Framework

**Purpose:**
Handles client queries, forwards them to the Hugging Face Space `/infer` endpoint, and aggregates responses concurrently.

#### 🔹 Endpoints

| Method | Endpoint        | Description                    |
| :----- | :-------------- | :----------------------------- |
| `POST` | `/chat`         | Single inference request       |
| `POST` | `/chat/batched` | Concurrent multiple inferences |

#### 🔹 Example: Single Query

```bash
curl -X POST "http://localhost:8080/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "chat_id": "1",
    "system_prompt": "You are helpful.",
    "user_prompt": "Explain transformers."
  }'
```

#### 🔹 Example: Batched Queries

```bash
curl -X POST "http://localhost:8080/chat/batched" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      {"chat_id": "1", "system_prompt": "You are helpful.", "user_prompt": "Explain AI."},
      {"chat_id": "2", "system_prompt": "You are helpful.", "user_prompt": "What is overfitting?"}
    ]
  }'
```

#### 🔹 Example Response

```json
{
  "responses": [
    "Artificial intelligence is ...",
    "Overfitting is when a model performs well on training data but poorly on new data."
  ]
}
```

---

## ⚡ Concurrency Implementation

* Each query in `/chat/batched` runs in its **own goroutine**.
* `sync.WaitGroup` ensures safe synchronization.
* Responses are collected and returned as a unified JSON list.

```go
var wg sync.WaitGroup
for _, q := range batch.Queries {
    wg.Add(1)
    go func(query ChatQuery) {
        defer wg.Done()
        // POST to HF /infer
    }(q)
}
wg.Wait()
```

✅ This allows multiple model calls to run **simultaneously**, improving latency without blocking.

---

## 🧠 Design Choices

| Concern       | Decision         | Reason                              |
| :------------ | :--------------- | :---------------------------------- |
| Model Serving | Python (FastAPI) | `transformers` compatibility        |
| API Gateway   | Go (Gin)         | High concurrency + clean routing    |
| Communication | HTTP POST        | Simple, portable                    |
| Concurrency   | Goroutines       | Non-blocking requests               |
| Hardware      | CPU-only         | Lightweight and deployable anywhere |

---

## 🚀 Deployment

### 1️⃣ Deploy on Hugging Face (Model Host)

* Space → **New Space → Docker backend**
* Upload:

  * `app.py`
  * `requirements.txt`
  * `Dockerfile`
* Click **Rebuild Space**
* Once running, note your endpoint URL -> like https://user-name-dash-sperated.hf.space

---

### 2️⃣ Run Go API Server Locally

```bash
cd api_server
go mod tidy
go run main.go
```

---

### 3️⃣ Test Everything

#### Single

```bash
curl -X POST "http://localhost:8080/chat" \
  -H "Content-Type: application/json" \
  -d '{"chat_id":"1","system_prompt":"You are helpful.","user_prompt":"Explain AI."}'
```

#### Batched

```bash
curl -X POST "http://localhost:8080/chat/batched" \
  -H "Content-Type: application/json" \
  -d '{"queries":[{"chat_id":"1","system_prompt":"You are helpful.","user_prompt":"Explain AI."},{"chat_id":"2","system_prompt":"You are helpful.","user_prompt":"What is overfitting?"}]}'
```

---

## 🧾 Evaluation Summary

| Criteria            | Status                  |
| :------------------ | :---------------------- |
| ✅ Model Integration | SmolLM2-135M-Instruct   |
| ✅ API Gateway       | Go (Gin)                |
| ✅ Concurrency       | Goroutines + WaitGroup  |
| ✅ Error Handling    | Graceful JSON responses |
| ✅ Separation        | Python ↔ Go layers      |
| ✅ CPU Compatible    | No GPU required         |
| 🧩 Bonus            | Supports batch mode      |

---



## 🐳 Dockerfile (for Hugging Face Space)

```dockerfile
FROM python:3.10-slim

WORKDIR /home/user/app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py app.py

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
```

---

## 📦 requirements.txt

```
transformers==4.43.3
torch
fastapi
uvicorn
```

---
