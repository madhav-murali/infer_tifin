from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI, Request
import torch
import uvicorn

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
    prompt = f"System-Prompt: {system_prompt}\nUser-Prompt: {user_prompt}\n Assistant-Answer: "
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=128)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
