from fastapi import FastAPI, Request
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import uvicorn
import os

app = FastAPI()

model_name = "microsoft/phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

@app.get("/")
async def root():
    return {"message": "Welcome to Phi-3 Mini API!"}

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"response": generated_text}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
