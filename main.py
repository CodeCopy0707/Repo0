from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import uvicorn

app = FastAPI()

model_name = "microsoft/phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to("cpu")

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
    <head><title>Phi-3 Mini Demo</title></head>
    <body style='font-family: Arial, sans-serif; padding: 20px;'>
      <h1>Ask something to Phi-3 Mini!</h1>
      <form action='/ask' method='post'>
        <input name='prompt' type='text' style='width: 300px;' required/>
        <button type='submit'>Ask</button>
      </form>
    </body>
    </html>
    """

@app.post("/ask", response_class=HTMLResponse)
async def ask(prompt: str = Form(...)):
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    outputs = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return f"""
    <html>
    <head><title>Response</title></head>
    <body style='font-family: Arial, sans-serif; padding: 20px;'>
      <h2>Prompt:</h2><p>{prompt}</p>
      <h2>Response:</h2><p>{response}</p>
      <a href='/'>Ask another</a>
    </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
