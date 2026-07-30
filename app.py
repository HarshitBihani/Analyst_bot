import os
import json
import time
import threading
import requests
from fastapi import FastAPI
from fastapi.responses import FileResponse
from openai import OpenAI
import sys
import io

# Environment Variables
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")
LOG_FILE = "run.jsonl"

app = FastAPI()

# Initialize the client using Gemini's OpenAI compatibility endpoint
client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Helper: Log to JSONL
def log_run(data):
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(data) + "\n")

# Tool: Execute Python Code
def run_python(code: str) -> str:
    old_stdout = sys.stdout
    redirected_output = sys.stdout = io.StringIO()
    try:
        exec(code, {})
        output = redirected_output.getvalue()
    except Exception as e:
        output = str(e)
    finally:
        sys.stdout = old_stdout
    return output[:8000] # Cap output to prevent massive strings

# LLM Agent Logic
def process_message(chat_id, text):
    system_prompt = f"""
    You are a data analyst. You answer questions by writing and executing Python code.
    Use the 'run_python' tool to fetch data and compute answers.
    Your final response MUST be exactly one JSON object and nothing else.
    Do not use markdown formatting or fences.
    Include a placeholder for log_url. Example: {{"answer": {{"state": "Assam"}}, "log_url": "URL"}}
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ]

    for _ in range(5):
        # 12-second delay to stay under the free tier Rate Limit (5 RPM)
        time.sleep(12) 
        
        response = client.chat.completions.create(
            model="gemini-3.5-flash", 
            messages=messages,
            tools=[{
                "type": "function",
                "function": {
                    "name": "run_python",
                    "description": "Execute python code server-side to analyze data",
                    "parameters": {
                        "type": "object",
                        "properties": {"code": {"type": "string"}},
                        "required": ["code"]
                    }
                }
            }]
        )
        
        msg = response.choices[0].message
        messages.append(msg)
        
        if msg.tool_calls:
            for tool_call in msg.tool_calls:
                args = json.loads(tool_call.function.arguments)
                result = run_python(args["code"])
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
        else:
            break
            
    # Safely parse the final text and handle NoneTypes
    try:
        if msg and msg.content:
            final_text = msg.content.strip()
            final_text = final_text.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(final_text)
            if "answer" not in parsed:
                parsed = {"answer": parsed}
            parsed["log_url"] = f"{BASE_URL}/run.jsonl"
            final_reply = json.dumps(parsed)
        else:
            final_reply = json.dumps({"answer": "Error: Empty response", "log_url": f"{BASE_URL}/run.jsonl"})
    except:
        final_reply = json.dumps({"answer": "error parsing", "log_url": f"{BASE_URL}/run.jsonl"})
        
    log_run({"chat_id": chat_id, "query": text, "reply": final_reply})
    
    requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                  json={"chat_id": chat_id, "text": final_reply})

# Telegram Long Polling Loop
def poll_telegram():
    offset = None
    while True:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
            params = {"timeout": 30, "offset": offset}
            resp = requests.get(url, params=params).json()
            
            for result in resp.get("result", []):
                offset = result["update_id"] + 1
                if "message" in result and "text" in result["message"]:
                    chat_id = result["message"]["chat"]["id"]
                    text = result["message"]["text"]
                    threading.Thread(target=process_message, args=(chat_id, text)).start()
        except Exception as e:
            time.sleep(5)

# FastAPI Endpoints
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/run.jsonl")
def get_logs():
    if not os.path.exists(LOG_FILE):
        return FileResponse(io.BytesIO(b""), media_type="application/jsonl")
    return FileResponse(LOG_FILE)

# Start Polling on Boot
@app.on_event("startup")
def startup_event():
    threading.Thread(target=poll_telegram, daemon=True).start()
