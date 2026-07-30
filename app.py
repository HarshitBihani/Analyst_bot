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

client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Global lock so only ONE Gemini request goes out at a time,
# no matter how many Telegram messages arrive concurrently.
gemini_lock = threading.Lock()

TOOLS = [{
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

def log_run(data):
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(data, default=str) + "\n")
    except Exception:
        pass

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
    return output[:8000]

def call_gemini_with_retry(messages, attempts=4):
    """Serialized + retried Gemini call. Returns response or raises last error."""
    last_err = None
    with gemini_lock:
        time.sleep(5)  # 15 RPM headroom on gemini-3.5-flash-lite
        for attempt in range(attempts):
            try:
                return client.chat.completions.create(
                    model="gemini-3.5-flash-lite",
                    messages=messages,
                    tools=TOOLS
                )
            except Exception as e:
                last_err = e
                log_run({"warning": f"gemini attempt {attempt} failed: {e}"})
                time.sleep(10 * (attempt + 1))
        raise last_err

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

    msg = None
    final_reply = None

    try:
        for _ in range(5):
            response = call_gemini_with_retry(messages)
            msg = response.choices[0].message
            messages.append(msg)

            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    try:
                        args = json.loads(tool_call.function.arguments)
                        result = run_python(args["code"])
                    except Exception as e:
                        result = f"tool error: {e}"
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })
            else:
                break

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

    except Exception as e:
        final_reply = json.dumps({"answer": "error", "log_url": f"{BASE_URL}/run.jsonl"})
        log_run({"chat_id": chat_id, "query": text, "fatal_error": str(e)})

    log_run({"chat_id": chat_id, "query": text, "reply": final_reply})

    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": chat_id, "text": final_reply},
            timeout=15
        )
    except Exception as send_err:
        log_run({"chat_id": chat_id, "send_error": str(send_err)})

def poll_telegram():
    offset = None
    while True:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
            params = {"timeout": 30, "offset": offset}
            resp = requests.get(url, params=params, timeout=40).json()

            for result in resp.get("result", []):
                offset = result["update_id"] + 1
                if "message" in result and "text" in result["message"]:
                    chat_id = result["message"]["chat"]["id"]
                    text = result["message"]["text"]
                    threading.Thread(target=process_message, args=(chat_id, text), daemon=True).start()
        except Exception as e:
            log_run({"poll_error": str(e)})
            time.sleep(5)

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/run.jsonl")
def get_logs():
    if not os.path.exists(LOG_FILE):
        open(LOG_FILE, "a").close()
    return FileResponse(LOG_FILE, media_type="application/jsonl")

@app.on_event("startup")
def startup_event():
    threading.Thread(target=poll_telegram, daemon=True).start()
