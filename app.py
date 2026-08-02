import os
import json
import time
import threading
import requests
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from openai import OpenAI
import sys
import io


# =========================================================
# ENVIRONMENT VARIABLES
# =========================================================

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Example:
# https://analyst-bot-xxxx.onrender.com
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000").rstrip("/")

LOG_FILE = "run.jsonl"


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI()


# =========================================================
# GEMINI CLIENT
# =========================================================

client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


# Only one Gemini request at a time to reduce rate-limit errors
gemini_lock = threading.Lock()


# =========================================================
# MULTI-TURN TELEGRAM MEMORY
# =========================================================

# Stores recent messages for each Telegram chat.
# Useful because IITM may send multi-turn questions.
chat_history = {}

history_lock = threading.Lock()

MAX_HISTORY_MESSAGES = 10


# =========================================================
# TOOLS
# =========================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Execute Python code server-side to fetch public datasets, "
                "perform calculations, analyze data and compute the answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string"
                    }
                },
                "required": ["code"]
            }
        }
    }
]


# =========================================================
# LOGGING
# =========================================================

def log_run(data):
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, default=str) + "\n")
    except Exception:
        pass


# =========================================================
# PYTHON EXECUTION TOOL
# =========================================================

def run_python(code: str) -> str:

    old_stdout = sys.stdout
    redirected_output = io.StringIO()

    sys.stdout = redirected_output

    try:

        # Provide a normal Python environment.
        execution_globals = {
            "__builtins__": __builtins__
        }

        exec(code, execution_globals)

        output = redirected_output.getvalue()

        if not output:
            output = "Python code executed successfully but printed no output."

    except Exception as e:

        output = f"Python execution error: {type(e).__name__}: {str(e)}"

    finally:

        sys.stdout = old_stdout

    # Prevent enormous tool responses
    return output[:12000]


# =========================================================
# GEMINI API WITH RETRIES
# =========================================================

def call_gemini_with_retry(messages, attempts=4):

    last_err = None

    with gemini_lock:

        # Small delay for Gemini API rate limits
        time.sleep(5)

        for attempt in range(attempts):

            try:

                return client.chat.completions.create(
                    model="gemini-3.5-flash-lite",
                    messages=messages,
                    tools=TOOLS
                )

            except Exception as e:

                last_err = e

                log_run({
                    "warning": f"Gemini attempt {attempt + 1} failed",
                    "error": str(e)
                })

                time.sleep(10 * (attempt + 1))

    raise last_err


# =========================================================
# CHAT HISTORY
# =========================================================

def get_chat_history(chat_id):

    with history_lock:

        return list(
            chat_history.get(chat_id, [])
        )


def save_user_message(chat_id, text):

    with history_lock:

        if chat_id not in chat_history:
            chat_history[chat_id] = []

        chat_history[chat_id].append({
            "role": "user",
            "content": text
        })

        # Keep only recent conversation
        chat_history[chat_id] = chat_history[chat_id][-MAX_HISTORY_MESSAGES:]


def save_assistant_message(chat_id, text):

    with history_lock:

        if chat_id not in chat_history:
            chat_history[chat_id] = []

        chat_history[chat_id].append({
            "role": "assistant",
            "content": text
        })

        chat_history[chat_id] = chat_history[chat_id][-MAX_HISTORY_MESSAGES:]


# =========================================================
# PROCESS TELEGRAM MESSAGE
# =========================================================

def process_message(chat_id, text):

    system_prompt = f"""
You are an autonomous data analyst.

Your job is to solve the user's data-analysis question accurately.

You may use the run_python tool to:

- perform calculations
- analyze tables
- parse data
- download public datasets
- call public APIs
- use pandas
- use numpy
- use requests
- process JSON
- process CSV data

IMPORTANT:

1. Carefully read the exact JSON format requested by the user.

2. Use Python whenever calculations or dataset analysis are required.

3. Do not guess numerical answers when they can be calculated.

4. Some questions may be multi-turn. Previous messages may contain
important data or context needed for the latest question.

5. Your FINAL response MUST be exactly ONE valid JSON object.

6. Do NOT use Markdown.

7. Do NOT use ```json fences.

8. Do NOT write explanations outside the JSON.

9. Preserve the exact answer structure requested by the user.

Example:

{{"answer": {{"state": "Assam"}}, "log_url": "URL"}}

The server will automatically replace log_url with the correct public URL.
"""


    # -----------------------------------------------------
    # Save incoming message
    # -----------------------------------------------------

    save_user_message(chat_id, text)


    # -----------------------------------------------------
    # Build conversation
    # -----------------------------------------------------

    history = get_chat_history(chat_id)

    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    messages.extend(history)


    msg = None
    final_reply = None


    try:

        # Allow Gemini several reasoning/tool rounds
        for _ in range(6):

            response = call_gemini_with_retry(messages)

            msg = response.choices[0].message

            messages.append(msg)


            # -------------------------------------------------
            # TOOL CALLS
            # -------------------------------------------------

            if msg.tool_calls:

                for tool_call in msg.tool_calls:

                    try:

                        args = json.loads(
                            tool_call.function.arguments
                        )

                        code = args.get("code", "")

                        log_run({
                            "chat_id": chat_id,
                            "tool": "run_python",
                            "code": code
                        })

                        result = run_python(code)

                        log_run({
                            "chat_id": chat_id,
                            "tool_result": result
                        })

                    except Exception as e:

                        result = f"Tool error: {str(e)}"


                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })

            else:

                # Gemini produced final answer
                break


        # -----------------------------------------------------
        # PARSE FINAL RESPONSE
        # -----------------------------------------------------

        if msg and msg.content:

            final_text = msg.content.strip()

            # Remove accidental Markdown fences
            final_text = (
                final_text
                .replace("```json", "")
                .replace("```JSON", "")
                .replace("```", "")
                .strip()
            )


            try:

                parsed = json.loads(final_text)

            except Exception:

                # Try extracting JSON if Gemini added accidental text
                start = final_text.find("{")
                end = final_text.rfind("}")

                if start != -1 and end != -1:

                    parsed = json.loads(
                        final_text[start:end + 1]
                    )

                else:

                    raise ValueError(
                        "Gemini did not return valid JSON"
                    )


            # -------------------------------------------------
            # ENSURE answer FIELD
            # -------------------------------------------------

            if "answer" not in parsed:

                parsed = {
                    "answer": parsed
                }


            # -------------------------------------------------
            # FORCE CORRECT LOG URL
            # -------------------------------------------------

            parsed["log_url"] = f"{BASE_URL}/run.jsonl"


            # Compact JSON
            final_reply = json.dumps(
                parsed,
                separators=(",", ":")
            )


        else:

            final_reply = json.dumps(
                {
                    "answer": "error",
                    "log_url": f"{BASE_URL}/run.jsonl"
                },
                separators=(",", ":")
            )


    except Exception as e:

        final_reply = json.dumps(
            {
                "answer": "error",
                "log_url": f"{BASE_URL}/run.jsonl"
            },
            separators=(",", ":")
        )

        log_run({
            "chat_id": chat_id,
            "query": text,
            "fatal_error": str(e)
        })


    # =====================================================
    # SAVE ASSISTANT RESPONSE TO MEMORY
    # =====================================================

    save_assistant_message(
        chat_id,
        final_reply
    )


    # =====================================================
    # LOG FINAL RESULT
    # =====================================================

    log_run({
        "chat_id": chat_id,
        "query": text,
        "reply": final_reply
    })


    # =====================================================
    # SEND TELEGRAM RESPONSE
    # =====================================================

    try:

        response = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": final_reply
            },
            timeout=20
        )

        log_run({
            "chat_id": chat_id,
            "telegram_status": response.status_code
        })

    except Exception as send_err:

        log_run({
            "chat_id": chat_id,
            "send_error": str(send_err)
        })


# =========================================================
# TELEGRAM WEBHOOK
# =========================================================

@app.post("/telegram-webhook")
async def telegram_webhook(request: Request):

    try:

        update = await request.json()

        message = update.get("message")

        if not message:
            return {"ok": True}


        chat = message.get("chat", {})

        chat_id = chat.get("id")

        text = message.get("text")


        if chat_id is None or not text:
            return {"ok": True}


        # -------------------------------------------------
        # Process asynchronously
        # -------------------------------------------------

        threading.Thread(
            target=process_message,
            args=(chat_id, text),
            daemon=True
        ).start()


        # Respond immediately to Telegram
        return {"ok": True}


    except Exception as e:

        log_run({
            "webhook_error": str(e)
        })

        return {"ok": True}


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():

    return {
        "ok": True,
        "service": "Analyst Telegram Bot"
    }


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():

    return {
        "status": "running",
        "bot": "AnalystMain_bot"
    }


# =========================================================
# PUBLIC LOG FILE
# =========================================================

@app.get("/run.jsonl")
def get_logs():

    if not os.path.exists(LOG_FILE):

        open(
            LOG_FILE,
            "a",
            encoding="utf-8"
        ).close()


    return FileResponse(
        LOG_FILE,
        media_type="application/jsonl"
    )
