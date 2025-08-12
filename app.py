# app.py — generic & fast (<3 min)
import os, re, sys, json, time, tempfile, subprocess, requests, traceback, base64
from io import BytesIO
from typing import List, Tuple, Optional
from bs4 import BeautifulSoup
from fastapi import FastAPI, UploadFile, Request  # << changed imports (Request, no File param)
from fastapi.responses import JSONResponse, PlainTextResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI

# -------------------
# Config & constants
# -------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in .env")

# For speed, you can switch to "gpt-5-mini"
MODEL_NAME = "gpt-5"
GLOBAL_DEADLINE_SEC = 165     # keep well under 180s
HTTP_TIMEOUT = 8              # fast fetch
SCRAPE_CHAR_LIMIT = 3000      # smaller context => fewer tokens & faster
CODE_TIMEOUT_SECONDS = 15     # short subprocess execution
MAX_CODE_ATTEMPTS = 1         # 0 or 1 retry total

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# -------------
# Helpers
# -------------
def now() -> float:
    return time.time()

def time_left(deadline: float) -> float:
    return max(0.0, deadline - now())

def extract_urls(text: str) -> List[str]:
    return re.findall(r"https?://[^\s,]+", text)

def fetch_and_clean_text(url: str) -> str:
    """Fast fetch with small timeout; strip nav/scripts; truncate."""
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "header", "footer", "nav", "aside", "noscript"]):
            tag.extract()
        txt = soup.get_text(separator="\n", strip=True)
        return txt[:SCRAPE_CHAR_LIMIT]
    except Exception as e:
        return f"[SCRAPE_ERROR] {e}"

def strip_code_fence(text: str) -> str:
    if not isinstance(text, str): return text
    t = text.strip()
    if t.startswith("```") and t.endswith("```"):
        parts = t.split("```")
        if len(parts) >= 2:
            return parts[1].lstrip("python\n").strip()
    return t

def run_code_subprocess(code: str, timeout: int = CODE_TIMEOUT_SECONDS) -> Tuple[bool, str]:
    """Run Python code quickly; contract: it must print final answer to stdout."""
    fd, path = tempfile.mkstemp(suffix=".py", text=True)
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        proc = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=timeout)
        if proc.returncode == 0:
            return True, proc.stdout.strip()
        return False, f"Non-zero exit ({proc.returncode}). Stderr: {proc.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return False, "TimeoutExpired"
    except Exception as e:
        return False, f"Exception: {e}"
    finally:
        try: os.remove(path)
        except: pass

def looks_like_html(s: str) -> bool:
    return isinstance(s, str) and bool(re.search(r'<!doctype html>|<html|<body|<table|<div|<p|<h[1-6]', s, re.I))

def is_data_uri_image(s: str) -> bool:
    return isinstance(s, str) and s.strip().startsWith("data:image/") if hasattr(str, "startsWith") else s.strip().startswith("data:image/")

def try_decode_json_recursive(s: str, max_depth: int = 3):
    if not isinstance(s, str): return None
    cand = s.strip()
    for _ in range(max_depth):
        try:
            parsed = json.loads(cand)
        except Exception:
            return None
        if isinstance(parsed, str):
            cand = parsed
            continue
        return parsed
    return None

def _decode_data_uri_png(data_uri: str) -> bytes:
    prefix = "data:image/png;base64,"
    if not data_uri.startswith(prefix):
        raise ValueError("Not a PNG data URI")
    return base64.b64decode(data_uri[len(prefix):])

def _encode_png_to_data_uri(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("utf-8")

def _downscale_png_until_under_limit(png_bytes: bytes, max_bytes: int = 100_000) -> bytes:
    try:
        from PIL import Image
    except ImportError:
        return png_bytes
    if len(png_bytes) <= max_bytes:
        return png_bytes
    from io import BytesIO
    im = Image.open(BytesIO(png_bytes)).convert("RGBA")
    w, h = im.size
    for scale in (0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3):
        new_w = max(1, int(w * scale)); new_h = max(1, int(h * scale))
        im_resized = im.resize((new_w, new_h), Image.LANCZOS)
        buf = BytesIO()
        im_resized.save(buf, format='PNG', optimize=True)
        data = buf.getvalue()
        if len(data) <= max_bytes:
            return data
    return data

def return_clear_response(user_instructions: str, final_answer: str):
    # Try JSON (handles double-encoded)
    parsed = try_decode_json_recursive(final_answer)
    if parsed is not None:
        return JSONResponse(parsed)
    # One-shot JSON parse
    try:
        return JSONResponse(json.loads(final_answer))
    except Exception:
        pass
    # Image data URI?
    if is_data_uri_image(final_answer):
        return JSONResponse({"data_uri": final_answer})
    # HTML?
    if looks_like_html(final_answer) or re.search(r'\breturn\b.*\bhtml\b', user_instructions, re.I):
        return HTMLResponse(final_answer)
    # Multiline => plain text
    if "\n" in final_answer:
        return PlainTextResponse(final_answer)
    # Fallback
    return PlainTextResponse(final_answer)

# -------------------
# Single fast LLM step
# -------------------
PLANNER_PROMPT = (
    "You are a fast assistant under a strict time budget.\n"
    "Given USER_INSTRUCTIONS and optional SCRAPED_CONTEXT:\n"
    "- If you can answer directly with high confidence and EXACTLY in the requested format, "
    "  RETURN ONLY THE FINAL ANSWER (no code, no extra text).\n"
    "- If you need computation/scraping/parsing, RETURN ONLY PYTHON CODE (no fences, no commentary) "
    "  that prints the final answer to stdout. If a plot is required, print a single PNG data URI "
    "  (data:image/png;base64,...) under 100,000 bytes.\n"
    "Rules: NEVER mix code and prose. Use only stdlib, requests, bs4, pandas, matplotlib if you return code."
)

def llm_decide_or_code(instructions: str, context: str, deadline: float) -> str:
    msg = (
        f"{PLANNER_PROMPT}\n\n"
        f"USER_INSTRUCTIONS:\n{instructions}\n\n"
        f"SCRAPED_CONTEXT (truncated):\n{context}\n\n"
        f"TIME_LEFT_SECONDS≈{int(time_left(deadline))}"
    )
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": msg}],
    )
    return resp.choices[0].message.content.strip()

# -------------
# Main endpoint (updated to accept ANY field name like 'questions.txt')
# -------------
@app.post("/api/")
async def api(request: Request):
    start = time.time()
    deadline = start + GLOBAL_DEADLINE_SEC

    try:
        form = await request.form()

        # Collect every uploaded file from ANY field name (e.g., questions.txt, image.png, data.csv)
        uploads: List[tuple[str, UploadFile]] = []
        for key in form.keys():
            for v in form.getlist(key):
                # Avoid isinstance checks; just detect 'file-like' parts by presence of .filename
                if hasattr(v, "filename") and v.filename:
                    uploads.append((key, v))

        if not uploads:
            return JSONResponse({"error": "No files uploaded. Send at least a 'questions.txt' file field."}, status_code=400)

        # Save all uploads
        temp_dir = tempfile.mkdtemp(prefix="uploads_")
        saved = []
        for field, u in uploads:
            raw = await u.read()
            fname = u.filename or field or "upload.bin"
            path = os.path.join(temp_dir, os.path.basename(fname))
            with open(path, "wb") as f:
                f.write(raw)
            saved.append({
                "field": field,
                "filename": fname,
                "path": path,
                "content_type": getattr(u, "content_type", "") or "",
                "size": len(raw),
            })

        # Find the questions file:
        # 1) field name exactly 'questions.txt' or 'question.txt'
        # 2) OR filename exactly 'questions.txt' or 'question.txt'
        # 3) OR any .txt/.md as a fallback
        possible_q_fields = {"questions.txt", "question.txt", "questions", "question"}
        q = next((s for s in saved if s["field"].lower() in possible_q_fields), None) \
            or next((s for s in saved if s["filename"].lower() in {"questions.txt", "question.txt"}), None) \
            or next((s for s in saved if s["filename"].lower().endswith((".txt", ".md"))), None)

        if q is None:
            return JSONResponse({"error": "No questions.txt (.txt/.md) found."}, status_code=400)

        # Read uploaded questions text
        instructions = open(q["path"], "r", encoding="utf-8", errors="ignore").read().strip()
        if not instructions:
            return JSONResponse({"error": "questions.txt is empty"}, status_code=400)

        # Quick scrape (cap to 2 URLs to save time)
        urls = extract_urls(instructions)
        contexts = []
        for u in urls[:2]:
            if time_left(deadline) < 8: break
            contexts.append(f"URL: {u}\n{fetch_and_clean_text(u)}")
        scraped_context = "\n\n".join(contexts)

        # Single LLM step: either final answer or code
        if time_left(deadline) < 8:
            return PlainTextResponse("Time budget too low before processing.")
        draft = llm_decide_or_code(instructions, scraped_context, deadline)

        # Try to treat as final answer (no code) if no obvious code signals
        looks_like_code = (
            "import " in draft or "def " in draft or
            (draft.count("\n") > 1 and "=" in draft) or
            draft.strip().startswith("```")
        )
        if not looks_like_code:
            final_answer = strip_code_fence(draft)
            # Enforce image size if data URI
            if is_data_uri_image(final_answer):
                try:
                    raw_png = _decode_data_uri_png(final_answer)
                    small_png = _downscale_png_until_under_limit(raw_png, 100_000)
                    final_answer = _encode_png_to_data_uri(small_png)
                except Exception:
                    pass
            return return_clear_response(instructions, final_answer)

        # Otherwise run code (at most 1 retry)
        code_candidate = strip_code_fence(draft)
        ok, out_or_err = run_code_subprocess(code_candidate, timeout=min(CODE_TIMEOUT_SECONDS, int(time_left(deadline)) or 5))
        if not ok and MAX_CODE_ATTEMPTS > 0 and time_left(deadline) >= 12:
            # Ask for one quick fix
            fix_prompt = (
                f"{PLANNER_PROMPT}\n\n"
                "Previous reply was code that failed. "
                "Return corrected PYTHON CODE ONLY that prints the final answer.\n"
                f"USER_INSTRUCTIONS:\n{instructions}\n\n"
                f"SCRAPED_CONTEXT:\n{scraped_context}\n\n"
                f"ERROR:\n{out_or_err}\n"
            )
            fix = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": fix_prompt}],
            ).choices[0].message.content
            code_candidate = strip_code_fence(fix)
            ok, out_or_err = run_code_subprocess(code_candidate, timeout=min(CODE_TIMEOUT_SECONDS, int(time_left(deadline)) or 5))

        # If stdout is a PNG data URI, size-limit it
        final_answer = out_or_err.strip()
        if is_data_uri_image(final_answer):
            try:
                raw_png = _decode_data_uri_png(final_answer)
                small_png = _downscale_png_until_under_limit(raw_png, 100_000)
                final_answer = _encode_png_to_data_uri(small_png)
            except Exception:
                pass

        return return_clear_response(instructions, final_answer)

    except Exception:
        traceback.print_exc()
        return JSONResponse({"error": "server error", "details": traceback.format_exc()}, status_code=500)
