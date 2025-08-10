import aiohttp
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional
import re
import base64
import tempfile
import subprocess
import requests
import pandas as pd
from io import StringIO
from fastapi import FastAPI

app = FastAPI()

# Add this root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to Data Analyst Agent API. Use /api/ endpoint."}

# ... rest of your existing code ...


AIPIPE_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZjIwMDM5OTJAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.dlQMi4pzdZ8yuaaHaUO5taTTpxlY-rXPf4cwgeHypp0"
AIPIPE_API_URL = "https://aipipe.org/openai/v1/chat/completions"

def extract_urls(text: str) -> List[str]:
    # Simple regex to extract URLs
    return re.findall(r'https?://[^\s]+', text)

def get_wikipedia_summary(query: str) -> Optional[str]:
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query}"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            return resp.json().get("extract")
    except:
        return None

def scrape_wikipedia_table(url: str) -> Optional[str]:
    try:
        tables = pd.read_html(url)
        # Choose the relevant table (heuristic: first large table)
        for table in tables:
            if table.shape[0] > 5 and table.shape[1] > 3:
                csv_data = table.to_csv(index=False)
                return csv_data
    except:
        return None

def build_prompt(question: str, data_csv: Optional[str]) -> str:
    base = (
        "You are a helpful data analyst assistant.\n"
        "Instructions:\n"
        "- Answer clearly and concisely.\n"
        "- If data is provided, analyze it.\n"
        "- If user asks for plots, return matplotlib python code in triple backticks labeled python.\n"
        "- Answer in a JSON array format if multiple questions.\n\n"
    )
    if data_csv:
        base += "Here is the dataset in CSV format:\n"
        # To keep prompt size manageable, maybe include just head or summary
        base += "\n".join(data_csv.splitlines()[:20]) + "\n\n"
    base += f"Question:\n{question}\n"
    return base

def extract_python_code(text: str) -> Optional[str]:
    match = re.search(r"```python(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def run_matplotlib_code(code: str) -> Optional[str]:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        img_path = tmpfile.name
    full_code = f"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
{code}
plt.savefig(r'{img_path}')
"""
    try:
        subprocess.run(["python", "-c", full_code], check=True, capture_output=True, timeout=20)
        with open(img_path, "rb") as f:
            img_bytes = f.read()
        return base64.b64encode(img_bytes).decode("utf-8")
    except Exception as e:
        print("Matplotlib run error:", e)
        return None

@app.post("/api/")
async def analyze(
    questions: UploadFile = File(...),
    files: Optional[List[UploadFile]] = None,
):
    question_text = (await questions.read()).decode("utf-8")
    urls = extract_urls(question_text)

    # If Wikipedia URL found, scrape table and fetch summary
    data_csv = None
    wiki_summary = None
    for url in urls:
        if "wikipedia.org" in url and "/wiki/" in url:
            wiki_summary = get_wikipedia_summary(url.split("/wiki/")[-1])
            scraped_csv = scrape_wikipedia_table(url)
            if scraped_csv:
                data_csv = scraped_csv
            break

    # Also if user uploaded files, try to read CSV files and append to prompt
    if files:
        for f in files:
            content = await f.read()
            try:
                # Check if CSV
                text = content.decode("utf-8")
                # Append file content or parse etc.
                if data_csv is None:
                    data_csv = text  # or merge multiple files logically
                else:
                    data_csv += "\n" + text
            except Exception:
                pass

    prompt = build_prompt(question_text, data_csv)
    if wiki_summary:
        prompt += "\nAdditional Wikipedia summary:\n" + wiki_summary + "\n"

    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 2048,
    }
    headers = {
        "Authorization": f"Bearer {AIPIPE_TOKEN}",
        "Content-Type": "application/json",
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(AIPIPE_API_URL, json=payload, headers=headers) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise HTTPException(status_code=resp.status, detail=f"AI Pipe API error: {text}")
            data = await resp.json()

    answer = data["choices"][0]["message"]["content"]
    python_code = extract_python_code(answer)
    image_b64 = run_matplotlib_code(python_code) if python_code else None

    response = {"answer": answer}
    if image_b64:
        response["plot_image_base64"] = image_b64
    return JSONResponse(content=response)
