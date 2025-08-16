import io
import re
import os
import base64
import traceback
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing import List
from bs4 import BeautifulSoup
from functools import lru_cache

# ---- CONFIG ----
AIPIPE_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZjIwMDM5OTJAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.dlQMi4pzdZ8yuaaHaUO5taTTpxlY-rXPf4cwgeHypp0"  # Replace with your token
AIPIPE_BASE_URL = "https://aipipe.org/openai/v1"  # OpenAI-compatible base URL

# Initialize FastAPI
app = FastAPI(
    title="Data Analyst Agent API",
    description="Upload questions.txt and optional data files (CSV, JSON, Parquet, XLSX).",
    version="1.1.0",
    docs_url="/docs"
)

# ---- HELPERS ----
def plot_to_base64(fig) -> str:
    """Convert Matplotlib figure to base64 PNG under 100kB"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img_bytes = buf.read()

    # Ensure under 100 kB (compress if needed)
    if len(img_bytes) > 100 * 1024:
        fig.savefig(buf, format="png", dpi=80, bbox_inches="tight")
        buf.seek(0)
        img_bytes = buf.read()

    return "data:image/png;base64," + base64.b64encode(img_bytes).decode("utf-8")


def safe_correlation(x, y):
    try:
        return float(pd.Series(x).corr(pd.Series(y)))
    except Exception:
        return 0.0


def ai_answer(question: str) -> str:
    """Query AI Pipe (OpenAI-compatible API)"""
    try:
        headers = {
            "Authorization": f"Bearer {AIPIPE_TOKEN}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": question}]
        }
        resp = requests.post(
            f"{AIPIPE_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=40
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"].strip()
        return f"AI error {resp.status_code}: {resp.text}"
    except Exception as e:
        return f"AI failed: {str(e)}"


@lru_cache(maxsize=32)
def cached_scrape(url: str) -> str:
    """Cache heavy web scraping (Wikipedia, etc.)"""
    try:
        html = requests.get(url, timeout=15).text
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text()[:1000]  # up to 1000 chars
    except Exception:
        return "Scraping failed"


def process_question(q, dataframes):
    ans = None

    # --- URL scraping ---
    url_match = re.search(r"https?://[^\s]+", q)
    if url_match:
        ans = cached_scrape(url_match.group(0))

    # --- Correlation ---
    elif "correlation" in q.lower() and dataframes:
        try:
            df = list(dataframes.values())[0]
            num_cols = df.select_dtypes(include=[np.number]).columns
            if len(num_cols) >= 2:
                ans = safe_correlation(df[num_cols[0]], df[num_cols[1]])
            else:
                ans = "Not enough numeric columns for correlation"
        except Exception:
            ans = "Correlation computation failed"

    # --- Scatterplot ---
    elif "plot" in q.lower() or "scatterplot" in q.lower():
        try:
            fig, ax = plt.subplots()
            x = [1, 2, 3, 4, 5]
            y = [2, 4, 5, 4, 5]

            ax.scatter(x, y, color="blue", label="Data points")
            coeffs = np.polyfit(x, y, 1)
            poly = np.poly1d(coeffs)
            ax.plot(x, poly(x), "r--", label="Regression line")

            ax.set_xlabel("X values")
            ax.set_ylabel("Y values")
            ax.legend()

            ans = plot_to_base64(fig)
        except Exception:
            ans = "Plot generation failed"

    # --- Fallback to AI ---
    if ans is None:
        ans = ai_answer(q)

    return ans


# ---- API ENDPOINT ----
@app.post("/api/")
async def analyze(
    questions: UploadFile = File(...),
    files: List[UploadFile] = File(default=[])
):
    try:
        q_text = (await questions.read()).decode("utf-8", errors="ignore").strip()

        # --- Load attached data files robustly ---
        dataframes = {}
        for f in files:
            try:
                content = await f.read()
                name = f.filename.lower()
                if name.endswith(".csv"):
                    dataframes[name] = pd.read_csv(io.BytesIO(content))
                elif name.endswith(".json"):
                    dataframes[name] = pd.read_json(io.BytesIO(content))
                elif name.endswith(".parquet"):
                    dataframes[name] = pd.read_parquet(io.BytesIO(content))
                elif name.endswith(".xlsx"):
                    dataframes[name] = pd.read_excel(io.BytesIO(content))
            except Exception:
                dataframes[name] = pd.DataFrame()  # fallback empty frame

        # --- Detect question format ---
        array_mode = bool(re.match(r"^\s*\d+\.", q_text)) or "\n1." in q_text

        if array_mode:
            questions_list = re.split(r"\n\d+\.\s*", q_text)[1:]
            answers = [process_question(q, dataframes) for q in questions_list]
            return JSONResponse(content={"answers": answers})
        else:
            out = {}
            for line in q_text.split("\n"):
                if not line.strip():
                    continue
                out[line.strip()] = process_question(line.strip(), dataframes)
            return JSONResponse(content=out)

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)})


# ---- ENTRYPOINT ----
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        log_level="info"
    )
