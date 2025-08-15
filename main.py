import io
import re
import base64
import traceback
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import requests

# ---- CONFIG ----
AIPIPE_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZjIwMDM5OTJAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.dlQMi4pzdZ8yuaaHaUO5taTTpxlY-rXPf4cwgeHypp0"

app = FastAPI(title="Data Analyst Agent API", docs_url="/docs")

# ---- HELPERS ----
def plot_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")

def safe_correlation(x, y):
    try:
        return float(pd.Series(x).corr(pd.Series(y)))
    except:
        return 0.0

def ai_answer(question: str) -> str:
    """Query AIPipe API for generic questions"""
    try:
        headers = {
            "Authorization": f"Bearer {AIPIPE_TOKEN}",
            "Content-Type": "application/json"
        }
        payload = {"question": question}
        resp = requests.post("https://api.aipipe.ai/ask", headers=headers, json=payload, timeout=20)
        if resp.status_code == 200:
            return resp.json().get("answer", "No answer")
        return f"AI error {resp.status_code}"
    except Exception as e:
        return f"AI failed: {str(e)}"

def process_question(q, dataframes):
    ans = None

    # --- URL scraping ---
    url_match = re.search(r"https?://[^\s]+", q)
    if url_match:
        try:
            url = url_match.group(0)
            html = requests.get(url, timeout=10).text
            soup = BeautifulSoup(html, "html.parser")
            ans = soup.get_text()[:500]  # first 500 chars
        except:
            ans = "Scraping failed"

    # --- Correlation ---
    elif "correlation" in q.lower() and dataframes:
        df = list(dataframes.values())[0]
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) >= 2:
            ans = safe_correlation(df[num_cols[0]], df[num_cols[1]])

    # --- Plot / Scatterplot ---
    elif "plot" in q.lower() or "scatterplot" in q.lower():
        fig, ax = plt.subplots()
        # Example plot: replace with actual data logic if needed
        ax.scatter([1, 2, 3], [3, 2, 5], color="blue")
        ax.plot([1, 2, 3], [3, 2, 5], "r--")
        ans = plot_to_base64(fig)

    # --- Fallback to AI ---
    if ans is None:
        ans = ai_answer(q)

    return ans

# ---- API ENDPOINT ----
@app.post("/api/")
async def analyze(questions: UploadFile = File(...), files: List[UploadFile] = File(default=[])):
    try:
        q_text = (await questions.read()).decode("utf-8", errors="ignore").strip()

        # --- Load attached data files ---
        dataframes = {}
        for f in files:
            content = await f.read()
            name = f.filename
            if name.endswith(".csv"):
                dataframes[name] = pd.read_csv(io.BytesIO(content))
            elif name.endswith(".json"):
                dataframes[name] = pd.read_json(io.BytesIO(content))
            elif name.endswith(".parquet"):
                dataframes[name] = pd.read_parquet(io.BytesIO(content))
            elif name.endswith(".xlsx"):
                dataframes[name] = pd.read_excel(io.BytesIO(content))

        # --- Detect output format ---
        array_mode = bool(re.match(r"^\s*\d+\.", q_text)) or "\n1." in q_text

        if array_mode:
            questions_list = re.split(r"\n\d+\.\s*", q_text)[1:]
            answers = [process_question(q, dataframes) for q in questions_list]
            return JSONResponse(content=answers)
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
