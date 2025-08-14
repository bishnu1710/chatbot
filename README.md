# Mental Health Chatbot (Retrieval + Gemini Flash)

## Quick setup (local, VS Code)
1. Create & activate venv
   - macOS / Linux:
     ```
     python -m venv .venv
     source .venv/bin/activate
     ```
   - Windows (PowerShell):
     ```
    py -3.10 -m venv .venv
    .\.venv\Scripts\Activate.ps1
   
2. Install dependencies:
   - Windows (PowerShell):
    pip install -r requirements.txt
3. Run:
  - Windows (PowerShell):
    python train_retrieval.py
    streamlit run app.py