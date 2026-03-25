#!/bin/bash
# ============================================================
# push_streamlit.sh — Push capstone-streamlit to GitHub
#                   — Upload model to Hugging Face Hub
# Run this from INSIDE the capstone-streamlit/ folder
# ============================================================

set -e

echo "============================================"
echo "  MeetPulse Streamlit — Push Script"
echo "  GitHub + Hugging Face Hub"
echo "============================================"

# ── Step 1: Collect inputs ────────────────────────────────
read -p "Enter your GitHub username: " GH_USER
read -p "Enter your GitHub repo name (e.g. meetpulse-streamlit): " GH_REPO
read -p "Enter your GitHub Personal Access Token (PAT): " -s GH_TOKEN
echo ""
read -p "Enter your Hugging Face username: " HF_USER
read -p "Enter your Hugging Face repo name (e.g. meetpulse-sentiment): " HF_REPO
read -p "Enter your Hugging Face Access Token (write): " -s HF_TOKEN
echo ""

GH_REMOTE="https://${GH_USER}:${GH_TOKEN}@github.com/${GH_USER}/${GH_REPO}.git"

# ── PART A: GitHub Push ───────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PART A: Pushing to GitHub"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "[A1/4] Initializing git repo..."
git init
git config user.name "$GH_USER"
git config user.email "${GH_USER}@users.noreply.github.com"

echo "[A2/4] Creating .gitignore..."
cat > .gitignore << 'EOF'
__pycache__/
*.pyc
.env
.venv/
venv/
.DS_Store
.ipynb_checkpoints/
EOF

echo ""
echo "⚠️  Commit .pkl model files to GitHub?"
echo "   (Streamlit Cloud needs them in the repo)"
read -p "   Commit .pkl files? [y/n]: " COMMIT_PKL
if [ "$COMMIT_PKL" != "y" ]; then
    echo "*.pkl" >> .gitignore
fi

echo "[A3/4] Creating GitHub repo '${GH_REPO}'..."
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
    -X POST \
    -H "Authorization: token ${GH_TOKEN}" \
    -H "Content-Type: application/json" \
    -d "{
        \"name\": \"${GH_REPO}\",
        \"description\": \"MeetPulse: Video Conferencing Sentiment Analysis — Streamlit App\",
        \"private\": false,
        \"auto_init\": false
    }" \
    "https://api.github.com/user/repos")

if [ "$HTTP_STATUS" = "201" ]; then
    echo "   ✅ Repo created: https://github.com/${GH_USER}/${GH_REPO}"
elif [ "$HTTP_STATUS" = "422" ]; then
    echo "   ℹ️  Repo already exists — continuing..."
else
    echo "   ❌ Failed (HTTP $HTTP_STATUS). Check PAT permissions."
    exit 1
fi

echo "[A4/4] Committing and pushing..."
git add .
git commit -m "Initial commit: MeetPulse Streamlit app

- 4-tab Streamlit UI: Analyze, Compare, Batch, History
- ML models: LR, SVM, DT, NB, MLP (scikit-learn)
- CSV batch analysis with downloadable results
- Streamlit Community Cloud ready"

git branch -M main
git remote remove origin 2>/dev/null || true
git remote add origin "$GH_REMOTE"
git push -u origin main

echo ""
echo "  ✅ GitHub done: https://github.com/${GH_USER}/${GH_REPO}"

# ── PART B: Hugging Face Hub ──────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PART B: Uploading models to Hugging Face"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check if huggingface_hub is installed
if ! python3 -c "import huggingface_hub" 2>/dev/null; then
    echo "[B0] Installing huggingface_hub..."
    pip install huggingface_hub -q
fi

echo "[B1/4] Creating HF repo '${HF_USER}/${HF_REPO}'..."
python3 - << PYEOF
from huggingface_hub import HfApi, create_repo
import os

api = HfApi()
token = "${HF_TOKEN}"

try:
    create_repo(
        repo_id="${HF_USER}/${HF_REPO}",
        token=token,
        repo_type="model",
        exist_ok=True,
        private=False
    )
    print("   ✅ HF repo ready: https://huggingface.co/${HF_USER}/${HF_REPO}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    exit(1)
PYEOF

echo "[B2/4] Writing model card (README)..."
cat > /tmp/hf_model_card.md << 'MDEOF'
---
language: en
license: mit
tags:
  - sentiment-analysis
  - text-classification
  - meeting-transcripts
  - sklearn
datasets:
  - custom
metrics:
  - accuracy
  - f1
---

# MeetPulse — Meeting Transcript Sentiment Analyzer

**Aditya Singh | Roll No: 23052212 | KIIT University**

## Model Description
Sentiment analysis models trained on video conferencing transcripts.
Classifies text as **Positive**, **Neutral**, or **Negative**.

## Models Included
| File | Model | Accuracy |
|------|-------|----------|
| `model.pkl` | Best model (scikit-learn) | ~88% |
| `tfidf.pkl` | TF-IDF vectorizer | — |
| `label_encoder.pkl` | Label encoder | — |

## Usage
\`\`\`python
import joblib

model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")
le    = joblib.load("label_encoder.pkl")

text = "The meeting was very productive and everyone was engaged."
vec  = tfidf.transform([text])
pred = model.predict(vec)
print(le.inverse_transform(pred))  # ['Positive']
\`\`\`

## Live Demo
[Streamlit App](https://huggingface.co/spaces/${HF_USER}/${HF_REPO})

## Training Data
Custom dataset of meeting/video-conferencing transcripts with sentiment labels.

## Citation
\`\`\`
@misc{meetpulse2024,
  author = {Aditya Singh},
  title  = {MeetPulse: Sentiment Analysis for Meeting Transcripts},
  year   = {2024},
  url    = {https://huggingface.co/${HF_USER}/${HF_REPO}}
}
\`\`\`
MDEOF

echo "[B3/4] Uploading model files to Hugging Face..."
python3 - << PYEOF
from huggingface_hub import HfApi
import os, glob

api = HfApi()
token = "${HF_TOKEN}"
repo_id = "${HF_USER}/${HF_REPO}"

# Upload model card
api.upload_file(
    path_or_fileobj="/tmp/hf_model_card.md",
    path_in_repo="README.md",
    repo_id=repo_id,
    token=token,
    repo_type="model"
)
print("   ✅ Model card uploaded")

# Upload .pkl files if they exist
pkl_files = glob.glob("*.pkl")
if not pkl_files:
    print("   ⚠️  No .pkl files found in current directory.")
    print("      Run notebook.ipynb first to generate model files.")
else:
    for f in pkl_files:
        api.upload_file(
            path_or_fileobj=f,
            path_in_repo=f,
            repo_id=repo_id,
            token=token,
            repo_type="model"
        )
        print(f"   ✅ Uploaded: {f}")

# Upload notebook
if os.path.exists("notebook.ipynb"):
    api.upload_file(
        path_or_fileobj="notebook.ipynb",
        path_in_repo="notebook.ipynb",
        repo_id=repo_id,
        token=token,
        repo_type="model"
    )
    print("   ✅ Uploaded: notebook.ipynb")
PYEOF

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅ ALL DONE!"
echo ""
echo "  📦 GitHub:       https://github.com/${GH_USER}/${GH_REPO}"
echo "  🤗 Hugging Face: https://huggingface.co/${HF_USER}/${HF_REPO}"
echo ""
echo "  Next steps:"
echo "  1. Deploy Streamlit app:"
echo "     → https://share.streamlit.io"
echo "     → New App → ${GH_USER}/${GH_REPO} → app.py"
echo ""
echo "  2. Update README.md with your live Streamlit URL"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
