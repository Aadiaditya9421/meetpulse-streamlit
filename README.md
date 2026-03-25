# MeetPulse — Streamlit Deployment
**ADITYA SINGH | Roll No: 23052212 | KIIT University**

## Deployed Model: MLPClassifier (F1=0.903)
- Best sklearn model — no TensorFlow dependency
- 2 hidden layers: 256 → 128 neurons, early stopping
- Previous: SVM (F1=0.888) — replaced for +1.5% accuracy improvement

## Setup
1. Run `notebook.ipynb` on Kaggle (outputs: `model.pkl`, `tfidf.pkl`, `label_encoder.pkl`)
2. Copy the 3 pkl files into this directory
3. Deploy via `streamlit run app.py` or Streamlit Cloud

## Features
- Real-time sentiment analysis with confidence scores
- Low-confidence warning (< 60%) 
- Batch CSV analysis with download
- Full model comparison charts (all 11 models)
- Session history with confidence trend chart
