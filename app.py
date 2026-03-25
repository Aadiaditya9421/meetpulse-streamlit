"""
MeetPulse Streamlit v3 — Video Conferencing Sentiment Analysis
ADITYA SINGH | Roll No: 23052212 | KIIT University

Primary Model  : MLPClassifier (F1=0.8805, Acc=0.8804)
Fallback Model : SVC Linear   (F1=0.7596, Acc=0.7594)

v3.0 Improvements:
  ✅ MLP primary with SVM fallback (auto-detected)
  ✅ Confidence levels: High/Moderate/Low with colour coding
  ✅ Feature importance (Explain tab) — top TF-IDF words
  ✅ Side-by-side MLP vs SVM comparison tab
  ✅ Batch analysis: max 1000 rows, progress bar, download
  ✅ Session history with trend chart
  ✅ Real model F1 scores from the final notebook
"""

import re
import time
from pathlib import Path

import joblib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="MeetPulse v3 — Meeting Analysis",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "could", "did", "do", "does", "doing",
    "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more",
    "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same",
    "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs",
    "them", "themselves", "then", "there", "these", "they", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when",
    "where", "which", "while", "who", "whom", "why", "will", "with", "you", "your",
    "yours", "yourself", "yourselves",
}

st.markdown("""
<style>
  .block-container { padding-top: 1.2rem; }

  .metric-card {
    background: white; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 1.1rem; text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,.06);
  }
  .metric-val { font-size: 1.9rem; font-weight: 700; line-height: 1; margin-bottom: .25rem; }
  .metric-lbl { font-size: .78rem; color: #64748b; }

  .result-positive { background:#dcfce7; border-left:5px solid #16a34a; border-radius:8px; padding:1rem; }
  .result-negative { background:#fee2e2; border-left:5px solid #dc2626; border-radius:8px; padding:1rem; }
  .result-neutral  { background:#fef3c7; border-left:5px solid #d97706; border-radius:8px; padding:1rem; }

  .app-header {
    background: linear-gradient(135deg, #1e40af, #1e3a8a);
    color: white; padding: 1.4rem 2rem; border-radius: 12px; margin-bottom: 1.3rem;
  }
  .app-header h1 { margin:0; font-size:1.7rem; }
  .app-header p  { margin:.35rem 0 0; opacity:.85; font-size:.9rem; }

  .conf-high     { background:#dcfce7; color:#16a34a; border-radius:4px; padding:2px 8px; font-weight:600; }
  .conf-moderate { background:#fef3c7; color:#d97706; border-radius:4px; padding:2px 8px; font-weight:600; }
  .conf-low      { background:#fee2e2; color:#dc2626; border-radius:4px; padding:2px 8px; font-weight:600; }

  .low-conf-warn { background:#fef3c7; border:1px solid #f59e0b; border-radius:6px;
                   padding:.55rem; font-size:.83rem; margin-top:.5rem; }
</style>
""", unsafe_allow_html=True)

MODEL_F1_MAP = {
    "MLPClassifier": 0.8805,
    "SVC": 0.7596,
    "LogisticRegression": 0.7465,
    "DecisionTreeClassifier": 0.4917,
    "MultinomialNB": 0.7066,
}


@st.cache_resource
def load_models():
    mlp = tfidf = le = None
    svm = svm_tf = svm_le = None
    mlp_error = svm_error = None

    try:
        mlp = joblib.load(BASE_DIR / "model.pkl")
        tfidf = joblib.load(BASE_DIR / "tfidf.pkl")
        le = joblib.load(BASE_DIR / "label_encoder.pkl")
        mlp_ok = True
    except Exception as exc:
        mlp_ok = False
        mlp_error = str(exc)

    try:
        svm = joblib.load(BASE_DIR / "svm_model.pkl")
        svm_tf = joblib.load(BASE_DIR / "svm_tfidf.pkl")
        svm_le = joblib.load(BASE_DIR / "svm_label_encoder.pkl")
        svm_ok = True
    except Exception as exc:
        svm_ok = False
        svm_error = str(exc)

    return mlp, tfidf, le, mlp_ok, mlp_error, svm, svm_tf, svm_le, svm_ok, svm_error


(
    mlp_model,
    mlp_tfidf,
    mlp_le,
    mlp_ok,
    mlp_error,
    svm_model,
    svm_tfidf,
    svm_le,
    svm_ok,
    svm_error,
) = load_models()

if mlp_ok:
    model, tfidf, le = mlp_model, mlp_tfidf, mlp_le
    model_loaded = True
elif svm_ok:
    model, tfidf, le = svm_model, svm_tfidf, svm_le
    model_loaded = True
    st.warning("⚠️ MLP not found — SVM is active as primary model.")
else:
    model = tfidf = le = None
    model_loaded = False


def preprocess(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = [t for t in text.split() if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)


def conf_level(conf: float) -> str:
    return "high" if conf >= 70 else ("moderate" if conf >= 55 else "low")


def get_probabilities(mdl, vec):
    if hasattr(mdl, "predict_proba"):
        return np.asarray(mdl.predict_proba(vec)[0], dtype=float)

    if hasattr(mdl, "decision_function"):
        decision = np.asarray(mdl.decision_function(vec), dtype=float)
        if decision.ndim == 1:
            if len(getattr(mdl, "classes_", [])) == 2:
                pos = 1.0 / (1.0 + np.exp(-decision[0]))
                return np.array([1.0 - pos, pos], dtype=float)
            exp_vals = np.exp(decision - np.max(decision))
            return exp_vals / exp_vals.sum()

        decision = decision[0]
        exp_vals = np.exp(decision - np.max(decision))
        return exp_vals / exp_vals.sum()

    pred = mdl.predict(vec)[0]
    classes = list(getattr(mdl, "classes_", []))
    if not classes:
        return np.array([1.0], dtype=float)
    probs = np.zeros(len(classes), dtype=float)
    probs[classes.index(pred)] = 1.0
    return probs


def predict(text: str, mdl=None, tv=None, encoder=None):
    mdl = mdl or model
    tv = tv or tfidf
    encoder = encoder or le
    if mdl is None or tv is None or encoder is None:
        return None
    clean = preprocess(text)
    if not clean:
        return None
    vec = tv.transform([clean])
    classes = encoder.classes_.tolist()
    proba = get_probabilities(mdl, vec)
    if len(proba) != len(classes):
        pred = mdl.predict(vec)[0]
        proba = np.zeros(len(classes), dtype=float)
        proba[classes.index(pred)] = 1.0
    idx = int(np.argmax(proba))
    conf = round(float(proba[idx]) * 100, 2)
    return {
        "prediction": classes[idx],
        "confidence": conf,
        "conf_level": conf_level(conf),
        "scores": {c: round(float(p) * 100, 2) for c, p in zip(classes, proba)},
        "word_count": len(text.split()),
        "clean_words": len(clean.split()),
        "low_confidence": conf < 55.0,
        "proba": proba,
        "classes": classes,
    }


def explain_features(text: str, mdl=None, tv=None, encoder=None, top_n=10):
    mdl = mdl or model
    tv = tv or tfidf
    encoder = encoder or le
    if mdl is None or tv is None or encoder is None:
        return []
    clean = preprocess(text)
    if not clean:
        return []
    vec = tv.transform([clean])
    vocab_inv = {v: k for k, v in tv.vocabulary_.items()}
    vec_arr = vec.toarray()[0]
    nonzero = np.where(vec_arr > 0)[0]
    scored = sorted(
        [(vocab_inv[i], float(vec_arr[i])) for i in nonzero if i in vocab_inv],
        key=lambda x: -x[1],
    )[:top_n]
    classes = encoder.classes_.tolist()
    pos_idx = classes.index("Positive") if "Positive" in classes else 0
    neg_idx = classes.index("Negative") if "Negative" in classes else min(1, len(classes) - 1)
    result = []
    for word, score in scored:
        w_vec = tv.transform([word])
        w_prob = get_probabilities(mdl, w_vec)
        direction = "positive" if w_prob[pos_idx] > w_prob[neg_idx] else "negative"
        result.append({"word": word, "tfidf_score": round(score, 4), "direction": direction})
    return result


active_name = type(model).__name__ if model else "N/A"
active_f1 = MODEL_F1_MAP.get(active_name, "—")
feature_count = len(getattr(tfidf, "vocabulary_", {}) or {}) if tfidf is not None else 0

st.markdown(f"""
<div class="app-header">
  <h1>🎙️ MeetPulse — Video Conferencing Analysis</h1>
  <p>Sentiment analysis for meeting transcripts &mdash;
     ADITYA SINGH | Roll No: 23052212 | KIIT University &mdash;
     Active: <strong>{active_name}</strong> (F1={active_f1})</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Model Status")

    if mlp_ok:
        st.success("✅ PRIMARY: **MLPClassifier** (F1=0.8805)")
    else:
        st.error(f"❌ model.pkl failed to load. {mlp_error or 'Missing or incompatible artifact.'}")

    if svm_ok:
        st.info("⚡ FALLBACK: **SVC Linear** (F1=0.7596) — loaded")
    else:
        st.warning(f"⚠️ svm_model.pkl failed to load. {svm_error or 'Missing or incompatible artifact.'}")

    st.markdown("---")
    if model_loaded:
        st.info(f"📊 TF-IDF Features: **{feature_count:,}**")
        st.info(f"🏷️ Classes: **{', '.join(le.classes_)}**")

    st.markdown("---")
    st.markdown("### 📌 About MLP")
    st.markdown("""
**Architecture:** 2 hidden layers
- Layer 1: 256 neurons
- Layer 2: 128 neurons
- Early stopping, val_fraction=0.1
- random_state=42

**Pipeline:** TF-IDF (5k bigrams) → MLP

**Confidence thresholds:**
- 🟢 High: ≥70%
- 🟡 Moderate: 55–70%
- 🔴 Low: <55%
    """)

    st.markdown("---")
    st.markdown("### 💡 Sample Texts")
    sample_options = {
        "Positive 😊": "Great progress on the sprint! Team delivered all user stories ahead of schedule and the demo impressed the client significantly.",
        "Negative 😟": "We are severely behind schedule. Critical blockers remain unresolved and the deployment pipeline keeps failing. Client escalation expected.",
        "Neutral 😐": "The team reviewed the backlog during today's planning session and estimated story points. Architecture trade-offs were also discussed.",
    }
    selected_sample = st.selectbox("Load sample:", list(sample_options.keys()))
    if st.button("📋 Load Sample"):
        st.session_state["sample_text"] = sample_options[selected_sample]

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔍 Analyze", "🧠 Explain", "⚔️ MLP vs SVM",
    "📊 Model Comparison", "📁 Batch Analysis", "📈 History"
])

with tab1:
    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.markdown("#### 📝 Enter Meeting Transcript")
        default_text = st.session_state.get("sample_text", "")
        text_input = st.text_area(
            "Paste your meeting transcript:",
            value=default_text, height=220, max_chars=5000,
            placeholder="Example: Great progress on the sprint, team delivered ahead of schedule…"
        )
        word_ct = len(text_input.split()) if text_input.strip() else 0
        st.caption(f"📝 {word_ct} words · {len(text_input)}/5000 chars")
        bc1, bc2 = st.columns(2)
        analyze_clicked = bc1.button("⚡ Analyze Sentiment", type="primary", use_container_width=True)
        if bc2.button("🗑️ Clear", use_container_width=True):
            st.session_state["sample_text"] = ""
            st.rerun()

    with col_out:
        st.markdown("#### 📊 Results")
        if analyze_clicked:
            if not model_loaded:
                st.error("Model not loaded. Confirm the .pkl artifacts are present in the app folder.")
            elif not text_input.strip():
                st.warning("⚠️ Please enter some text.")
            else:
                with st.spinner("Analyzing…"):
                    t0 = time.perf_counter()
                    result = predict(text_input)
                    latency = round((time.perf_counter() - t0) * 1000, 2)

                if result is None:
                    st.error("No meaningful text after preprocessing.")
                else:
                    pred = result["prediction"]
                    conf = result["confidence"]
                    level = result["conf_level"]

                    if "history" not in st.session_state:
                        st.session_state["history"] = []
                    st.session_state["history"].insert(0, {
                        "text": text_input[:80] + ("…" if len(text_input) > 80 else ""),
                        "sentiment": pred, "confidence": conf,
                        "conf_level": level,
                        "words": result["word_count"], "latency_ms": latency
                    })

                    icons = {"Positive": "😊", "Negative": "😟", "Neutral": "😐"}
                    colors = {"Positive": "green", "Negative": "red", "Neutral": "orange"}
                    css_cls = {"Positive": "result-positive", "Negative": "result-negative", "Neutral": "result-neutral"}
                    conf_css = {"high": "conf-high", "moderate": "conf-moderate", "low": "conf-low"}

                    st.markdown(f"""
                    <div class="{css_cls.get(pred, 'result-neutral')}">
                      <div style="font-size:2.4rem;text-align:center">{icons.get(pred, '📝')}</div>
                      <h3 style="text-align:center;color:{colors.get(pred, 'orange')};margin:.4rem 0">{pred}</h3>
                      <p style="text-align:center;margin:0;font-size:.88rem">
                        Confidence: <strong>{conf}%</strong>
                        &nbsp;<span class="{conf_css[level]}">{level.capitalize()}</span>
                      </p>
                    </div>
                    """, unsafe_allow_html=True)

                    if level in ("low", "moderate"):
                        st.markdown(
                            f"<div class='low-conf-warn'>⚠️ <strong>{level.capitalize()} confidence</strong>"
                            " — prediction may be uncertain. Review manually if critical.</div>",
                            unsafe_allow_html=True
                        )

                    st.markdown("##### Score Breakdown")
                    for label, score in sorted(result["scores"].items(), key=lambda x: -x[1]):
                        st.progress(score / 100, text=f"**{label}**: {score}%")

                    st.markdown("---")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Words", result["word_count"])
                    m2.metric("Latency", f"{latency} ms")
                    m3.metric("Model", active_name.replace("Classifier", ""))
                    m4.metric("F1", active_f1)
        else:
            st.info("👈 Enter text and click **Analyze Sentiment**.")

with tab2:
    st.markdown("#### 🧠 Feature Explanation — Top TF-IDF Contributing Words")
    st.caption("Shows which words in your text most strongly influenced the prediction.")

    exp_text = st.text_area("Text to explain:", height=130,
                             placeholder="Paste meeting text here…", key="exp_text")
    if st.button("🔍 Explain Prediction", type="primary"):
        if not model_loaded:
            st.error("Model not loaded.")
        elif not exp_text.strip():
            st.warning("Enter text first.")
        else:
            with st.spinner("Computing feature importance…"):
                r = predict(exp_text)
                feats = explain_features(exp_text)

            if r is None:
                st.error("No meaningful text after preprocessing.")
            else:
                pred = r["prediction"]
                icons = {"Positive": "😊", "Negative": "😟", "Neutral": "😐"}
                st.success(f"**Prediction:** {icons.get(pred, '📝')} {pred} — {r['confidence']}% ({r['conf_level']})")

                if feats:
                    st.markdown("##### Top Contributing Words")
                    df_feats = pd.DataFrame(feats)
                    fig, ax = plt.subplots(figsize=(9, 4))
                    colors_feat = ["#16a34a" if d == "positive" else "#dc2626" for d in df_feats["direction"]]
                    ax.barh(df_feats["word"], df_feats["tfidf_score"],
                            color=colors_feat, alpha=0.85, edgecolor="white")
                    ax.set_xlabel("TF-IDF Score")
                    ax.set_title("Top Words by TF-IDF Weight (green=positive signal, red=negative)")
                    ax.invert_yaxis()
                    green_patch = mpatches.Patch(color="#16a34a", label="Positive signal")
                    red_patch = mpatches.Patch(color="#dc2626", label="Negative signal")
                    ax.legend(handles=[green_patch, red_patch], fontsize=9)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    st.dataframe(
                        df_feats.rename(columns={
                            "word": "Word",
                            "tfidf_score": "TF-IDF Score",
                            "direction": "Sentiment Signal"
                        }),
                        use_container_width=True
                    )
                else:
                    st.warning("No features found — text may be too short.")

with tab3:
    st.markdown("#### ⚔️ MLP vs SVM Side-by-Side Comparison")

    if not mlp_ok and not svm_ok:
        st.error("Neither model is loaded. Confirm the exported artifacts are committed to the repo.")
    elif not svm_ok:
        st.warning("svm_model.pkl not found or failed to load, so only the primary model is available.")
    else:
        cmp_text = st.text_area("Text for comparison:", height=130, key="cmp_text",
                                 placeholder="Enter meeting text to compare both models…")
        if st.button("⚔️ Compare Models", type="primary"):
            if not cmp_text.strip():
                st.warning("Enter text first.")
            else:
                with st.spinner("Running both models…"):
                    mlp_r = predict(cmp_text, mlp_model, mlp_tfidf, mlp_le) if mlp_ok else None
                    svm_r = predict(cmp_text, svm_model, svm_tfidf, svm_le) if svm_ok else None

                c1, c2 = st.columns(2)
                icons = {"Positive": "😊", "Negative": "😟", "Neutral": "😐"}

                with c1:
                    st.markdown("##### 🧠 MLP (Primary)")
                    if mlp_r:
                        st.metric("Prediction", f"{icons.get(mlp_r['prediction'], '📝')} {mlp_r['prediction']}")
                        st.metric("Confidence", f"{mlp_r['confidence']}%")
                        st.metric("F1 Score", "0.8805")
                        for lbl, sc in sorted(mlp_r["scores"].items(), key=lambda x: -x[1]):
                            st.progress(sc / 100, text=f"{lbl}: {sc}%")
                    else:
                        st.error("MLP not available")

                with c2:
                    st.markdown("##### ⚡ SVM (Fallback)")
                    if svm_r:
                        st.metric("Prediction", f"{icons.get(svm_r['prediction'], '📝')} {svm_r['prediction']}")
                        st.metric("Confidence", f"{svm_r['confidence']}%")
                        st.metric("F1 Score", "0.7596")
                        for lbl, sc in sorted(svm_r["scores"].items(), key=lambda x: -x[1]):
                            st.progress(sc / 100, text=f"{lbl}: {sc}%")
                    else:
                        st.error("SVM not available")

                if mlp_r and svm_r:
                    agreed = mlp_r["prediction"] == svm_r["prediction"]
                    if agreed:
                        st.success(f"✅ Both models **agree**: {mlp_r['prediction']}")
                    else:
                        st.error(f"⚠️ Models **disagree**: MLP→{mlp_r['prediction']} | SVM→{svm_r['prediction']}")

with tab4:
    st.markdown("#### 📊 All Models Performance Comparison")
    st.caption("Results from training on ~8,500 meeting transcript samples (80/20 split, stratified)")

    model_data = {
        "Model": [
            "Linear Regression*", "Multiple Linear Reg.*",
            "Logistic Regression", "Decision Tree",
            "SVM (Linear) ⚡", "Naive Bayes",
            "Single Layer Perceptron", "Multi Layer Perceptron ✅",
            "CNN (1D Keras) 🏆", "RNN (SimpleRNN)",
            "LR + SVD"
        ],
        "Accuracy": [0.621, 0.639, 0.7463, 0.5105, 0.7594, 0.7087, 0.7714, 0.8804, 0.8599, 0.3333, 0.5962],
        "Precision": [0.600, 0.618, 0.7467, 0.6172, 0.7598, 0.7108, 0.7721, 0.8826, 0.8617, 0.1111, 0.6002],
        "Recall": [0.621, 0.639, 0.7463, 0.5105, 0.7594, 0.7087, 0.7714, 0.8804, 0.8599, 0.3333, 0.5962],
        "F1-Score": [0.608, 0.626, 0.7465, 0.4917, 0.7596, 0.7066, 0.7716, 0.8805, 0.8600, 0.1667, 0.5971],
        "Type": [
            "Regression", "Regression",
            "Classification", "Classification",
            "Classification", "Classification",
            "Neural Network", "Neural Network",
            "Neural Network", "Neural Network",
            "Dim. Reduction"
        ],
        "Role": ["", "", "", "", "FALLBACK ⚡", "", "", "PRIMARY ✅", "Best overall", "Unstable", ""]
    }
    df_models = pd.DataFrame(model_data)

    fig1, ax1 = plt.subplots(figsize=(12, 5))
    type_colors = {
        "Regression": "#94a3b8", "Classification": "#2563eb",
        "Neural Network": "#16a34a", "Dim. Reduction": "#d97706"
    }
    bar_cols = [type_colors[t] for t in df_models["Type"]]
    bars = ax1.barh(df_models["Model"], df_models["F1-Score"],
                    color=bar_cols, alpha=0.88, edgecolor="white", linewidth=1.2)
    ax1.set_xlim(0, 1.08)
    ax1.set_xlabel("F1-Score (macro)", fontsize=11)
    ax1.set_title("Model F1-Score Comparison — All Models", fontsize=13, fontweight="bold")
    ax1.axvline(0.80, color="orange", linestyle="--", alpha=0.5, linewidth=1.5, label="0.80 threshold")
    ax1.axvline(0.88, color="purple", linestyle=":", alpha=0.4, linewidth=1.5, label="0.88 threshold")
    for bar, val, role in zip(bars, df_models["F1-Score"], df_models["Role"]):
        lbl = f"{val:.4f}" + (f"  ← {role}" if role else "")
        ax1.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                 lbl, va="center", fontsize=8.5,
                 fontweight="bold" if role else "normal",
                 color="#1e40af" if "PRIMARY" in role else ("#dc2626" if "FALLBACK" in role else "black"))
    legend_patches = [mpatches.Patch(color=c, label=l) for l, c in type_colors.items()]
    ax1.legend(handles=legend_patches + [
        plt.Line2D([0], [0], color="orange", linestyle="--", label="0.80 threshold"),
        plt.Line2D([0], [0], color="purple", linestyle=":", label="0.88 threshold"),
    ], loc="lower right", fontsize=9)
    ax1.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close()

    c_a, c_b = st.columns(2)
    c_a.success("✅ **PRIMARY: MLPClassifier (F1=0.8805)**\n\nBest sklearn model — no TF dependency. 2 hidden layers (256→128), early stopping.")
    c_b.info("⚡ **FALLBACK: SVM Linear (F1=0.7596)**\n\nFast, reliable. Activated automatically if MLP pkl is missing at boot.")

    st.markdown("---")
    st.markdown("##### 📋 Full Results Table")
    styled_df = df_models.sort_values("F1-Score", ascending=False).reset_index(drop=True)
    styled_df.index += 1
    st.dataframe(
        styled_df.style
            .highlight_max(subset=["Accuracy", "Precision", "Recall", "F1-Score"], color="#dcfce7", axis=0)
            .format({"Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}", "F1-Score": "{:.4f}"}),
        use_container_width=True
    )

with tab5:
    st.markdown("#### 📁 Batch Analysis — Upload CSV")
    st.info("Upload a CSV with a `text` column. Max 1000 rows processed.")

    uploaded = st.file_uploader("Choose CSV file", type=["csv"])
    if uploaded:
        try:
            df_batch = pd.read_csv(uploaded)
            if "text" not in df_batch.columns:
                st.error("CSV must have a column named `text`.")
            elif not model_loaded:
                st.error("Model not loaded.")
            else:
                df_batch = df_batch[df_batch["text"].notna()].head(1000)
                st.success(f"✅ Loaded {len(df_batch)} rows.")
                progress_bar = st.progress(0)
                preds, confs, levels, lows = [], [], [], []
                s_pos, s_neg, s_neu = [], [], []

                for i, txt in enumerate(df_batch["text"]):
                    r = predict(str(txt))
                    if r:
                        preds.append(r["prediction"])
                        confs.append(r["confidence"])
                        levels.append(r["conf_level"])
                        lows.append(r["low_confidence"])
                        s_pos.append(r["scores"].get("Positive", 0))
                        s_neg.append(r["scores"].get("Negative", 0))
                        s_neu.append(r["scores"].get("Neutral", 0))
                    else:
                        preds.append("Unknown")
                        confs.append(0)
                        levels.append("low")
                        lows.append(False)
                        s_pos.append(0)
                        s_neg.append(0)
                        s_neu.append(0)
                    progress_bar.progress((i + 1) / len(df_batch))

                df_batch["Sentiment"] = preds
                df_batch["Confidence (%)"] = confs
                df_batch["Conf Level"] = levels
                df_batch["Low Confidence"] = lows
                df_batch["Score_Positive"] = s_pos
                df_batch["Score_Negative"] = s_neg
                df_batch["Score_Neutral"] = s_neu

                val_counts = pd.Series(preds).value_counts()
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                c1.metric("Positive 😊", val_counts.get("Positive", 0))
                c2.metric("Negative 😟", val_counts.get("Negative", 0))
                c3.metric("Neutral 😐", val_counts.get("Neutral", 0))
                c4.metric("Avg Conf", f"{np.mean(confs):.1f}%")
                c5.metric("Low Conf ⚠️", sum(lows))
                c6.metric("Total", len(df_batch))

                col_p, col_b = st.columns(2)
                with col_p:
                    fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
                    ax_pie.pie([val_counts.get(c, 0) for c in ["Positive", "Negative", "Neutral"]],
                               labels=["Positive", "Negative", "Neutral"],
                               colors=["#16a34a", "#dc2626", "#d97706"],
                               autopct="%1.1f%%", startangle=90)
                    ax_pie.set_title("Sentiment Distribution", fontweight="bold")
                    st.pyplot(fig_pie)
                    plt.close()
                with col_b:
                    level_counts = pd.Series(levels).value_counts()
                    fig_l, ax_l = plt.subplots(figsize=(4, 4))
                    ax_l.bar(level_counts.index, level_counts.values,
                             color=["#16a34a" if l == "high" else "#d97706" if l == "moderate" else "#dc2626"
                                    for l in level_counts.index])
                    ax_l.set_title("Confidence Level Distribution", fontweight="bold")
                    ax_l.set_ylabel("Count")
                    plt.tight_layout()
                    st.pyplot(fig_l)
                    plt.close()

                st.markdown("##### Preview (first 20 rows)")
                st.dataframe(df_batch[["text", "Sentiment", "Confidence (%)", "Conf Level", "Low Confidence"]].head(20),
                             use_container_width=True)

                csv_out = df_batch.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Results CSV", data=csv_out,
                                   file_name="meetpulse_results.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.markdown("""
**Sample CSV format:**
```
text
"Great sprint planning, team is aligned on all deliverables."
"Build is broken again, deployment keeps failing."
"Reviewed backlog items and story point estimates for next sprint."
```
        """)

with tab6:
    st.markdown("#### 📈 Session Analysis History")

    if "history" not in st.session_state or not st.session_state["history"]:
        st.info("No analyses yet. Go to **Analyze** tab to start.")
    else:
        history = st.session_state["history"]
        df_hist = pd.DataFrame(history)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Analyses", len(history))
        c2.metric("Avg Confidence", f"{df_hist['confidence'].mean():.1f}%")
        c3.metric("Most Common", df_hist["sentiment"].mode()[0])
        low_pct = (df_hist["conf_level"].isin(["low", "moderate"]).sum() / len(df_hist) * 100) if len(df_hist) else 0
        c4.metric("Low/Mod Conf %", f"{low_pct:.0f}%")

        if len(history) >= 2:
            fig_h, ax_h = plt.subplots(figsize=(10, 3))
            colors_map = {"Positive": "#16a34a", "Negative": "#dc2626", "Neutral": "#d97706"}
            ax_h.bar(range(len(df_hist)), df_hist["confidence"],
                     color=[colors_map.get(s, "gray") for s in df_hist["sentiment"]], alpha=0.8)
            ax_h.axhline(70, color="green", linestyle="--", alpha=0.4, label="High threshold (70%)")
            ax_h.axhline(55, color="orange", linestyle="--", alpha=0.4, label="Moderate threshold (55%)")
            ax_h.set_xlabel("Analysis #")
            ax_h.set_ylabel("Confidence (%)")
            ax_h.set_title("Confidence by Analysis (coloured by sentiment)")
            ax_h.set_ylim(0, 110)
            patches = [mpatches.Patch(color=c, label=l) for l, c in colors_map.items()]
            ax_h.legend(handles=patches + [
                plt.Line2D([0], [0], color="green", linestyle="--", label="High threshold (70%)"),
                plt.Line2D([0], [0], color="orange", linestyle="--", label="Moderate threshold (55%)"),
            ], fontsize=9)
            plt.tight_layout()
            st.pyplot(fig_h)
            plt.close()

        st.dataframe(df_hist, use_container_width=True)
        if st.button("🗑️ Clear History"):
            st.session_state["history"] = []
            st.rerun()

st.markdown("---")
st.markdown(
    f"<small style='color:#94a3b8'>MeetPulse Streamlit v3 &mdash; "
    f"ADITYA SINGH | Roll No: 23052212 | KIIT University &mdash; "
    f"Primary: <strong>MLPClassifier</strong> (F1=0.8805) | "
    f"Fallback: <strong>SVM Linear</strong> (F1=0.7596)</small>",
    unsafe_allow_html=True
)
