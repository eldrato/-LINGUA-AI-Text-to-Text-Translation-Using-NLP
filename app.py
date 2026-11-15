"""
LINGUA AI - Intelligent Text-to-Text Translation Application.
Professional Streamlit interface integrated with MarianMT Transformer models.
"""
import site
import sys
import time
from pathlib import Path

# Ensure User Site-Packages (where pip installs packages) and Project Root are in sys.path
_user_site = site.getusersitepackages()
if _user_site and Path(_user_site).exists() and _user_site not in sys.path:
    sys.path.insert(0, _user_site)

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Safe package imports with IDE type hints and fallbacks
try:
    import torch  # type: ignore
    TORCH_VERSION = getattr(torch, "__version__", "Available")
except Exception:
    torch = None
    TORCH_VERSION = "Unavailable"

try:
    import streamlit as st  # type: ignore
except ImportError:
    sys.stderr.write(
        f"\n[ERROR] Streamlit is not installed in the active environment: {sys.executable}\n"
        "Please install requirements using: pip install -r requirements.txt\n\n"
    )
    raise

# Enable seamless execution via `python app.py` by auto-forwarding to `streamlit run`
if __name__ == "__main__" and hasattr(st, "runtime") and not st.runtime.exists():  # type: ignore
    from streamlit.web import cli as stcli  # type: ignore
    sys.argv = ["streamlit", "run", str(Path(__file__).resolve())]
    sys.exit(stcli.main())

import config
from src.model_manager import get_model_manager
from src.translation import Translator

# Page Configuration
st.set_page_config(
    page_title="LINGUA AI - Neural Machine Translation",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load Custom CSS
def load_css():
    css_file = config.CSS_PATH
    if css_file.exists():
        with open(css_file, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Language Options
LANG_OPTIONS = ["English", "Hindi"]

# Initialize Session State
if "history" not in st.session_state:
    st.session_state.history = []
if "translation_count" not in st.session_state:
    st.session_state.translation_count = 0
if "source_text_input" not in st.session_state:
    st.session_state.source_text_input = ""
if "translated_output" not in st.session_state:
    st.session_state.translated_output = ""
if "src_lang" not in st.session_state:
    st.session_state.src_lang = "English"
if "tgt_lang" not in st.session_state:
    st.session_state.tgt_lang = "Hindi"
if "src_select_box" not in st.session_state:
    st.session_state.src_select_box = "English"
if "tgt_select_box" not in st.session_state:
    st.session_state.tgt_select_box = "Hindi"
if "last_latency" not in st.session_state:
    st.session_state.last_latency = 0.0
if "success_msg" not in st.session_state:
    st.session_state.success_msg = ""

@st.cache_resource
def get_cached_translator():
    """Initializes and caches the Translator instance across Streamlit reruns."""
    return Translator()

translator = get_cached_translator()
model_manager = get_model_manager()

# Top Header
st.markdown("""
<div class="app-header">
    <div class="app-title">LINGUA AI</div>
    <div class="app-subtitle">Break Language Barriers with Intelligent Translation</div>
    <div>
        <span class="badge-tag">MarianMT Seq2Seq Transformer</span>
        <span class="badge-tag">PyTorch & Hugging Face</span>
        <span class="badge-tag">OPUS Parallel Corpus</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Determine Current Language Pair Codes
src_code = config.SUPPORTED_LANGUAGES.get(st.session_state.src_lang, "en")
tgt_code = config.SUPPORTED_LANGUAGES.get(st.session_state.tgt_lang, "hi")
current_pair_key = f"{src_code}-{tgt_code}"

# Sidebar - Settings and Diagnostics
with st.sidebar:
    st.markdown("### 🧠 Model Engine Mode")
    model_choice = st.radio(
        "Select Model Architecture:",
        options=[
            "Pretrained MarianMT (Highest Open-Domain Accuracy)",
            "Fine-Tuned Checkpoint (OPUS Corpus Domain)"
        ],
        index=0,
        help="Pretrained MarianMT offers broad vocabulary and highest fluency for open-domain sentences. Fine-Tuned Checkpoint is domain-adapted on technical / localization parallel data."
    )
    prefer_finetuned = (model_choice == "Fine-Tuned Checkpoint (OPUS Corpus Domain)")

    st.markdown("---")
    st.markdown("### ⚙️ Generation Parameters")
    num_beams = st.slider(
        "Beam Search Width",
        min_value=1,
        max_value=8,
        value=5,
        help="Number of beams for sequence search. Higher values improve translation quality and grammatical structure."
    )
    max_length = st.slider(
        "Max Sequence Length",
        min_value=32,
        max_value=256,
        value=160,
        step=16,
        help="Maximum number of tokens generated per sentence."
    )
    repetition_penalty = st.slider(
        "Repetition Penalty",
        min_value=1.0,
        max_value=1.5,
        value=1.12,
        step=0.02,
        help="Penalizes repetitive n-grams for cleaner fluency without degrading natural reduplications."
    )
    length_penalty = st.slider(
        "Length Penalty",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Penalize short (values < 1.0) or long (values > 1.0) sequences."
    )

    st.markdown("---")
    st.markdown("### 🖥️ System & Hardware")
    st.write(f"**Compute Device:** `{model_manager.device.upper()}`")
    st.write(f"**PyTorch Version:** `{TORCH_VERSION}`")

    # Update metadata based on selected engine mode
    meta = model_manager.get_metadata(current_pair_key, prefer_finetuned=prefer_finetuned) if current_pair_key in ["en-hi", "hi-en"] else None

    if meta:
        st.write(f"**Model ID:** `{Path(meta['model_id']).name}`")
        status_badge = "✅ Fine-Tuned Checkpoint" if meta["is_finetuned"] else "⚡ Pretrained MarianMT (High-Accuracy)"
        st.write(f"**Model Status:** {status_badge}")
        st.write(f"**In-Memory Cached:** `{'Yes' if meta['is_cached'] else 'Pending first run'}`")
    else:
        st.write(f"**Model Status:** Identity Mapping (`{st.session_state.src_lang} → {st.session_state.tgt_lang}`)")

    st.markdown("---")
    if st.button("🗑️ Clear Model Memory Cache", use_container_width=True):
        model_manager.clear_cache()
        st.success("Model memory cache cleared!")

# Dashboard Statistics Section
col_s1, col_s2, col_s3, col_s4 = st.columns(4)

with col_s1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{st.session_state.translation_count}</div>
        <div class="metric-label">Translations This Session</div>
    </div>
    """, unsafe_allow_html=True)

with col_s2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{st.session_state.src_lang} → {st.session_state.tgt_lang}</div>
        <div class="metric-label">Selected Language Pair</div>
    </div>
    """, unsafe_allow_html=True)

with col_s3:
    if meta and meta.get("model_id"):
        model_display_name = Path(meta['model_id']).name
    else:
        model_display_name = "MarianMT"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="font-size: 1.1rem; overflow: hidden; text-overflow: ellipsis;">{model_display_name}</div>
        <div class="metric-label">Active Model Identifier</div>
    </div>
    """, unsafe_allow_html=True)

with col_s4:
    if meta:
        status_label = "Fine-Tuned" if meta["is_finetuned"] else "Pretrained"
    else:
        status_label = "Direct"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{status_label} ({model_manager.device.upper()})</div>
        <div class="metric-label">Model Status & Device</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Language Selection Bar with Swap Functionality
col_lang1, col_swap, col_lang2 = st.columns([5, 1, 5])

def on_src_change():
    st.session_state.src_lang = st.session_state.src_select_box

def on_tgt_change():
    st.session_state.tgt_lang = st.session_state.tgt_select_box

with col_lang1:
    st.selectbox(
        "Source Language",
        options=LANG_OPTIONS,
        index=LANG_OPTIONS.index(st.session_state.src_lang),
        key="src_select_box",
        on_change=on_src_change
    )

with col_swap:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    if st.button("⇄", help="Swap Source and Target Languages", use_container_width=True):
        # Swap languages
        old_src = st.session_state.src_lang
        old_tgt = st.session_state.tgt_lang
        st.session_state.src_lang = old_tgt
        st.session_state.tgt_lang = old_src
        st.session_state.src_select_box = old_tgt
        st.session_state.tgt_select_box = old_src
        # Swap input and output texts
        old_in = st.session_state.source_text_input
        old_out = st.session_state.translated_output
        st.session_state.source_text_input = old_out
        st.session_state.translated_output = old_in
        st.session_state.success_msg = ""
        st.rerun()

with col_lang2:
    st.selectbox(
        "Target Language",
        options=LANG_OPTIONS,
        index=LANG_OPTIONS.index(st.session_state.tgt_lang),
        key="tgt_select_box",
        on_change=on_tgt_change
    )

# Main Translation Panels
col_left, col_right = st.columns(2)

with col_left:
    st.markdown(f"**Enter text in {st.session_state.src_lang}:**")
    input_text = st.text_area(
        label="Input Text Area",
        value=st.session_state.source_text_input,
        placeholder="Type or paste sentences/paragraphs here to translate...",
        height=220,
        label_visibility="collapsed",
        key="main_input_area"
    )
    # Synchronize session state with user typing
    st.session_state.source_text_input = input_text

    # Character and word count
    char_len = len(input_text)
    word_len = len(input_text.split()) if input_text.strip() else 0
    st.markdown(f"<div class='char-counter'>Characters: {char_len} | Words: {word_len}</div>", unsafe_allow_html=True)

    col_btn1, col_btn2, col_btn3 = st.columns([3, 2, 2])
    with col_btn1:
        translate_clicked = st.button("🚀 Translate Text", type="primary", use_container_width=True)
    with col_btn2:
        if st.button("🧹 Clear Input", use_container_width=True):
            st.session_state.source_text_input = ""
            st.session_state.translated_output = ""
            st.session_state.success_msg = ""
            st.rerun()
    with col_btn3:
        if st.button("💡 Quick Sample", use_container_width=True):
            if st.session_state.src_lang == "English":
                st.session_state.source_text_input = "Natural Language Processing allows machines to understand and translate human languages with great accuracy."
            else:
                st.session_state.source_text_input = "प्राकृतिक भाषा प्रसंस्करण मशीनों को मानवीय भाषाओं को समझने और सटीक अनुवाद करने में सक्षम बनाता है।"
            st.session_state.success_msg = ""
            st.rerun()

with col_right:
    st.markdown(f"**Translated text in {st.session_state.tgt_lang}:**")
    
    # Display success notification if present
    if st.session_state.success_msg:
        st.success(st.session_state.success_msg)

    # Display translated output
    output_text = st.session_state.translated_output
    if output_text:
        st.text_area(
            label="Translated Text Area",
            value=output_text,
            height=220,
            disabled=True,
            label_visibility="collapsed",
            key="main_output_area"
        )
    else:
        st.markdown(
            """<div class="output-box" style="display: flex; align-items: center; justify-content: center; color: #64748b;">
                Translation will appear here after clicking 'Translate Text'.
            </div>""",
            unsafe_allow_html=True
        )

    # Action buttons for right panel
    col_out1, col_out2 = st.columns(2)
    with col_out1:
        if output_text:
            st.download_button(
                label="📥 Download (.txt)",
                data=output_text,
                file_name=f"translation_{src_code}_to_{tgt_code}.txt",
                mime="text/plain",
                use_container_width=True
            )
    with col_out2:
        if output_text:
            st.code(output_text, language=None)

# Translation Processing Logic
if translate_clicked:
    if not input_text or not input_text.strip():
        st.warning("⚠️ Please enter some text to translate before proceeding.")
    else:
        with st.spinner("Translating text using MarianMT Transformer..."):
            try:
                res = translator.translate(
                    text=input_text,
                    source_lang=st.session_state.src_lang,
                    target_lang=st.session_state.tgt_lang,
                    num_beams=num_beams,
                    max_length=max_length,
                    length_penalty=length_penalty,
                    repetition_penalty=repetition_penalty,
                    prefer_finetuned=prefer_finetuned
                )

                st.session_state.translated_output = res["translated_text"]
                st.session_state.translation_count += 1
                st.session_state.last_latency = res["latency_sec"]
                st.session_state.success_msg = f"Translation completed in {res['latency_sec']} seconds on {res['device'].upper()}!"

                # Append to history
                st.session_state.history.insert(0, {
                    "timestamp": time.strftime("%H:%M:%S"),
                    "source_lang": res["source_lang"],
                    "target_lang": res["target_lang"],
                    "source_text": input_text,
                    "translated_text": res["translated_text"],
                    "latency": res["latency_sec"],
                    "model_id": res["model_id"],
                })

                st.rerun()

            except Exception as e:
                st.error(f"❌ Translation Error: {e!s}")

# Bottom Tabs: Translation History, Model Insights, Dataset Overview
tab_history, tab_eval, tab_dataset = st.tabs(["📜 Translation History", "📊 Model Evaluation & Benchmarks", "📁 Dataset Architecture"])

with tab_history:
    st.markdown("### Session Translation History")
    if not st.session_state.history:
        st.info("No translations performed in this session yet.")
    else:
        col_hist_action, _ = st.columns([2, 8])
        with col_hist_action:
            if st.button("Clear Translation History"):
                st.session_state.history = []
                st.rerun()

        for item in st.session_state.history:
            st.markdown(f"""
            <div class="history-item">
                <div class="history-meta">
                    <span>{item['source_lang'].upper()} → {item['target_lang'].upper()} | Model: {item['model_id']}</span>
                    <span>{item['timestamp']} ({item['latency']}s)</span>
                </div>
                <div class="history-text"><strong>Original:</strong> {item['source_text']}</div>
                <div class="history-trans"><strong>Translated:</strong> {item['translated_text']}</div>
            </div>
            """, unsafe_allow_html=True)

with tab_eval:
    st.markdown("### Real-World Evaluation Metrics (SacreBLEU)")
    st.write("Translation performance evaluated on held-out test datasets using standard corpus BLEU:")

    # Check for existing evaluation results
    eval_en_hi = config.EVALUATION_DIR / "eval_results_en_hi.json"
    eval_hi_en = config.EVALUATION_DIR / "eval_results_hi_en.json"

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.markdown("#### English → Hindi (en-hi)")
        if eval_en_hi.exists():
            import json
            with open(eval_en_hi, "r", encoding="utf-8") as f:
                data_en_hi = json.load(f)
            st.metric("SacreBLEU Corpus Score", f"{data_en_hi.get('sacrebleu_corpus_score', 'N/A')}")
            st.metric("Avg Sentence BLEU", f"{data_en_hi.get('average_sentence_bleu', 'N/A')}")
            st.write(f"**Test Samples Evaluated:** {data_en_hi.get('num_test_examples', 'N/A')}")
            st.write(f"**Avg Latency per Sentence:** {data_en_hi.get('avg_inference_time_per_sentence_sec', 'N/A')}s")
            st.write(f"**Model Type:** {'Fine-Tuned Checkpoint' if data_en_hi.get('is_finetuned') else 'Pretrained MarianMT'}")
        else:
            st.info("Evaluation results for English → Hindi not yet generated. Run `python src/evaluate.py --lang_pair en-hi` to generate metrics.")

    with col_e2:
        st.markdown("#### Hindi → English (hi-en)")
        if eval_hi_en.exists():
            import json
            with open(eval_hi_en, "r", encoding="utf-8") as f:
                data_hi_en = json.load(f)
            st.metric("SacreBLEU Corpus Score", f"{data_hi_en.get('sacrebleu_corpus_score', 'N/A')}")
            st.metric("Avg Sentence BLEU", f"{data_hi_en.get('average_sentence_bleu', 'N/A')}")
            st.write(f"**Test Samples Evaluated:** {data_hi_en.get('num_test_examples', 'N/A')}")
            st.write(f"**Avg Latency per Sentence:** {data_hi_en.get('avg_inference_time_per_sentence_sec', 'N/A')}s")
            st.write(f"**Model Type:** {'Fine-Tuned Checkpoint' if data_hi_en.get('is_finetuned') else 'Pretrained MarianMT'}")
        else:
            st.info("Evaluation results for Hindi → English not yet generated. Run `python src/evaluate.py --lang_pair hi-en` to generate metrics.")

    # Show training loss curves if available
    loss_curve_en_hi = config.TRAINING_LOGS_DIR / "loss_curve_en_hi.png"
    if loss_curve_en_hi.exists():
        st.markdown("#### Fine-Tuning Loss Curve (en-hi)")
        try:
            st.image(str(loss_curve_en_hi), caption="Training & Validation Loss progression across epochs", width="stretch")
        except Exception:
            st.image(str(loss_curve_en_hi), caption="Training & Validation Loss progression across epochs")

with tab_dataset:
    st.markdown("### Dataset Architecture & Preprocessing Pipeline")
    st.markdown("""
    - **Corpus Sources:** OPUS Parallel Translation Datasets (IIT Bombay English-Hindi Corpus, OPUS Books, Tatoeba).
    - **Unicode Normalization:** Standardized NFC representation keeping Hindi Devanagari script (`\\u0900-\\u097F`), Virama, Nukta, Matras, and punctuation (`।`, `॥`) intact.
    - **Split Ratio:** 80% Training, 10% Validation, 10% Testing (reproducible seed `42`).
    - **Sequence Constraints:** Cleaned for minimum 1 character, maximum 500 characters, tokenized to 128 max length.
    """)

    # Check if processed datasets exist
    train_file = config.PROCESSED_DATASET_DIR / "opus_en_hi_train.csv"
    if train_file.exists():
        import pandas as pd
        sample_df = pd.read_csv(train_file, nrows=5)
        st.markdown("#### Processed Training Split Preview")
        try:
            st.dataframe(sample_df, width="stretch")
        except Exception:
            st.dataframe(sample_df)
    else:
        st.info("Processed dataset files will appear here once dataset loading and preprocessing is executed.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 1rem 0;">
    LINGUA AI &bull; Built with PyTorch, Hugging Face Transformers & MarianMT &bull; Academic NLP Research Project
</div>
""", unsafe_allow_html=True)
