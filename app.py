import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient
from PIL import Image

# ── Constants ────────────────────────────────────────────────────────────────

MODEL_ID = "dhruv-kothari-yrwsq/casting-defect-tcdc-demo-1-resnet18-t1"
UNIVERSE_URL = "https://universe.roboflow.com/dhruv-kothari-yrwsq/casting-defect-tcdc-demo"
GITHUB_URL = "https://github.com/dhruvk23/casting-demo-app"

SAMPLES_DIR = Path(__file__).parent / "samples"
CONFUSION_MATRIX = Path(__file__).parent / "confusion_matrix.png"

DEFECT_SAMPLES = ["def-1.jpg", "def-2.jpg", "def-3.jpg"]
OK_SAMPLES = [
    "ok-1.jpg",
    "ok-2.jpg",
    "cast_ok_0_638_jpeg.rf.0d7a513386260b3cf2033e434b208381.jpg",
]

# ── API key ───────────────────────────────────────────────────────────────────

def get_api_key():
    try:
        return st.secrets["ROBOFLOW_API_KEY"]
    except Exception:
        load_dotenv(Path(__file__).parent / ".env")
        key = os.environ.get("ROBOFLOW_API_KEY", "")
        if not key or key == "PASTE_KEY_HERE":
            st.error("ROBOFLOW_API_KEY is not configured.")
            st.stop()
        return key


@st.cache_resource
def get_client():
    return InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=get_api_key(),
    )


def run_inference(image_path: str):
    client = get_client()
    result = client.infer(image_path, model_id=MODEL_ID)
    predicted = result["top"]
    confidence = float(result["confidence"])
    return predicted, confidence


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Casting Inspection — TCDC Demo",
    page_icon="🏭",
    layout="wide",
)

# ── Global styles ─────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* Verdict banners */
    .verdict-pass {
        background: #0a7c2f;
        color: #ffffff;
        font-size: 2.6rem;
        font-weight: 900;
        letter-spacing: 0.12em;
        text-align: center;
        padding: 1.1rem 2rem;
        border-radius: 8px;
        border: 3px solid #05551f;
        margin: 1rem 0 0.4rem 0;
        font-family: monospace;
    }
    .verdict-fail {
        background: #b81c1c;
        color: #ffffff;
        font-size: 2.6rem;
        font-weight: 900;
        letter-spacing: 0.12em;
        text-align: center;
        padding: 1.1rem 2rem;
        border-radius: 8px;
        border: 3px solid #7a0f0f;
        margin: 1rem 0 0.4rem 0;
        font-family: monospace;
    }
    .confidence-line {
        text-align: center;
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 0.4rem;
    }
    .inference-caption {
        text-align: center;
        font-size: 0.78rem;
        color: #999;
        margin-top: 0.2rem;
    }
    /* Sample thumbnails */
    div[data-testid="column"] button {
        width: 100%;
        padding: 0;
        border: none;
        background: none;
    }
    /* Footer */
    .footer {
        text-align: center;
        font-size: 0.8rem;
        color: #888;
        padding: 2rem 0 1rem 0;
        border-top: 1px solid #e0e0e0;
        margin-top: 3rem;
    }
    .footer a { color: #888; }
    /* Stat bar */
    .stat-bar {
        background: #f0f4ff;
        border-left: 4px solid #3a6dc9;
        padding: 0.65rem 1rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.9rem;
        color: #2a2a2a;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────

st.title("🏭 Automated Casting Inspection — Demo for Twin City Die Castings")
st.markdown(
    f"Built on [Roboflow]({UNIVERSE_URL}), trained in an afternoon on ~1,300 public images "
    "of aluminum castings analogous to TCDC's parts."
)

# ── Validation stat bar ───────────────────────────────────────────────────────

st.markdown(
    "<div class='stat-bar'>"
    "✅ <strong>Validated at 100% accuracy</strong> on a 130-image held-out test set "
    "(87 defective, 43 good) — zero missed defects — on this controlled public dataset. "
    "Production environments are harder, which is what a pilot would test."
    "</div>",
    unsafe_allow_html=True,
)

with st.expander("Validation details — confusion matrix"):
    if CONFUSION_MATRIX.exists():
        st.image(str(CONFUSION_MATRIX), caption="Confusion matrix — held-out test set", width=480)
    else:
        st.info("Run evaluate.py to generate confusion_matrix.png.")

st.divider()

# ── Main panel ────────────────────────────────────────────────────────────────

left_col, right_col = st.columns([3, 2], gap="large")

with left_col:
    st.subheader("Select a sample or upload your own image")

    # ── Sample grid ──────────────────────────────────────────────────────────
    st.markdown("**Sample defective parts**")
    def_cols = st.columns(3)
    for col, fname in zip(def_cols, DEFECT_SAMPLES):
        img_path = SAMPLES_DIR / fname
        with col:
            st.image(str(img_path), use_container_width=True)
            if st.button("Run ▶", key=f"btn_{fname}"):
                st.session_state["selected_image"] = str(img_path)
                st.session_state["selected_label"] = "defect sample"

    st.markdown("**Sample good parts**")
    ok_cols = st.columns(3)
    for col, fname in zip(ok_cols, OK_SAMPLES):
        img_path = SAMPLES_DIR / fname
        with col:
            st.image(str(img_path), use_container_width=True)
            if st.button("Run ▶", key=f"btn_{fname}"):
                st.session_state["selected_image"] = str(img_path)
                st.session_state["selected_label"] = "good sample"

    st.markdown("---")
    st.markdown("**Or upload your own casting image**")
    uploaded = st.file_uploader(
        label="Drag & drop or click to browse",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed",
    )
    if uploaded:
        suffix = Path(uploaded.name).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            st.session_state["selected_image"] = tmp.name
            st.session_state["selected_label"] = uploaded.name

# ── Verdict panel ─────────────────────────────────────────────────────────────

with right_col:
    st.subheader("Inspection result")

    if "selected_image" not in st.session_state:
        st.info("Select a sample image or upload one to run inspection.")
    else:
        img_path = st.session_state["selected_image"]
        label = st.session_state.get("selected_label", "")

        st.image(img_path, caption=label, use_container_width=True)

        with st.spinner("Running model…"):
            try:
                predicted, confidence = run_inference(img_path)
            except Exception as exc:
                st.error(f"Inference failed: {exc}")
                st.stop()

        if predicted == "ok":
            st.markdown("<div class='verdict-pass'>✔ PASS</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='verdict-fail'>✖ FAIL — DEFECT DETECTED</div>", unsafe_allow_html=True)

        pct = confidence * 100
        st.markdown(
            f"<div class='confidence-line'>Model confidence: <strong>{pct:.1f}%</strong></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='inference-caption'>Inference by Roboflow serverless API</div>",
            unsafe_allow_html=True,
        )

# ── Sidebar: ROI calculator ───────────────────────────────────────────────────

with st.sidebar:
    st.header("💰 Scrap-Cost ROI Calculator")
    st.caption("Illustrative assumptions — for demo purposes only.")

    annual_volume = st.number_input(
        "Annual good-part volume",
        min_value=0,
        value=2_000_000,
        step=100_000,
        format="%d",
    )
    ppm = st.number_input(
        "Current escaped-defect rate (PPM)",
        min_value=0,
        value=500,
        step=50,
        format="%d",
    )
    cost_per_incident = st.number_input(
        "Avg cost per escaped-defect incident (chargebacks / sorting / 8D)",
        min_value=0,
        value=15_000,
        step=1_000,
        format="%d",
    )
    catch_rate = st.slider(
        "Expected automated-inspection catch rate (%)",
        min_value=0,
        max_value=100,
        value=90,
        step=1,
    )
    cost_per_casting = st.number_input(
        "Avg cost per casting (context only, $ )",
        min_value=0.0,
        value=8.0,
        step=0.5,
        format="%.2f",
    )

    st.divider()

    escaped_per_year = annual_volume * (ppm / 1_000_000)
    incidents_prevented = escaped_per_year * (catch_rate / 100)
    annual_savings = incidents_prevented * cost_per_incident

    st.markdown("**Formula**")
    st.markdown(
        f"""
```
Escaped defects / yr  = {annual_volume:,} × {ppm} / 1,000,000
                      = {escaped_per_year:,.0f} parts

Incidents prevented   = {escaped_per_year:,.0f} × {catch_rate}%
                      = {incidents_prevented:,.0f}

Annual savings        = {incidents_prevented:,.0f} × ${cost_per_incident:,}
```
"""
    )

    st.metric(
        label="Estimated annual savings",
        value=f"${annual_savings:,.0f}",
    )

    st.caption(
        "All figures are illustrative assumptions for demo purposes. "
        "Actual results depend on defect rates, incident costs, and "
        "deployment configuration specific to your facility."
    )

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown(
    f"<div class='footer'>"
    f"Built by Dhruv Kothari — portfolio demo for Roboflow SDR application &nbsp;|&nbsp; "
    f"<a href='{UNIVERSE_URL}' target='_blank'>Roboflow Universe project</a> &nbsp;|&nbsp; "
    f"<a href='{GITHUB_URL}' target='_blank'>GitHub repo</a>"
    f"</div>",
    unsafe_allow_html=True,
)
