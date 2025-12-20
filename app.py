import re
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import html
import streamlit.components.v1 as components

DATA_DIR = Path("data")
SUMMARIES_PATH = DATA_DIR / "summaries.csv"

st.set_page_config(
    page_title="IUI Demo – DiSCo Summaries",
    layout="wide",
)

if "selected_domain" not in st.session_state:
    st.session_state.selected_domain = None

if "selected_accommodation" not in st.session_state:
    st.session_state.selected_accommodation = None

# ---------------------------
# Utilities
# ---------------------------

DOMAIN_SIG_RE = re.compile(r"^(?P<domain>.+)_signatures\.csv$")

def parse_element(element: str):
    """
    element format: {topic_name}_{sentiment}
    sentiment assumed in {positive, negative, neutral} but we keep it generic.
    """
    if not isinstance(element, str) or "_" not in element:
        return element, ""
    topic, sentiment = element.rsplit("_", 1)
    return topic.replace("__", "_"), sentiment  # small safety if topic had underscores encoded

def format_topic(topic: str) -> str:
    # Human-readable label
    return topic.replace("_", " ").strip().title()

def sentiment_badge(sent: str) -> str:
    # simple text indicator; can be replaced with icons later
    s = (sent or "").lower()
    if s == "positive":
        return "positive"
    if s == "negative":
        return "negative"
    if s == "neutral":
        return "neutral"
    if s == "mixed":
        return "mixed"
    return s if s else "—"

@st.cache_data(show_spinner=False)
def list_domains(data_dir: Path) -> list[str]:
    domains = []
    if not data_dir.exists():
        return domains
    for p in data_dir.glob("*_signatures.csv"):
        m = DOMAIN_SIG_RE.match(p.name)
        if m:
            domains.append(m.group("domain"))
    return sorted(domains)

def sentiment_icon(sentiment: str) -> str:
    s = (sentiment or "").lower()
    if s == "positive":
        return "👍"
    if s == "negative":
        return "👎"
    if s == "neutral":
        return "~"
    return "~"

def render_topic_bubbles(
    topics_df: pd.DataFrame,
    max_per_row: int = 5,
    row_spacing_px: int = 16,
    bottom_spacing_px: int = 0,
    min_height_px: int = 52,
):
    if topics_df.empty:
        st.info("No topics to display.")
        return

    rows = [
        topics_df.iloc[i:i + max_per_row]
        for i in range(0, len(topics_df), max_per_row)
    ]

    for row_idx, row_df in enumerate(rows):
        bubbles_html = ""

        for _, r in row_df.iterrows():
            icon = sentiment_icon(r["sentiment_label"])
            topic = html.escape(str(r["topic_label"]))  # IMPORTANT: prevents broken HTML

            bubbles_html += f"""
              <div class="bubble">
                <div class="icon">{icon}</div>
                <div class="label">{topic}</div>
              </div>
            """

        # placeholders so the row always has 5 fixed-width slots
        empty_slots = max_per_row - len(row_df)
        for _ in range(empty_slots):
            bubbles_html += """<div class="placeholder"></div>"""

        row_html = f"""
        <div class="row">
          {bubbles_html}
        </div>
        """

        css = f"""
        <style>
          .row {{
              display: flex;
              gap: 12px;
              align-items: stretch;
              font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                           Roboto, Oxygen, Ubuntu, Cantarell,
                           "Helvetica Neue", Arial, sans-serif;
            }}
          .bubble, .placeholder {{
            flex: 1 1 0;
          }}
          .bubble {{
            border: 1px solid #ddd;
            border-radius: 16px;
            padding: 10px;
            background: #fafafa;
            text-align: center;

            display: flex;
            flex-direction: column;
            justify-content: center;

            min-height: {min_height_px}px;
            box-sizing: border-box;
          }}
          .icon {{
            font-size: 22px;
            margin-bottom: 4px;
            line-height: 1;
          }}
          .label {{
            font-size: 0.875rem;
            line-height: 1.2;
            word-break: break-word;
          }}
          .placeholder {{
            /* keep the 5-column grid without showing an empty bubble */
          }}
        </style>
        """

        # Render this row as a self-contained HTML fragment
        components.html(css + row_html, height=min_height_px + 40)

        # Space between rows
        if row_idx < len(rows) - 1:
            st.markdown(f"<div style='height:{row_spacing_px}px'></div>", unsafe_allow_html=True)

    # Space after the whole grid
    st.markdown(f"<div style='height:{bottom_spacing_px}px'></div>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_domain_freqs(domain: str) -> pd.DataFrame:
    path = DATA_DIR / f"{domain}_freqs.csv"
    df = pd.read_csv(path)

    required = {"accommodation_id", "element", "frequency"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")

    df["accommodation_id"] = df["accommodation_id"].astype(str)
    df["element"] = df["element"].astype(str)
    df["frequency"] = pd.to_numeric(df["frequency"], errors="coerce").fillna(0.0)

    return df

@st.cache_data(show_spinner=False)
def load_summaries(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["accommodation_id", "baseline_summary", "expectations_aware_summary"])
    df = pd.read_csv(path)
    # Ensure required columns exist
    for c in ["accommodation_id", "baseline_summary", "expectations_aware_summary"]:
        if c not in df.columns:
            raise ValueError(f"missing column {c}")
    # Normalize accommodation_id type to string for consistent joins
    df["accommodation_id"] = df["accommodation_id"].astype(str)
    return df

summ_df = load_summaries(SUMMARIES_PATH)

@st.cache_data(show_spinner=False)
def load_domain_signatures(domain: str) -> pd.DataFrame:
    path = DATA_DIR / f"{domain}_signatures.csv"
    df = pd.read_csv(path)
    # first column is accommodation_id (per your spec)
    if df.shape[1] < 2:
        raise ValueError(f"{path.name} must have accommodation_id + topic columns.")
    df = df.copy()
    df.iloc[:, 0] = df.iloc[:, 0].astype(str)
    df.rename(columns={df.columns[0]: "accommodation_id"}, inplace=True)
    return df

@st.cache_data(show_spinner=False)
def load_domain_dvr(domain: str) -> pd.DataFrame:
    path = DATA_DIR / f"{domain}_dvr.csv"
    df = pd.read_csv(path)
    if "element" not in df.columns or "global_weight" not in df.columns:
        raise ValueError(f"{path.name} must have columns: element, global_weight.")
    df = df.copy()
    df["element"] = df["element"].astype(str)
    df["global_weight"] = pd.to_numeric(df["global_weight"], errors="coerce").fillna(0.0)
    return df.sort_values("global_weight", ascending=False)

def top_k_domain_topics(dvr_df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    top = dvr_df.head(k).copy()
    top["topic"], top["sentiment"] = zip(*top["element"].map(parse_element))
    top["topic_label"] = top["topic"].map(format_topic)
    top["sentiment_label"] = top["sentiment"].map(sentiment_badge)
    top = top[["topic_label", "sentiment_label", "global_weight"]]
    top.rename(columns={"global_weight": "weight"}, inplace=True)
    return top

def top_k_property_topics(
    freqs_df: pd.DataFrame,
    accommodation_id: str,
    k: int = 10
) -> pd.DataFrame:
    df = freqs_df[freqs_df["accommodation_id"] == accommodation_id]

    if df.empty:
        return pd.DataFrame(
            columns=["topic_label", "sentiment_label", "frequency"]
        )

    top = (
        df.sort_values("frequency", ascending=False)
          .head(k)
          .copy()
    )

    top["topic"], top["sentiment"] = zip(
        *top["element"].map(parse_element)
    )
    top["topic_label"] = top["topic"].map(format_topic)
    top["sentiment_label"] = top["sentiment"].map(sentiment_badge)

    return top[["topic_label", "sentiment_label", "frequency"]]

def top_k_diverging_topics(signatures_df: pd.DataFrame, accommodation_id: str, k: int = 10) -> pd.DataFrame:
    row = signatures_df.loc[signatures_df["accommodation_id"] == accommodation_id]
    if row.empty:
        return pd.DataFrame(columns=["topic_label", "sentiment_label", "divergence", "direction"])
    row = row.iloc[0]

    # topic columns are everything except accommodation_id
    topic_cols = [c for c in signatures_df.columns if c != "accommodation_id"]
    values = pd.to_numeric(row[topic_cols], errors="coerce")

    # handle all-NaN cases
    if values.isna().all():
        return pd.DataFrame(columns=["topic_label", "sentiment_label", "divergence", "direction"])

    # pick top by absolute divergence
    abs_sorted = values.abs().sort_values(ascending=False)
    top_cols = abs_sorted.head(k).index.tolist()

    top_vals = values[top_cols].fillna(0.0)
    out = pd.DataFrame({
        "element": top_cols,
        "divergence": top_vals.values,
    })

    out["topic"], out["sentiment"] = zip(*out["element"].map(parse_element))
    out["topic_label"] = out["topic"].map(format_topic)
    out["sentiment_label"] = out["sentiment"].map(sentiment_badge)

    out["direction"] = np.where(
        out["divergence"] < 0,
        "missing vs domain",
        "overrepresented vs domain"
    )
    out["strength"] = out["divergence"].abs()

    # sort by strength desc
    out = out.sort_values("strength", ascending=False)

    return out[["topic_label", "sentiment_label", "divergence", "direction"]]

# ---------------------------
# UI
# ---------------------------

st.title("DiSCo Review Summaries (IUI Demo)")
st.markdown(
    """
    <div style="margin-bottom: 16px;">
        👍 Positive sentiment &nbsp;&nbsp;
        👎 Negative sentiment &nbsp;&nbsp;
        ~ Neutral
    </div>
    """,
    unsafe_allow_html=True
)
with st.sidebar:
    st.header("Selection")

    domains = list_domains(DATA_DIR)
    if not domains:
        st.error("No domains found. Put domain files under data/ as {domain}_signatures.csv and {domain}_dvr.csv.")
        st.stop()

    domain = st.selectbox(
        "Accommodation domain",
        domains,
        key="domain_select"
    )

    # Load domain-specific data
    sig_df = load_domain_signatures(domain)
    dvr_df = load_domain_dvr(domain)
    freqs_df = load_domain_freqs(domain)

    # Accommodation IDs from domain signatures
    domain_ids = set(
        sig_df["accommodation_id"]
        .astype(str)
        .unique()
    )

    # Accommodation IDs that actually have summaries
    summary_ids = set(
        summ_df["accommodation_id"]
        .astype(str)
        .unique()
    )

    # Intersection: ONLY IDs valid for this domain AND with summaries
    domain_accommodation_ids = sorted(domain_ids.intersection(summary_ids))

    if st.session_state.selected_domain != domain:
        st.session_state.selected_domain = domain
        st.session_state.selected_accommodation = domain_accommodation_ids[0]

    accommodation_id = st.selectbox(
        "Accommodation",
        domain_accommodation_ids,
        key="selected_accommodation",
        disabled=(len(domain_accommodation_ids) == 0)
    )

    st.divider()
    k_topics = st.slider("Top-K topics", min_value=5, max_value=20, value=5, step=1)

# Load summaries
summ_row = summ_df.loc[summ_df["accommodation_id"] == str(accommodation_id)]
baseline_text = summ_row["baseline_summary"].iloc[0] if not summ_row.empty else ""
ea_text = summ_row["expectations_aware_summary"].iloc[0] if not summ_row.empty else ""

# Compute tables
domain_top = top_k_domain_topics(dvr_df, k=k_topics)
diverging_top = top_k_diverging_topics(sig_df, str(accommodation_id), k=k_topics)
property_top = top_k_property_topics(freqs_df, str(accommodation_id), k=k_topics)

# ---------------------------
# Layout: side-by-side summaries
# ---------------------------

left, right = st.columns(2, gap="large")

with left:
    st.subheader("Standard summary")
    st.caption("What guests talk about in this accommodation")

    st.markdown("**Most mentioned topics in this accommodation**")
    # st.caption(
    #     "Topics most frequently mentioned in this accommodation’s reviews."
    # )
    render_topic_bubbles(property_top)

    st.markdown("**Textual summary**")
    st.write(baseline_text)

over_df = diverging_top[diverging_top["divergence"] > 0]
under_df = diverging_top[diverging_top["divergence"] < 0]
over_df = over_df.sort_values("divergence", ascending=False)
under_df = under_df.sort_values("divergence")

with right:
    st.subheader("DiSCo summary")
    st.caption("What stands out / absent compared to other accommodations from the same domain")

    st.markdown("**Most mentioned topics in other accommodations from the same domain**")
    st.caption(
        "These topics reflect what guests typically mention in other accommodations from the same domain, and serve as a reference point."
    )
    render_topic_bubbles(domain_top, max_per_row=5)

    st.markdown(
        """
        <div style="font-weight: 600;">
            Topics mentioned more than expected in this accommodation
            <span style="color: #888; font-weight: 400;">
                (overrepresented)
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )
    render_topic_bubbles(over_df)

    st.markdown(
        """
        <div style="font-weight: 600;">
            Topics mentioned less than expected in this accommodation
            <span style="color: #888; font-weight: 400;">
                (underrepresented)
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )
    render_topic_bubbles(under_df)

    st.markdown("**Textual summary**")
    st.write(ea_text)

# ---------------------------
# Footer / explanation (good for demos)
# ---------------------------
with st.expander("How to interpret divergence"):
    st.write(
        """
- **Positive divergence**: the topic is **overrepresented** in this accommodation’s reviews compared to the domain.
- **Negative divergence**: the topic is **missing/underrepresented** in this accommodation’s reviews compared to the domain.
- We rank diverging topics by **absolute divergence magnitude** (strongest deviations first).
        """.strip()
    )
