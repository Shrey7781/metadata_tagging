import json
import sys
import os
from pathlib import Path

# Suppress internal transformers docstring errors
os.environ["TRANSFORMERS_VERBOSITY"] = "critical"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src import corpus, pipeline

st.set_page_config(page_title="ScriptTagger", layout="wide")
st.title("ScriptTagger — AI Metadata Tagging from Transcripts")
st.caption("Extract topics, named entities, sentiment/emotion, speakers, scenes and genres from movie screenplays.")


@st.cache_data(show_spinner=False)
def load_index() -> pd.DataFrame:
    idx = corpus.load_index()
    if idx.empty:
        idx = corpus.build_index()
    return idx


@st.cache_data(show_spinner=False)
def tag(imdb_id: str, title: str, text: str, use_transformers: bool, include_dialogue: bool):
    # Check disk cache (.json.gz or .json in outputs/ directory)
    cached = pipeline.load_cached_metadata(imdb_id) if imdb_id else None
    if cached is not None:
        has_dialogue = any(bool(s.get("dialogue")) for s in cached.get("segments", []))
        satisfies_request = not use_transformers and (not include_dialogue or has_dialogue)
        if satisfies_request or not text:
            # Either the cache already covers what was asked for, or it
            # doesn't (e.g. missing per-line dialogue / transformer output)
            # but there's no raw script text in this deployment to actually
            # regenerate it from — best effort: serve the cache as-is
            # rather than crashing on an empty-text retag.
            return cached

    meta = pipeline.tag_script(
        text,
        imdb_id=imdb_id,
        title=title,
        use_transformers=use_transformers,
        include_dialogue=include_dialogue,
    )
    pipeline.save_metadata(imdb_id or title, meta)
    return meta


def render_badges(genres):
    if genres:
        st.markdown(
            " ".join(
                f"<span style='background:#3b82f6;color:white;padding:2px 8px;"
                f"border-radius:10px;margin-right:4px'>{g['genre']} ({g['score']:.2f})</span>"
                for g in genres
            ),
            unsafe_allow_html=True,
        )


def main():
    idx = load_index()

    if "meta" not in st.session_state:
        st.session_state.meta = None
    if "meta_sig" not in st.session_state:
        st.session_state.meta_sig = None

    imdb_id, title, text, run = "", "", "", False

    with st.sidebar:
        st.header("Input")
        source = st.radio("Source", ["Corpus script", "Upload file"])
        use_transformers = st.checkbox(
            "Use transformer emotion model (slow, ~CPU)", value=False
        )
        include_dialogue = st.checkbox("Include per-line dialogue in output", value=True)

        if source == "Corpus script":
            query = st.text_input("Search title", "")
            df = idx
            if query:
                df = df[df["title"].fillna("").str.contains(query, case=False, regex=False)]
            if df.empty:
                st.warning("No matches")
                return
            label = df["title"].fillna("") + "  (" + df["imdbid"] + ")"
            choice = st.selectbox("Script", label.tolist(), index=0)
            row = df[label == choice].iloc[0]
            imdb_id = str(row["imdbid"]).zfill(7)
            title = row.get("title") or ""
            try:
                text = corpus.read_script(imdb_id)
            except KeyError:
                # Raw script text isn't available (cached-outputs-only
                # deployment, no raw dataset) — fine as long as this movie
                # already has cached tagged metadata, which tag() checks
                # for before ever needing `text`.
                text = ""

            run = st.button("Generate metadata", type="primary", width="stretch")
        else:
            up = st.file_uploader("Upload .txt screenplay", type=["txt"])
            imdb_id, title, text, run = "", "", "", False
            if up is not None:
                text = up.getvalue().decode("utf-8", errors="replace")
                title = Path(up.name).stem
                run = st.button("Generate metadata", type="primary", width="stretch")

    sig = (imdb_id, title, text, use_transformers, include_dialogue)

    if run:
        with st.spinner("Loading metadata…"):
            st.session_state.meta = tag(*sig)
        st.session_state.meta_sig = sig

    meta = st.session_state.meta
    if meta is None or st.session_state.meta_sig != sig:
        st.info("Select a script (or upload one) and click **Generate metadata**.")
        return

    st.subheader(meta.get("title") or "Untitled")
    render_badges(meta.get("genres", []))

    overall = meta.get("overall", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Scenes", overall.get("num_scenes", 0))
    c2.metric("Dialogue lines", overall.get("num_dialogue_lines", 0))
    c3.metric("Words", overall.get("num_words", 0))
    c4.metric("Overall sentiment", overall.get("sentiment", {}).get("label", "n/a"))

    tab_seg, tab_topics, tab_ents, tab_spk, tab_sent, tab_json = st.tabs(
        ["Segments", "Topics", "Entities", "Speakers", "Sentiment & Emotion", "JSON"]
    )

    with tab_seg:
        segs = meta.get("segments", [])
        if not segs:
            st.warning("No scenes found")
        else:
            labels = [
                f"S{seg['segment_id']:03d}  {seg['start']}-{seg['end']}  "
                f"{seg['heading'][:50]}"
                for seg in segs
            ]
            sel = st.selectbox("Scene", labels, index=0)
            seg = segs[labels.index(sel)]
            st.markdown(f"**{seg['heading'] or '—'}**  \n*{seg.get('location') or ''} · "
                        f"{seg.get('time_of_day') or ''}* · {seg['start']}–{seg['end']}")
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.write(f"Speakers: {', '.join(seg['speakers']) if seg.get('speakers') else '—'}")
            col_b.write("Topics: " + ", ".join(t["keyword"] for t in seg.get("topics", [])[:5]))
            
            sent_dict = seg.get("sentiment") if isinstance(seg.get("sentiment"), dict) else {}
            emo_dict = seg.get("emotion") if isinstance(seg.get("emotion"), dict) else {}
            sent_lbl = sent_dict.get("label", "n/a")
            sent_cmp = sent_dict.get("compound", 0.0)
            emo_lbl = emo_dict.get("label", "n/a")

            col_c.write(f"Sentiment: {sent_lbl} ({sent_cmp})")
            col_d.write(f"Emotion: {emo_lbl}")
            if seg.get("dialogue"):
                with st.expander("💬 Dialogue lines", expanded=True):
                    for d in seg["dialogue"]:
                        sp = d.get("speaker") or "?"
                        st.markdown(f"**{sp}:** {d['text']}")
            else:
                st.caption("ℹ️ Dialogue lines for this scene are not stored in cached output.")

    with tab_topics:
        topics = overall.get("topics", [])
        if topics:
            st.bar_chart(pd.DataFrame(topics).set_index("keyword")["score"])
        else:
            st.info("No topics extracted")

    with tab_ents:
        ents = overall.get("entities", [])
        if ents:
            st.dataframe(pd.DataFrame(ents), width="stretch")
        else:
            st.info("No entities found")

    with tab_spk:
        spk = meta.get("speakers", [])
        if spk:
            st.dataframe(pd.DataFrame(spk), width="stretch")
        else:
            st.info("No speakers detected")

    with tab_sent:
        df_sent = pd.DataFrame(
            [
                {
                    "scene": s["segment_id"],
                    "compound": (s.get("sentiment") or {}).get("compound", 0.0),
                    "label": (s.get("sentiment") or {}).get("label", "n/a"),
                }
                for s in meta.get("segments", [])
            ]
        )
        if not df_sent.empty:
            st.subheader("Scene sentiment (VADER compound)")
            st.bar_chart(df_sent.set_index("scene")["compound"])
        emo = pd.DataFrame(
            [
                {
                    "scene": s["segment_id"],
                    "emotion": (s.get("emotion") or {}).get("label", "n/a"),
                }
                for s in meta.get("segments", [])
            ]
        )
        if not emo.empty:
            st.subheader("Scene emotion labels")
            st.bar_chart(emo["emotion"].value_counts())

    with tab_json:
        st.download_button(
            "Download metadata JSON",
            data=json.dumps(meta, indent=2),
            file_name=f"{imdb_id or title}.metadata.json",
            mime="application/json",
        )
        st.json(meta, expanded=False)


if __name__ == "__main__":
    main()