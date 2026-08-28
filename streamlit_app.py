"""Amazon bestseller analytics portfolio app."""

import hashlib

import streamlit as st

from src.data_io import load_default_data, load_uploaded_data, prepare_data
from src.data_pipeline import DataValidationError
from src.ui import render_global_filters, reset_filters


st.set_page_config(
    page_title="Amazon bestseller analytics",
    page_icon=":material/menu_book:",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.subheader("Data source")
    uploaded_file = st.file_uploader(
        "Upload a compatible CSV",
        type=["csv"],
        help="Expected fields: Name, Author, User Rating, Reviews, Price, Year, Genre.",
    )
    st.caption("The bundled 2009–2019 file is used when no upload is provided.")

try:
    if uploaded_file is None:
        raw_data = load_default_data()
        source_label = "Bundled Amazon bestseller CSV"
        source_token = "bundled-amazon-bestsellers"
    else:
        uploaded_bytes = uploaded_file.getvalue()
        raw_data = load_uploaded_data(uploaded_bytes)
        source_label = uploaded_file.name
        source_token = hashlib.sha256(uploaded_bytes).hexdigest()

    clean_result = prepare_data(raw_data)
except (DataValidationError, UnicodeDecodeError, ValueError) as exc:
    st.error(str(exc), icon=":material/error:")
    st.stop()

if st.session_state.get("data_source_token") != source_token:
    reset_filters()
    st.session_state["data_source_token"] = source_token

with st.sidebar:
    filtered_data = render_global_filters(clean_result.data)
    st.caption(
        f"Showing {len(filtered_data):,} of {len(clean_result.data):,} list records."
    )

st.session_state["raw_data"] = raw_data
st.session_state["clean_result"] = clean_result
st.session_state["filtered_data"] = filtered_data
st.session_state["source_label"] = source_label

page = st.navigation(
    [
        st.Page(
            "app_pages/overview.py",
            title="Executive overview",
            icon=":material/dashboard:",
        ),
        st.Page(
            "app_pages/data_quality.py",
            title="Data quality",
            icon=":material/fact_check:",
        ),
        st.Page(
            "app_pages/exploratory_analysis.py",
            title="Exploratory analysis",
            icon=":material/query_stats:",
        ),
        st.Page(
            "app_pages/insights.py",
            title="Insights & actions",
            icon=":material/lightbulb:",
        ),
    ],
    position="top",
)

st.title(f"{page.icon} {page.title}")
st.caption(
    "Analysis of bestseller list appearances—not units sold or revenue. "
    f"Source: {source_label}."
)
page.run()
