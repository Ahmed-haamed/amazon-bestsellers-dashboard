"""Cached file I/O used by the Streamlit application."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_pipeline import CleanResult, clean_amazon_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "amazon_bestsellers.csv"


@st.cache_data(max_entries=3, show_spinner="Loading the source data…")
def load_default_data(path_text: str = str(DEFAULT_DATA_PATH)) -> pd.DataFrame:
    return pd.read_csv(path_text, encoding="utf-8-sig")


@st.cache_data(max_entries=5, show_spinner="Reading the uploaded CSV…")
def load_uploaded_data(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(file_bytes), encoding="utf-8-sig")


@st.cache_data(max_entries=5, show_spinner="Cleaning and validating the data…")
def prepare_data(raw: pd.DataFrame) -> CleanResult:
    return clean_amazon_data(raw)


@st.cache_data(max_entries=10)
def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")

