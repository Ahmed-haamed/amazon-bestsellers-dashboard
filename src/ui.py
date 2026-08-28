"""Shared Streamlit UI helpers and global filtering."""

from __future__ import annotations

import pandas as pd
import streamlit as st


FILTER_KEYS = [
    "filter_years",
    "filter_genres",
    "filter_rating",
    "filter_price",
    "filter_search",
]


def reset_filters() -> None:
    for key in FILTER_KEYS:
        st.session_state.pop(key, None)


def render_global_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Render sidebar filters in the entry point so state survives page changes."""

    min_year, max_year = int(df["year"].min()), int(df["year"].max())
    min_rating, max_rating = float(df["user_rating"].min()), float(
        df["user_rating"].max()
    )
    max_price = int(df["price"].max())
    genres = sorted(df["genre"].dropna().unique().tolist())

    st.subheader("Filters")
    year_range = st.slider(
        "Year range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        key="filter_years",
    )
    selected_genres = st.pills(
        "Genres",
        genres,
        default=genres,
        selection_mode="multi",
        key="filter_genres",
        width="stretch",
    )
    rating_range = st.slider(
        "User rating",
        min_value=min_rating,
        max_value=max_rating,
        value=(min_rating, max_rating),
        step=0.1,
        key="filter_rating",
    )
    price_range = st.slider(
        "Price ($)",
        min_value=0,
        max_value=max_price,
        value=(0, max_price),
        key="filter_price",
    )
    search_text = st.text_input(
        "Search title or author",
        placeholder="e.g. Rowling",
        key="filter_search",
    )
    st.button(
        "Reset filters",
        icon=":material/restart_alt:",
        type="tertiary",
        on_click=reset_filters,
    )

    mask = (
        df["year"].between(*year_range)
        & df["user_rating"].between(*rating_range)
        & df["price"].between(*price_range)
    )
    if selected_genres:
        mask &= df["genre"].isin(selected_genres)
    else:
        mask &= False
    if search_text.strip():
        query = search_text.strip()
        mask &= df["title"].str.contains(query, case=False, regex=False) | df[
            "author"
        ].str.contains(query, case=False, regex=False)
    return df.loc[mask].copy()


def require_filtered_data() -> pd.DataFrame:
    df = st.session_state.get("filtered_data")
    if df is None or df.empty:
        st.warning(
            "No records match the active filters. Reset or broaden the filters.",
            icon=":material/filter_alt_off:",
        )
        st.stop()
    return df


def compact_number(value: float | int) -> str:
    number = float(value)
    if abs(number) >= 1_000_000:
        return f"{number / 1_000_000:.1f}M"
    if abs(number) >= 1_000:
        return f"{number / 1_000:.1f}K"
    return f"{number:,.0f}"


def bestseller_column_config() -> dict[str, object]:
    return {
        "title": st.column_config.TextColumn("Title", pinned=True),
        "author": st.column_config.TextColumn("Author"),
        "genre": st.column_config.TextColumn("Genre"),
        "user_rating": st.column_config.NumberColumn(
            "Rating", format="%.1f", min_value=1, max_value=5
        ),
        "reviews": st.column_config.NumberColumn("Reviews", format="%,d"),
        "price": st.column_config.NumberColumn("Price", format="$%d"),
        "year": st.column_config.NumberColumn("Year", format="%d"),
        "years_on_list": st.column_config.NumberColumn(
            "Years on list", format="%d"
        ),
        "median_rating": st.column_config.NumberColumn(
            "Median rating", format="%.1f"
        ),
        "review_snapshot": st.column_config.NumberColumn(
            "Review snapshot", format="%,d"
        ),
        "median_price": st.column_config.NumberColumn(
            "Median price", format="$%.1f"
        ),
    }

