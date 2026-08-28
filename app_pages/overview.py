"""Executive overview page."""

import streamlit as st

from src.analytics import author_summary, book_summary, yearly_summary
from src.charts import (
    author_strategy_figure,
    book_persistence_figure,
    genre_by_year_figure,
)
from src.ui import bestseller_column_config, compact_number, require_filtered_data


df = require_filtered_data()
books = book_summary(df)
authors = author_summary(df)
yearly = yearly_summary(df)

st.info(
    "One row is a bestseller-list record. A title can appear in multiple years, "
    "so recurring rows are useful evidence of persistence—not duplicates to delete.",
    icon=":material/info:",
)

rating_sparkline = yearly["average_rating"].round(2).tolist()
price_sparkline = yearly["median_price"].round(2).tolist()

with st.container(horizontal=True):
    st.metric(
        "List records",
        len(df),
        border=True,
        icon=":material/format_list_numbered:",
    )
    st.metric(
        "Unique titles",
        int(df["title"].nunique()),
        border=True,
        icon=":material/menu_book:",
    )
    st.metric(
        "Average rating",
        float(df["user_rating"].mean()),
        format="%.2f",
        border=True,
        chart_data=rating_sparkline,
        chart_type="line",
        delta_color="blue",
        icon=":material/star:",
    )
    st.metric(
        "Median price",
        float(df["price"].median()),
        format="dollar",
        border=True,
        chart_data=price_sparkline,
        chart_type="bar",
        delta_color="violet",
        icon=":material/sell:",
    )

left, right = st.columns([1.15, 1])
with left:
    with st.container(border=True):
        st.plotly_chart(
            genre_by_year_figure(df),
            width="stretch",
            key="overview_genre_year",
            config={"displaylogo": False},
        )
with right:
    with st.container(border=True):
        st.plotly_chart(
            book_persistence_figure(df),
            width="stretch",
            key="overview_persistence",
            config={"displaylogo": False},
        )

with st.container(border=True):
    st.plotly_chart(
        author_strategy_figure(df),
        width="stretch",
        key="overview_author_strategy",
        config={"displaylogo": False},
    )
    st.caption(
        "Bubble size uses non-duplicated review reach: each title contributes its "
        "maximum review snapshot once, avoiding inflation from repeated years."
    )

st.subheader("Top persistent books")
top_books = books.head(15).copy()
st.dataframe(
    top_books,
    column_order=(
        "title",
        "author",
        "genre",
        "years_on_list",
        "first_year",
        "last_year",
        "median_rating",
        "review_snapshot",
        "median_price",
    ),
    column_config=bestseller_column_config(),
    hide_index=True,
    key="overview_top_books",
)

with st.expander("How to read this dashboard", icon=":material/help:"):
    st.markdown(
        """
- **List records** measure appearances in the annual Top-50 lists, not sales volume.
- **Years on list** counts distinct years, so two prices for the same title/year do not inflate persistence.
- **Reviews** are treated as a snapshot because recurring titles usually repeat the same count across years.
- **Median price** is preferred to the mean because specialty manuals and box sets create valid high-price outliers.
"""
    )
    st.caption(
        f"Current filtered review total is not shown because summing repeated snapshots "
        f"would overstate reach. Largest single-title snapshot: {compact_number(books['review_snapshot'].max())}."
    )

