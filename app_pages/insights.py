"""Decision-focused insights and recommendations."""

import math

import streamlit as st

from src.analytics import (
    author_summary,
    book_summary,
    correlation_matrix,
    genre_summary,
    recommendation_table,
    yearly_summary,
)
from src.charts import engagement_matrix_figure
from src.ui import compact_number, require_filtered_data


df = require_filtered_data()
books = book_summary(df)
authors = author_summary(df)
genres = genre_summary(df)
years = yearly_summary(df)
corr = correlation_matrix(df, method="spearman")

top_book = books.iloc[0]
top_author = authors.iloc[0]
top_reviewed = books.sort_values("review_snapshot", ascending=False).iloc[0]
rating_review_corr = float(corr.loc["user_rating", "reviews"])
price_rating_corr = float(corr.loc["price", "user_rating"])

st.warning(
    "These insights describe list presence, price, rating, and review snapshots. "
    "The dataset cannot prove sales lift, revenue, or causation.",
    icon=":material/analytics:",
)

with st.container(horizontal=True):
    st.metric(
        "Longest persistence",
        f"{int(top_book['years_on_list'])} years",
        border=True,
        icon=":material/history:",
    )
    st.metric(
        "Largest review snapshot",
        compact_number(top_reviewed["review_snapshot"]),
        border=True,
        icon=":material/forum:",
    )
    st.metric(
        "Rating–review association",
        None if math.isnan(rating_review_corr) else rating_review_corr,
        format="%.2f",
        border=True,
        icon=":material/compare_arrows:",
    )
    st.metric(
        "Price–rating association",
        None if math.isnan(price_rating_corr) else price_rating_corr,
        format="%.2f",
        border=True,
        icon=":material/price_check:",
    )

insight_left, insight_right = st.columns(2)
with insight_left:
    with st.container(border=True, height="stretch"):
        st.subheader("Evergreen beats one-off attention")
        st.markdown(
            f"**{top_book['title']}** leads the active view with "
            f"**{int(top_book['years_on_list'])} distinct years** on the list. "
            "Measure longevity by distinct years, not raw row count, because one "
            "title can have multiple same-year price records."
        )
        st.caption(
            "Action: maintain a separate evergreen watchlist and use recurring list "
            "presence as a portfolio-retention signal."
        )
with insight_right:
    with st.container(border=True, height="stretch"):
        st.subheader("Breadth and persistence are different author strategies")
        st.markdown(
            f"**{top_author['author']}** has {int(top_author['list_records'])} list "
            f"records across **{int(top_author['unique_titles'])} unique titles** in "
            "the active view. Other authors can lead through one durable book instead."
        )
        st.caption(
            "Action: compare prolific authors on title breadth and evergreen authors "
            "on years of persistence; do not rank both with one count."
        )

genre_left, genre_right = st.columns(2)
if len(genres) >= 2:
    leading = genres.iloc[0]
    other = genres.iloc[1]
    with genre_left:
        with st.container(border=True, height="stretch"):
            st.subheader("Genre volume does not equal audience response")
            st.markdown(
                f"**{leading['genre']}** leads with {int(leading['list_records'])} list "
                f"records, while **{other['genre']}** has a median review snapshot of "
                f"{compact_number(other['median_reviews'])}."
            )
            st.caption(
                "Action: set genre-specific benchmarks for reach, rating, and price."
            )
else:
    with genre_left:
        st.info("Select both genres to compare their different behavior.")

with genre_right:
    with st.container(border=True, height="stretch"):
        st.subheader("Higher price is not a quality guarantee")
        correlation_text = (
            "not estimable with the current filters"
            if math.isnan(price_rating_corr)
            else f"{price_rating_corr:.2f}"
        )
        st.markdown(
            f"The Spearman price–rating association is **{correlation_text}**. "
            "Premium reference books and box sets create valid price outliers, so "
            "median price and product-type segmentation are safer than a global mean."
        )
        st.caption(
            "Action: validate zero prices and compare premium products within their own segment."
        )

with st.container(border=True):
    st.plotly_chart(
        engagement_matrix_figure(df),
        width="stretch",
        key="insights_matrix",
        config={"displaylogo": False},
    )
    st.caption(
        "Upper-right books combine reach and satisfaction. Upper-left books are highly "
        "rated but niche; lower-right books are popular but more polarizing."
    )

st.subheader("Prioritized recommendations")
recommendations = recommendation_table(df)
st.dataframe(
    recommendations,
    column_config={
        "priority": st.column_config.NumberColumn("Priority", format="%d", pinned=True),
        "recommendation": st.column_config.TextColumn(
            "Recommendation", width="large"
        ),
        "evidence": st.column_config.TextColumn("Evidence", width="large"),
        "why_it_matters": st.column_config.TextColumn(
            "Why it matters", width="large"
        ),
    },
    hide_index=True,
    key="insights_recommendations",
)

with st.expander("What data should be collected next?", icon=":material/add_chart:"):
    st.markdown(
        """
1. **Rank and capture date** — to measure movement within each annual list.
2. **Units sold and revenue** — to connect list presence to commercial outcomes.
3. **Edition / format / ISBN** — to explain same-title/year price differences.
4. **Historical review timestamp** — to measure genuine review growth.
5. **Category and subcategory** — to benchmark comparable products more fairly.
6. **Promotion and discount flags** — to distinguish free offers from missing prices.
"""
    )

