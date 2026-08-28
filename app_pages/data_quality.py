"""Data-quality diagnostics and cleaning audit page."""

import pandas as pd
import streamlit as st

from src.charts import missingness_figure, outlier_boxplot_figure
from src.data_io import dataframe_to_csv_bytes
from src.ui import bestseller_column_config


result = st.session_state["clean_result"]
raw = st.session_state["raw_data"]
cleaned = result.data
summary = result.summary

st.info(
    "This page always audits the full uploaded file; dashboard filters do not change "
    "quality counts.",
    icon=":material/shield:",
)

with st.container(horizontal=True):
    st.metric(
        "Completeness",
        summary["completeness_pct"] / 100,
        format="percent",
        border=True,
        icon=":material/data_check:",
    )
    st.metric(
        "Rows retained",
        summary["clean_rows"],
        delta=f"{summary['removed_rows']} removed",
        delta_color="off",
        delta_arrow="off",
        border=True,
        icon=":material/table_rows:",
    )
    st.metric(
        "Text / alias fixes",
        len(result.change_log),
        border=True,
        icon=":material/find_replace:",
    )
    st.metric(
        "Rows needing review",
        int(
            cleaned[
                [
                    "is_zero_price",
                    "is_title_year_duplicate",
                    "is_truncated_text",
                ]
            ]
            .any(axis=1)
            .sum()
        ),
        border=True,
        icon=":material/warning:",
    )

st.subheader("Issue register: what happened, why, and what we did")
st.dataframe(
    result.issues,
    column_config={
        "issue": st.column_config.TextColumn("Issue", pinned=True),
        "severity": st.column_config.TextColumn("Status"),
        "affected_records": st.column_config.NumberColumn(
            "Affected", format="%,d"
        ),
        "what_happened": st.column_config.TextColumn("What happened", width="large"),
        "likely_cause": st.column_config.TextColumn("Likely cause", width="large"),
        "action": st.column_config.TextColumn("Cleaning decision", width="large"),
    },
    hide_index=True,
    height=430,
    key="quality_issue_register",
)

left, right = st.columns(2)
with left:
    with st.container(border=True):
        st.pyplot(missingness_figure(raw), width="stretch", clear_figure=True)
        st.caption("Zero-height bars confirm the source has no null values.")
with right:
    with st.container(border=True):
        st.pyplot(
            outlier_boxplot_figure(cleaned), width="stretch", clear_figure=True
        )
        st.caption("Outliers are flagged and retained; they are not automatically errors.")

st.subheader("Before / after change log")
if result.change_log.empty:
    st.success("No cell-level text corrections were required.")
else:
    st.dataframe(
        result.change_log,
        column_config={
            "source_row": st.column_config.NumberColumn("Source row", format="%d"),
            "column": "Column",
            "before": st.column_config.TextColumn("Before", width="large"),
            "after": st.column_config.TextColumn("After", width="large"),
            "reason": st.column_config.TextColumn("Reason", width="large"),
        },
        hide_index=True,
        key="quality_change_log",
    )

st.subheader("Records requiring human verification")
issue_choice = st.segmented_control(
    "Issue type",
    ["Zero price", "Same title/year", "Truncated text", "Any statistical outlier"],
    default="Zero price",
    key="quality_issue_choice",
)

issue_masks = {
    "Zero price": cleaned["is_zero_price"],
    "Same title/year": cleaned["is_title_year_duplicate"],
    "Truncated text": cleaned["is_truncated_text"],
    "Any statistical outlier": cleaned[
        ["is_price_outlier", "is_review_outlier", "is_rating_outlier"]
    ].any(axis=1),
}
selected_issues = cleaned.loc[issue_masks[issue_choice]].copy()
st.dataframe(
    selected_issues,
    column_order=(
        "source_row",
        "title",
        "author",
        "year",
        "genre",
        "user_rating",
        "reviews",
        "price",
    ),
    column_config=bestseller_column_config(),
    hide_index=True,
    key="quality_review_rows",
)

with st.container(horizontal=True):
    st.download_button(
        "Download cleaned data",
        data=dataframe_to_csv_bytes(cleaned),
        file_name="amazon_bestsellers_clean.csv",
        mime="text/csv",
        icon=":material/download:",
        type="primary",
        on_click="ignore",
    )
    st.download_button(
        "Download change log",
        data=dataframe_to_csv_bytes(result.change_log),
        file_name="cleaning_change_log.csv",
        mime="text/csv",
        icon=":material/history:",
        on_click="ignore",
    )

with st.expander("Cleaning pipeline", icon=":material/account_tree:"):
    cleaning_steps = pd.DataFrame(
        [
            (1, "Validate schema", "Require the seven expected business fields."),
            (2, "Repair text", "Fix control-character punctuation, trim whitespace, normalize Unicode to NFC."),
            (3, "Standardize entities", "Map only verified author/title variants to canonical names."),
            (4, "Convert types", "Coerce rating, reviews, price, and year to numeric types."),
            (5, "Remove proven errors", "Drop invalid required rows and exact duplicates only."),
            (6, "Retain meaningful repeats", "Keep annual appearances and possible edition-level records."),
            (7, "Engineer flags", "Add zero-price, duplicate-key, truncation, and IQR outlier flags."),
            (8, "Create analysis fields", "Build rating, price, engagement, and persistence features."),
        ],
        columns=["step", "stage", "rule"],
    )
    st.dataframe(cleaning_steps, hide_index=True)

