"""Interactive EDA and aggregation playground."""

import streamlit as st

from src.analytics import (
    AGGREGATION_FUNCTIONS,
    DIMENSION_COLUMNS,
    METRIC_COLUMNS,
    aggregate_data,
)
from src.charts import (
    aggregation_figure,
    correlation_heatmap_figure,
    engagement_matrix_figure,
    matplotlib_yearly_figure,
    seaborn_distribution_figure,
)
from src.ui import bestseller_column_config, require_filtered_data


df = require_filtered_data()

st.info(
    "Analyst mindset: choose the grain first, then the metric, then an aggregation "
    "that does not double-count repeated snapshots.",
    icon=":material/psychology:",
)

st.subheader("Aggregation lab")
control_columns = st.columns(3)
with control_columns[0]:
    dimension_label = st.selectbox(
        "Group by", list(DIMENSION_COLUMNS), key="eda_dimension"
    )
with control_columns[1]:
    metric_label = st.selectbox(
        "Metric", list(METRIC_COLUMNS), key="eda_metric"
    )
with control_columns[2]:
    if metric_label == "Records":
        allowed_aggregations = ["Count", "Unique count"]
    elif metric_label == "Reviews":
        allowed_aggregations = ["Median", "Mean", "Maximum", "Minimum"]
    else:
        allowed_aggregations = ["Median", "Mean", "Maximum", "Minimum"]
    aggregation_label = st.selectbox(
        "Aggregation", allowed_aggregations, key="eda_aggregation"
    )

aggregated = aggregate_data(
    df, dimension_label, metric_label, aggregation_label
)
dimension_column = DIMENSION_COLUMNS[dimension_label]

chart_column, table_column = st.columns([1.6, 1])
with chart_column:
    with st.container(border=True):
        st.plotly_chart(
            aggregation_figure(
                aggregated,
                dimension_column,
                f"{aggregation_label} {metric_label.lower()}",
            ),
            width="stretch",
            key="eda_aggregation_chart",
            config={"displaylogo": False},
        )
with table_column:
    with st.container(border=True, height="stretch"):
        st.markdown("**Aggregated values**")
        st.dataframe(
            aggregated,
            column_config={
                dimension_column: dimension_label,
                "value": st.column_config.NumberColumn(
                    f"{aggregation_label} {metric_label.lower()}", format="%.2f"
                ),
            },
            hide_index=True,
            height=390,
            key="eda_aggregation_table",
        )

if metric_label == "Reviews":
    st.warning(
        "Review sums are intentionally unavailable: recurring titles usually reuse "
        "one review snapshot across years, so summing would double-count engagement.",
        icon=":material/warning:",
    )

st.subheader("Visualization lab")
library = st.segmented_control(
    "Chart library",
    ["Plotly", "Seaborn", "Matplotlib + NumPy"],
    default="Plotly",
    key="eda_library",
)

if library == "Plotly":
    left, right = st.columns([1.35, 1])
    with left:
        with st.container(border=True):
            st.plotly_chart(
                engagement_matrix_figure(df),
                width="stretch",
                key="eda_engagement_matrix",
                config={"displaylogo": False},
            )
    with right:
        with st.container(border=True):
            st.plotly_chart(
                correlation_heatmap_figure(df),
                width="stretch",
                key="eda_corr_heatmap",
                config={"displaylogo": False},
            )
    st.caption(
        "The review axis is logarithmic because engagement is strongly right-skewed. "
        "Quadrant lines use the book-level medians."
    )
elif library == "Seaborn":
    with st.container(border=True):
        st.pyplot(
            seaborn_distribution_figure(df), width="stretch", clear_figure=True
        )
    st.caption(
        "Seaborn highlights distribution shape and robust price comparisons by genre."
    )
else:
    with st.container(border=True):
        st.pyplot(
            matplotlib_yearly_figure(df), width="stretch", clear_figure=True
        )
    st.caption(
        "NumPy builds aligned bar positions; Matplotlib controls the chart primitives."
    )

with st.expander("Explore the cleaned records", icon=":material/table_view:"):
    st.dataframe(
        df,
        column_order=(
            "title",
            "author",
            "year",
            "genre",
            "user_rating",
            "reviews",
            "price",
            "rating_band",
            "price_band",
            "engagement_segment",
        ),
        column_config=bestseller_column_config(),
        hide_index=True,
        key="eda_clean_records",
    )

