"""Plotly, Seaborn, Matplotlib, and NumPy visualizations."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

from src.analytics import author_summary, book_summary, correlation_matrix


BLUE = "#60A5FA"
GREEN = "#34D399"
VIOLET = "#A78BFA"
RED = "#F87171"
YELLOW = "#FBBF24"
SLATE = "#0F172A"
TEXT = "#F1F5F9"
GRID = "#334155"
GENRE_COLORS = {"Fiction": BLUE, "Non Fiction": GREEN}


def _polish_plotly(fig: go.Figure, height: int = 430) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=55, b=20),
        legend_title_text="",
        hoverlabel=dict(namelength=-1),
    )
    return fig


def genre_by_year_figure(df: pd.DataFrame) -> go.Figure:
    counts = (
        df.groupby(["year", "genre"], as_index=False)
        .size()
        .rename(columns={"size": "list_records"})
    )
    fig = px.bar(
        counts,
        x="year",
        y="list_records",
        color="genre",
        barmode="stack",
        color_discrete_map=GENRE_COLORS,
        labels={"year": "Year", "list_records": "List records", "genre": "Genre"},
        title="Genre mix by year",
    )
    fig.update_xaxes(dtick=1)
    return _polish_plotly(fig)


def book_persistence_figure(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    books = book_summary(df).head(top_n).sort_values("years_on_list")
    books["display_title"] = books["title"].map(
        lambda value: value if len(value) <= 32 else f"{value[:29]}…"
    )
    fig = px.bar(
        books,
        x="years_on_list",
        y="display_title",
        orientation="h",
        color="genre",
        color_discrete_map=GENRE_COLORS,
        custom_data=[
            "title",
            "author",
            "first_year",
            "last_year",
            "review_snapshot",
        ],
        labels={"years_on_list": "Distinct years on list", "display_title": "Book"},
        title=f"Top {min(top_n, len(books))} persistent titles",
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>Years on list: %{x}"
            "<br>Author: %{customdata[1]}"
            "<br>Window: %{customdata[2]}–%{customdata[3]}"
            "<br>Review snapshot: %{customdata[4]:,}<extra></extra>"
        )
    )
    return _polish_plotly(fig, height=500)


def author_strategy_figure(df: pd.DataFrame, top_n: int = 30) -> go.Figure:
    authors = author_summary(df).head(top_n).copy()
    fig = px.scatter(
        authors,
        x="unique_titles",
        y="list_records",
        size="nonduplicated_review_reach",
        color="median_rating",
        hover_name="author",
        color_continuous_scale="Viridis",
        labels={
            "unique_titles": "Unique titles (breadth)",
            "list_records": "List records (presence)",
            "median_rating": "Median rating",
            "nonduplicated_review_reach": "Review reach",
        },
        title="Author breadth vs. repeated list presence",
    )
    return _polish_plotly(fig)


def engagement_matrix_figure(df: pd.DataFrame) -> go.Figure:
    books = book_summary(df)
    rating_cut = float(books["median_rating"].median())
    review_cut = float(books["review_snapshot"].median())
    books["marker_size"] = books["median_price"].clip(lower=1)

    fig = px.scatter(
        books,
        x="review_snapshot",
        y="median_rating",
        log_x=True,
        color="genre",
        size="marker_size",
        size_max=35,
        color_discrete_map=GENRE_COLORS,
        hover_name="title",
        hover_data={
            "author": True,
            "review_snapshot": ":,",
            "median_rating": ":.1f",
            "median_price": ":$.0f",
            "years_on_list": True,
            "marker_size": False,
        },
        labels={
            "review_snapshot": "Reviews (log scale)",
            "median_rating": "Median user rating",
        },
        title="Popularity–satisfaction matrix",
    )
    fig.add_hline(y=rating_cut, line_dash="dot", line_color=YELLOW)
    fig.add_vline(x=review_cut, line_dash="dot", line_color=YELLOW)
    return _polish_plotly(fig, height=520)


def correlation_heatmap_figure(df: pd.DataFrame) -> go.Figure:
    corr = correlation_matrix(df, method="spearman")
    labels = ["User rating", "Reviews", "Price"]
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.to_numpy(),
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            text=np.round(corr.to_numpy(), 2),
            texttemplate="%{text:.2f}",
            hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(title="Spearman correlations")
    return _polish_plotly(fig, height=420)


def aggregation_figure(
    aggregated: pd.DataFrame,
    dimension_column: str,
    value_label: str,
) -> go.Figure:
    frame = aggregated.head(25) if dimension_column == "author" else aggregated
    horizontal = dimension_column not in {"year"}
    if horizontal:
        frame = frame.sort_values("value")
        fig = px.bar(
            frame,
            x="value",
            y=dimension_column,
            orientation="h",
            color="value",
            color_continuous_scale="Blues",
            labels={dimension_column: "Group", "value": value_label},
        )
    else:
        fig = px.bar(
            frame,
            x=dimension_column,
            y="value",
            color="value",
            color_continuous_scale="Blues",
            labels={dimension_column: "Year", "value": value_label},
        )
        fig.update_xaxes(dtick=1)
    fig.update_layout(title=f"{value_label} by {dimension_column.replace('_', ' ')}")
    return _polish_plotly(fig, height=460)


def seaborn_distribution_figure(df: pd.DataFrame) -> Figure:
    sns.set_theme(style="darkgrid")
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    figure.patch.set_facecolor(SLATE)

    sns.histplot(
        data=df,
        x="user_rating",
        hue="genre",
        multiple="layer",
        bins=np.arange(3.25, 5.05, 0.1),
        palette=GENRE_COLORS,
        alpha=0.55,
        ax=axes[0],
    )
    axes[0].set_title("Rating distribution by genre")
    axes[0].set_xlabel("User rating")
    axes[0].set_ylabel("List records")

    paid = df[df["price"].gt(0)]
    sns.boxplot(
        data=paid,
        x="genre",
        y="price",
        hue="genre",
        palette=GENRE_COLORS,
        legend=False,
        ax=axes[1],
    )
    axes[1].set_yscale("log")
    axes[1].set_title("Paid-price spread (log scale)")
    axes[1].set_xlabel("Genre")
    axes[1].set_ylabel("Price ($)")

    for axis in axes:
        axis.set_facecolor(SLATE)
        axis.tick_params(colors=TEXT)
        axis.xaxis.label.set_color(TEXT)
        axis.yaxis.label.set_color(TEXT)
        axis.title.set_color(TEXT)
        for spine in axis.spines.values():
            spine.set_color(GRID)
    figure.tight_layout()
    return figure


def matplotlib_yearly_figure(df: pd.DataFrame) -> Figure:
    counts = pd.crosstab(df["year"], df["genre"])
    years = counts.index.to_numpy()
    positions = np.arange(len(years))
    width = 0.38

    fiction = counts.get("Fiction", pd.Series(0, index=counts.index)).to_numpy()
    nonfiction = counts.get(
        "Non Fiction", pd.Series(0, index=counts.index)
    ).to_numpy()

    figure, axis = plt.subplots(figsize=(13, 5))
    figure.patch.set_facecolor(SLATE)
    axis.set_facecolor(SLATE)
    axis.bar(positions - width / 2, fiction, width, label="Fiction", color=BLUE)
    axis.bar(
        positions + width / 2,
        nonfiction,
        width,
        label="Non Fiction",
        color=GREEN,
    )
    axis.set_xticks(positions, years)
    axis.set_xlabel("Year")
    axis.set_ylabel("List records")
    axis.set_title("Annual genre composition — Matplotlib + NumPy")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.22)
    axis.tick_params(colors=TEXT)
    axis.xaxis.label.set_color(TEXT)
    axis.yaxis.label.set_color(TEXT)
    axis.title.set_color(TEXT)
    for spine in axis.spines.values():
        spine.set_color(GRID)
    figure.tight_layout()
    return figure


def missingness_figure(raw: pd.DataFrame) -> Figure:
    missing = raw.isna().mean().mul(100).sort_values(ascending=False)
    figure, axis = plt.subplots(figsize=(10, 3.8))
    figure.patch.set_facecolor(SLATE)
    axis.set_facecolor(SLATE)
    sns.barplot(x=missing.index, y=missing.values, color=BLUE, ax=axis)
    axis.set_ylim(0, max(5, float(missing.max()) * 1.2))
    axis.set_title("Missing values by source column")
    axis.set_xlabel("Column")
    axis.set_ylabel("Missing (%)")
    axis.tick_params(axis="x", rotation=25, colors=TEXT)
    axis.tick_params(axis="y", colors=TEXT)
    axis.xaxis.label.set_color(TEXT)
    axis.yaxis.label.set_color(TEXT)
    axis.title.set_color(TEXT)
    for spine in axis.spines.values():
        spine.set_color(GRID)
    figure.tight_layout()
    return figure


def outlier_boxplot_figure(df: pd.DataFrame) -> Figure:
    long = df[["user_rating", "reviews", "price"]].copy()
    long["reviews"] = np.log10(long["reviews"].clip(lower=1))
    long = long.rename(
        columns={
            "user_rating": "User rating",
            "reviews": "Reviews (log10)",
            "price": "Price",
        }
    ).melt(var_name="Metric", value_name="Value")

    figure, axis = plt.subplots(figsize=(10, 4.2))
    figure.patch.set_facecolor(SLATE)
    axis.set_facecolor(SLATE)
    sns.boxplot(
        data=long,
        x="Metric",
        y="Value",
        hue="Metric",
        palette=[BLUE, GREEN, VIOLET],
        legend=False,
        ax=axis,
    )
    axis.set_title("IQR outlier view (reviews transformed to log10)")
    axis.set_xlabel("")
    axis.tick_params(colors=TEXT)
    axis.yaxis.label.set_color(TEXT)
    axis.title.set_color(TEXT)
    for spine in axis.spines.values():
        spine.set_color(GRID)
    figure.tight_layout()
    return figure
