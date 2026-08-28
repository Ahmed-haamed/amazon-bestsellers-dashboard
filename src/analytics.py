"""Reusable Pandas and NumPy aggregations for the dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd


DIMENSION_COLUMNS = {
    "Year": "year",
    "Genre": "genre",
    "Author": "author",
    "Rating band": "rating_band",
    "Price band": "price_band",
}

METRIC_COLUMNS = {
    "User rating": "user_rating",
    "Reviews": "reviews",
    "Price": "price",
    "Records": "title",
}

AGGREGATION_FUNCTIONS = {
    "Mean": "mean",
    "Median": "median",
    "Minimum": "min",
    "Maximum": "max",
    "Count": "count",
    "Unique count": "nunique",
}


def book_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse annual/edition appearances to one row per canonical title."""

    return (
        df.groupby(["title", "author", "genre"], as_index=False)
        .agg(
            list_records=("record_id", "size"),
            years_on_list=("year", "nunique"),
            first_year=("year", "min"),
            last_year=("year", "max"),
            median_rating=("user_rating", "median"),
            review_snapshot=("reviews", "max"),
            median_price=("price", "median"),
        )
        .sort_values(
            ["years_on_list", "review_snapshot", "median_rating"],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
    )


def author_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compare author breadth (titles) with recurring list presence."""

    authors = (
        df.groupby("author", as_index=False)
        .agg(
            list_records=("record_id", "size"),
            unique_titles=("title", "nunique"),
            active_years=("year", "nunique"),
            median_rating=("user_rating", "median"),
            median_price=("price", "median"),
        )
    )
    title_reach = (
        df.groupby(["author", "title"], as_index=False)["reviews"]
        .max()
        .groupby("author", as_index=False)["reviews"]
        .sum()
        .rename(columns={"reviews": "nonduplicated_review_reach"})
    )
    return (
        authors.merge(title_reach, on="author", how="left")
        .sort_values(
            ["list_records", "unique_titles", "nonduplicated_review_reach"],
            ascending=False,
        )
        .reset_index(drop=True)
    )


def genre_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("genre", as_index=False)
        .agg(
            list_records=("record_id", "size"),
            unique_titles=("title", "nunique"),
            average_rating=("user_rating", "mean"),
            median_reviews=("reviews", "median"),
            median_price=("price", "median"),
            average_price=("price", "mean"),
        )
        .sort_values("list_records", ascending=False)
        .reset_index(drop=True)
    )


def yearly_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("year", as_index=False)
        .agg(
            list_records=("record_id", "size"),
            unique_titles=("title", "nunique"),
            average_rating=("user_rating", "mean"),
            median_reviews=("reviews", "median"),
            median_price=("price", "median"),
            unique_authors=("author", "nunique"),
        )
        .sort_values("year")
    )


def correlation_matrix(df: pd.DataFrame, method: str = "spearman") -> pd.DataFrame:
    columns = ["user_rating", "reviews", "price"]
    return df[columns].corr(method=method)


def aggregate_data(
    df: pd.DataFrame,
    dimension_label: str,
    metric_label: str,
    aggregation_label: str,
) -> pd.DataFrame:
    """Run a controlled groupby aggregation selected in the UI."""

    dimension = DIMENSION_COLUMNS[dimension_label]
    metric = METRIC_COLUMNS[metric_label]
    aggregation = AGGREGATION_FUNCTIONS[aggregation_label]

    grouped = (
        df.groupby(dimension, dropna=False)[metric]
        .agg(aggregation)
        .reset_index(name="value")
    )
    if dimension == "year":
        return grouped.sort_values(dimension)
    return grouped.sort_values("value", ascending=False)


def robust_percent_change(start: float, end: float) -> float:
    if start == 0 or pd.isna(start) or pd.isna(end):
        return float("nan")
    return float(np.divide(end - start, start) * 100)


def recommendation_table(df: pd.DataFrame) -> pd.DataFrame:
    """Evidence-linked recommendations that adapt to active filters."""

    books = book_summary(df)
    genres = genre_summary(df)
    corr = correlation_matrix(df)

    top_book = books.iloc[0]
    leading_genre = genres.iloc[0]
    rating_review_corr = float(corr.loc["user_rating", "reviews"])
    zero_prices = int(df["is_zero_price"].sum())
    duplicate_pairs = int(
        (df.groupby(["title", "year"]).size() - 1).clip(lower=0).sum()
    )

    return pd.DataFrame(
        [
            {
                "priority": 1,
                "recommendation": "Protect evergreen inventory and campaigns",
                "evidence": f"{top_book['title']} appears in {int(top_book['years_on_list'])} distinct years.",
                "why_it_matters": "Recurring list presence is a stronger persistence signal than one-year popularity.",
            },
            {
                "priority": 2,
                "recommendation": "Segment genre strategy instead of using one benchmark",
                "evidence": f"{leading_genre['genre']} leads with {int(leading_genre['list_records'])} list records in the active view.",
                "why_it_matters": "Genre mixes differ in price, rating, and review behavior.",
            },
            {
                "priority": 3,
                "recommendation": "Use rating and reach together for acquisition",
                "evidence": f"Spearman rating–review correlation is {rating_review_corr:.2f}.",
                "why_it_matters": "Popularity and satisfaction are related only weakly; neither should be a standalone decision rule.",
            },
            {
                "priority": 4,
                "recommendation": "Validate ambiguous prices and editions",
                "evidence": f"{zero_prices} zero-price records and {duplicate_pairs} extra same-title/year records remain flagged.",
                "why_it_matters": "Treating these as ordinary paid listings can distort price comparisons.",
            },
            {
                "priority": 5,
                "recommendation": "Collect sales, rank, format, and timestamp fields next",
                "evidence": "The current file contains list presence, ratings, review snapshots, and price—but no units or revenue.",
                "why_it_matters": "Those fields are required for causal sales, edition, and time-series conclusions.",
            },
        ]
    )

