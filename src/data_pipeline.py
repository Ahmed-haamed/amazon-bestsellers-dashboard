"""Validation, cleaning, feature engineering, and data-quality auditing."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

import numpy as np
import pandas as pd


CORE_COLUMNS = [
    "title",
    "author",
    "user_rating",
    "reviews",
    "price",
    "year",
    "genre",
]

HEADER_ALIASES = {
    "name": "title",
    "title": "title",
    "author": "author",
    "user_rating": "user_rating",
    "userrating": "user_rating",
    "reviews": "reviews",
    "price": "price",
    "year": "year",
    "genre": "genre",
}

# These characters are Windows-1252 punctuation that was stored as Unicode
# control characters in the source file. Repairing them is deterministic.
CP1252_CONTROL_REPAIRS = {
    "\u0080": "€",
    "\u0082": "‚",
    "\u0083": "ƒ",
    "\u0084": "„",
    "\u0085": "…",
    "\u0086": "†",
    "\u0087": "‡",
    "\u0088": "ˆ",
    "\u0089": "‰",
    "\u008a": "Š",
    "\u008b": "‹",
    "\u008c": "Œ",
    "\u008e": "Ž",
    "\u0091": "‘",
    "\u0092": "’",
    "\u0093": "“",
    "\u0094": "”",
    "\u0095": "•",
    "\u0096": "–",
    "\u0097": "—",
    "\u0098": "˜",
    "\u0099": "™",
    "\u009a": "š",
    "\u009b": "›",
    "\u009c": "œ",
    "\u009e": "ž",
    "\u009f": "Ÿ",
}

AUTHOR_ALIASES = {
    "J. K. Rowling": "J.K. Rowling",
    "George R. R. Martin": "George R.R. Martin",
}

TITLE_ALIASES = {
    "The 5 Love Languages: The Secret to Love That Lasts": (
        "The 5 Love Languages: The Secret to Love that Lasts"
    ),
}


class DataValidationError(ValueError):
    """Raised when the uploaded file cannot support this analysis."""


@dataclass(frozen=True)
class CleanResult:
    """All outputs from the auditable cleaning pipeline."""

    data: pd.DataFrame
    summary: dict[str, int | float]
    issues: pd.DataFrame
    change_log: pd.DataFrame


def _header_key(value: object) -> str:
    text = str(value).strip().casefold()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def _repair_text(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    text = str(value).translate(str.maketrans(CP1252_CONTROL_REPAIRS))
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip()


def _contains_c1_control(value: object) -> bool:
    if pd.isna(value):
        return False
    return any(0x80 <= ord(character) <= 0x9F for character in str(value))


def _iqr_outlier_mask(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    q1, q3 = numeric.quantile([0.25, 0.75])
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        return pd.Series(False, index=series.index)
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return numeric.lt(lower) | numeric.gt(upper)


def _append_changes(
    changes: list[dict[str, object]],
    source_rows: pd.Series,
    column: str,
    before: pd.Series,
    after: pd.Series,
    reason: str,
) -> None:
    changed = before.astype("string").fillna("<missing>").ne(
        after.astype("string").fillna("<missing>")
    )
    for idx in before.index[changed]:
        changes.append(
            {
                "source_row": int(source_rows.loc[idx]),
                "column": column,
                "before": before.loc[idx],
                "after": after.loc[idx],
                "reason": reason,
            }
        )


def _normalize_schema(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        raise DataValidationError("The CSV file is empty.")

    rename_map: dict[object, str] = {}
    for column in raw.columns:
        key = _header_key(column)
        if key in HEADER_ALIASES:
            rename_map[column] = HEADER_ALIASES[key]

    normalized = raw.rename(columns=rename_map).copy()
    missing = [column for column in CORE_COLUMNS if column not in normalized.columns]
    if missing:
        expected = ", ".join(CORE_COLUMNS)
        missing_text = ", ".join(missing)
        raise DataValidationError(
            f"Missing required columns: {missing_text}. Expected: {expected}."
        )
    return normalized[CORE_COLUMNS].copy()


def clean_amazon_data(raw: pd.DataFrame) -> CleanResult:
    """Clean the Amazon bestseller data without deleting meaningful repeats.

    Exact duplicate rows and rows missing a required field are removable errors.
    Repeated titles across years and same-title/year rows with different prices
    are retained because they can represent persistence or separate editions.
    """

    df = _normalize_schema(raw)
    source_rows = pd.Series(np.arange(1, len(df) + 1), index=df.index)
    changes: list[dict[str, object]] = []

    source_rows_count = int(len(df))
    source_columns_count = int(len(df.columns))
    source_missing_cells = int(df.isna().sum().sum())
    source_blank_cells = int(
        sum(
            df[column].astype("string").str.strip().eq("").fillna(False).sum()
            for column in ["title", "author", "genre"]
        )
    )
    encoding_affected_rows = int(
        pd.DataFrame(
            {
                column: df[column].map(_contains_c1_control)
                for column in ["title", "author", "genre"]
            }
        )
        .any(axis=1)
        .sum()
    )

    for column in ["title", "author", "genre"]:
        before = df[column].copy()
        after = before.map(_repair_text).astype("string")
        _append_changes(
            changes,
            source_rows,
            column,
            before,
            after,
            "Encoding, whitespace, or Unicode normalization",
        )
        df[column] = after

    before_author = df["author"].copy()
    df["author"] = df["author"].replace(AUTHOR_ALIASES)
    _append_changes(
        changes,
        source_rows,
        "author",
        before_author,
        df["author"],
        "Canonical author spelling",
    )

    before_title = df["title"].copy()
    df["title"] = df["title"].replace(TITLE_ALIASES)
    _append_changes(
        changes,
        source_rows,
        "title",
        before_title,
        df["title"],
        "Canonical title spelling",
    )

    for column in ["user_rating", "reviews", "price", "year"]:
        before = df[column].copy()
        df[column] = pd.to_numeric(df[column], errors="coerce")
        newly_missing = before.notna() & df[column].isna()
        for idx in df.index[newly_missing]:
            changes.append(
                {
                    "source_row": int(source_rows.loc[idx]),
                    "column": column,
                    "before": before.loc[idx],
                    "after": pd.NA,
                    "reason": "Invalid numeric value coerced to missing",
                }
            )

    required_missing = df[CORE_COLUMNS].isna().any(axis=1)
    required_blank = df[["title", "author", "genre"]].eq("").any(axis=1)
    invalid_domain = (
        ~df["user_rating"].between(1, 5)
        | df["reviews"].lt(0)
        | df["price"].lt(0)
        | ~df["year"].between(1900, 2100)
    ).fillna(True)
    removed_missing_or_invalid = required_missing | required_blank | invalid_domain
    invalid_rows_removed = int(removed_missing_or_invalid.sum())
    df = df.loc[~removed_missing_or_invalid].copy()
    source_rows = source_rows.loc[df.index]

    exact_duplicate_mask = df.duplicated(CORE_COLUMNS, keep="first")
    exact_duplicates_removed = int(exact_duplicate_mask.sum())
    df = df.loc[~exact_duplicate_mask].copy()
    source_rows = source_rows.loc[df.index]

    df["reviews"] = df["reviews"].astype("int64")
    df["price"] = df["price"].astype("int64")
    df["year"] = df["year"].astype("int64")
    df["user_rating"] = df["user_rating"].astype("float64")

    df["is_zero_price"] = df["price"].eq(0)
    df["is_price_outlier"] = _iqr_outlier_mask(df["price"])
    df["is_review_outlier"] = _iqr_outlier_mask(df["reviews"])
    df["is_rating_outlier"] = _iqr_outlier_mask(df["user_rating"])
    df["is_title_year_duplicate"] = df.duplicated(
        ["title", "year"], keep=False
    )
    df["is_truncated_text"] = (
        df["title"].str.endswith("…")
        | df["author"].str.endswith("…")
        | df["title"].eq("JOURNEY TO THE ICE P")
    )

    df["rating_band"] = np.select(
        [
            df["user_rating"].ge(4.8),
            df["user_rating"].ge(4.5),
            df["user_rating"].ge(4.0),
        ],
        ["Exceptional (4.8+)", "Strong (4.5–4.7)", "Solid (4.0–4.4)"],
        default="Below 4.0",
    )
    df["price_band"] = np.select(
        [
            df["price"].eq(0),
            df["price"].le(10),
            df["price"].le(20),
        ],
        ["Zero / verify", "Budget ($1–10)", "Mid-range ($11–20)"],
        default="Premium ($21+)",
    )

    rating_median = float(df["user_rating"].median())
    reviews_median = float(df["reviews"].median())
    high_rating = df["user_rating"].ge(rating_median)
    high_reviews = df["reviews"].ge(reviews_median)
    df["engagement_segment"] = np.select(
        [
            high_rating & high_reviews,
            high_rating & ~high_reviews,
            ~high_rating & high_reviews,
        ],
        ["Stars", "Loved niche", "Popular but polarizing"],
        default="Developing",
    )
    df["review_log10"] = np.log10(df["reviews"].clip(lower=1))

    title_stats = df.groupby("title")["year"].agg(
        years_on_list="nunique", first_year="min", last_year="max"
    )
    df = df.join(title_stats, on="title")
    df.insert(0, "record_id", np.arange(1, len(df) + 1))
    df.insert(1, "source_row", source_rows.astype(int).to_numpy())
    df = df.reset_index(drop=True)

    repeated_title_sizes = df.groupby("title").size()
    repeated_titles = repeated_title_sizes[repeated_title_sizes.gt(1)].index
    repeated_rows = int(df["title"].isin(repeated_titles).sum())
    constant_review_titles = int(
        (
            df[df["title"].isin(repeated_titles)]
            .groupby("title")["reviews"]
            .nunique()
            .eq(1)
        ).sum()
    )

    title_year_sizes = df.groupby(["title", "year"]).size()
    title_year_extra = int((title_year_sizes - 1).clip(lower=0).sum())
    title_year_affected = int(df["is_title_year_duplicate"].sum())

    change_log = pd.DataFrame(
        changes,
        columns=["source_row", "column", "before", "after", "reason"],
    )
    alias_changes = int(
        change_log["reason"].isin(
            ["Canonical author spelling", "Canonical title spelling"]
        ).sum()
    )

    completeness = 100.0
    denominator = source_rows_count * source_columns_count
    if denominator:
        completeness = round(
            100 * (1 - (source_missing_cells + source_blank_cells) / denominator),
            2,
        )

    summary: dict[str, int | float] = {
        "source_rows": source_rows_count,
        "clean_rows": int(len(df)),
        "removed_rows": invalid_rows_removed + exact_duplicates_removed,
        "source_columns": source_columns_count,
        "completeness_pct": completeness,
        "source_missing_cells": source_missing_cells,
        "source_blank_cells": source_blank_cells,
        "invalid_rows_removed": invalid_rows_removed,
        "exact_duplicates_removed": exact_duplicates_removed,
        "encoding_affected_rows": encoding_affected_rows,
        "alias_changes": alias_changes,
        "unique_titles": int(df["title"].nunique()),
        "unique_authors": int(df["author"].nunique()),
        "repeated_titles": int(len(repeated_titles)),
        "repeated_rows": repeated_rows,
        "constant_review_repeat_titles": constant_review_titles,
        "title_year_duplicate_extra": title_year_extra,
        "title_year_duplicate_affected": title_year_affected,
        "zero_price_rows": int(df["is_zero_price"].sum()),
        "price_outlier_rows": int(df["is_price_outlier"].sum()),
        "review_outlier_rows": int(df["is_review_outlier"].sum()),
        "rating_outlier_rows": int(df["is_rating_outlier"].sum()),
        "truncated_text_rows": int(df["is_truncated_text"].sum()),
    }

    issues = pd.DataFrame(
        [
            {
                "issue": "Missing or blank required values",
                "severity": "Pass" if source_missing_cells + source_blank_cells == 0 else "High",
                "affected_records": source_missing_cells + source_blank_cells,
                "what_happened": "No missing values were found in this file." if source_missing_cells + source_blank_cells == 0 else "Required values were absent.",
                "likely_cause": "Source completeness or extraction gaps.",
                "action": "Rows failing required-field checks are removed and counted.",
            },
            {
                "issue": "Encoding / Unicode artifacts",
                "severity": "Fixed" if encoding_affected_rows else "Pass",
                "affected_records": encoding_affected_rows,
                "what_happened": "Windows-1252 punctuation appeared as control characters.",
                "likely_cause": "An earlier encoding conversion treated smart punctuation incorrectly.",
                "action": "Repair punctuation, normalize to NFC, and preserve readable text.",
            },
            {
                "issue": "Author or title spelling variants",
                "severity": "Fixed" if alias_changes else "Pass",
                "affected_records": alias_changes,
                "what_happened": "The same entity used different punctuation, spacing, or capitalization.",
                "likely_cause": "Manual entry and inconsistent catalog naming.",
                "action": "Map only verified aliases to one canonical display name.",
            },
            {
                "issue": "Exact duplicate rows",
                "severity": "Fixed" if exact_duplicates_removed else "Pass",
                "affected_records": exact_duplicates_removed,
                "what_happened": "Rows matched across every business field.",
                "likely_cause": "Potential duplicate extraction.",
                "action": "Remove exact duplicates only; keep meaningful repeats.",
            },
            {
                "issue": "Repeated titles across years",
                "severity": "Context",
                "affected_records": repeated_rows,
                "what_happened": f"{len(repeated_titles)} titles appear more than once.",
                "likely_cause": "This is an annual Top-50 list, so repeated titles represent persistence.",
                "action": "Retain records and aggregate by unique title when the question is book-level.",
            },
            {
                "issue": "Same title and year, different price",
                "severity": "Review",
                "affected_records": title_year_affected,
                "what_happened": f"{title_year_extra} extra records share title and year.",
                "likely_cause": "Likely separate formats or editions, but the source has no edition field.",
                "action": "Retain, flag, and avoid assuming these are accidental duplicates.",
            },
            {
                "issue": "Zero price",
                "severity": "Review",
                "affected_records": int(df["is_zero_price"].sum()),
                "what_happened": "Price is exactly zero for some listings.",
                "likely_cause": "Could be a promotion, a free edition, or missing price encoded as zero.",
                "action": "Do not impute without evidence; flag and exclude from paid-price sensitivity checks.",
            },
            {
                "issue": "Statistical outliers",
                "severity": "Context",
                "affected_records": int(
                    (df[["is_price_outlier", "is_review_outlier", "is_rating_outlier"]]).any(axis=1).sum()
                ),
                "what_happened": "IQR rules flag premium manuals, box sets, viral books, or unusually low ratings.",
                "likely_cause": "Real product differences and heavy-tailed engagement, not necessarily errors.",
                "action": "Keep outliers, show robust medians, and use a log scale for reviews.",
            },
            {
                "issue": "Truncated source text",
                "severity": "Review",
                "affected_records": int(df["is_truncated_text"].sum()),
                "what_happened": "Some titles or authors end with an ellipsis or appear visibly cut off.",
                "likely_cause": "The upstream listing/export likely truncated long display text.",
                "action": "Flag only; restoring full text requires a trusted catalog source.",
            },
            {
                "issue": "Review count repeated across years",
                "severity": "High context",
                "affected_records": constant_review_titles,
                "what_happened": "Nearly every recurring title keeps the same review count in each year.",
                "likely_cause": "Reviews are probably a single snapshot copied to each annual appearance.",
                "action": "Never interpret review counts as annual growth or sum them across years.",
            },
        ]
    )

    return CleanResult(data=df, summary=summary, issues=issues, change_log=change_log)

