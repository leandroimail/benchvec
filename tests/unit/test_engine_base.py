"""Unit tests for common engine contracts and metadata filters."""

from __future__ import annotations

import pytest

from benchvec.engines.base import FilterClause, MetadataFilter


@pytest.mark.unit
def test_metadata_filter_matches_expected_values() -> None:
    metadata: dict[str, str | int | float | bool] = {
        "tag": "engenharia_dados",
        "year": 2025,
        "score": 0.9,
    }
    filters = MetadataFilter(
        clauses=(
            FilterClause(field="tag", operator="eq", value="engenharia_dados"),
            FilterClause(field="year", operator="gte", value=2024),
        )
    )

    assert filters.matches(metadata)


@pytest.mark.unit
def test_metadata_filter_serializes_for_backends() -> None:
    filters = MetadataFilter(
        clauses=(
            FilterClause(field="tag", operator="eq", value="engenharia_dados"),
            FilterClause(field="year", operator="gte", value=2024),
        )
    )

    assert filters.to_chroma_where() == {
        "$and": [
            {"tag": "engenharia_dados"},
            {"year": {"$gte": 2024}},
        ]
    }
    assert filters.to_lancedb_where() == 'tag = "engenharia_dados" AND year >= 2024'
