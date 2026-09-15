"""Tests for user-uploaded CSV/Excel ingestion.

Every case here was reproduced live against a real file before being
written down as a test - two real bugs were found this way (a duplicate
header silently renamed by pandas before the check could see it, and a
NaN that survived JSON serialization) and both are pinned here so they
cannot come back silently.
"""

import json

import pandas as pd
import pytest

from src.agent.data_import import (
    USER_DATA_DB_PATH,
    detect_schema,
    import_dataset,
    list_imported_datasets,
    list_sheets,
    query_imported_dataset,
    refresh_tuik_series,
)


@pytest.fixture(autouse=True)
def _fresh_user_data_db(monkeypatch, tmp_path):
    """Every test gets its own user_data.duckdb, so imports in one test
    can never leak into another - a shared file would make test order
    matter, which is exactly the kind of flakiness a real demo cannot
    afford."""
    import src.agent.data_import as di

    db_path = tmp_path / "user_data_test.duckdb"
    monkeypatch.setattr(di, "USER_DATA_DB_PATH", db_path)
    yield db_path


@pytest.fixture
def csv_path(tmp_path):
    p = tmp_path / "il_hastane.csv"
    p.write_text(
        "Il,Hastane_Sayisi,Yatak_Sayisi,Tarih\n"
        "Ankara,42,15200,2024-01-01\n"
        "Istanbul,88,31500,2024-01-01\n"
        "Izmir,35,11400,2024-01-01\n"
        "Mardin,7,,2024-01-01\n"
        "Sirnak,4,1200,2024-01-01\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def xlsx_path(tmp_path):
    p = tmp_path / "test.xlsx"
    with pd.ExcelWriter(p) as writer:
        pd.DataFrame({"Sehir": ["A", "B"], "Nufus": [100, 200]}).to_excel(writer, sheet_name="Sayfa1", index=False)
        pd.DataFrame({"X": [1, 2, 3]}).to_excel(writer, sheet_name="Sayfa2", index=False)
    return p


# --- schema detection ----------------------------------------------------

def test_detect_schema_infers_types(csv_path):
    schema = detect_schema(csv_path)

    types = {c["column"]: c["inferred_type"] for c in schema["columns"]}
    assert types["Il"] == "text"
    assert types["Hastane_Sayisi"] == "numeric"
    assert types["Tarih"] == "date_like_text"
    assert schema["row_count"] == 5


def test_detect_schema_reports_null_counts(csv_path):
    schema = detect_schema(csv_path)

    yatak = next(c for c in schema["columns"] if c["column"] == "Yatak_Sayisi")
    assert yatak["null_count"] == 1
    assert yatak["non_null_count"] == 4


def test_detect_schema_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        detect_schema("no_such_file.csv")


def test_detect_schema_unsupported_extension(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("just text", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file type"):
        detect_schema(p)


def test_detect_schema_empty_file_is_a_clean_valueerror(tmp_path):
    """pandas raises its own EmptyDataError for this - wrapped so every
    caller sees the same exception family as every other bad-input case."""
    p = tmp_path / "empty.csv"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        detect_schema(p)


def test_duplicate_csv_header_is_rejected_not_silently_renamed(tmp_path):
    """pandas renames a repeated header ('Deger' -> 'Deger', 'Deger.1')
    during the parse itself, which would otherwise hide the collision
    entirely - reproduced live before this test was written."""
    p = tmp_path / "dup.csv"
    p.write_text("Il,Deger,Deger\nAnkara,1,2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate column"):
        detect_schema(p)


def test_duplicate_excel_header_is_also_rejected(tmp_path):
    p = tmp_path / "dup.xlsx"
    pd.DataFrame([[1, 2]], columns=["A", "A"]).to_excel(p, index=False)

    with pytest.raises(ValueError, match="Duplicate column"):
        detect_schema(p)


def test_file_too_large_is_rejected(tmp_path, monkeypatch):
    import src.agent.data_import as di

    monkeypatch.setattr(di, "MAX_FILE_BYTES", 10)
    p = tmp_path / "big.csv"
    p.write_text("a,b\n1,2\n3,4\n5,6\n", encoding="utf-8")

    with pytest.raises(ValueError, match="MB"):
        detect_schema(p)


def test_turkish_characters_survive(tmp_path):
    p = tmp_path / "turkce.csv"
    p.write_text("Şehir,Değer\nİstanbul,5\nAnkara,3\n", encoding="utf-8")

    schema = detect_schema(p)

    assert [c["column"] for c in schema["columns"]] == ["Şehir", "Değer"]


# --- sheet handling --------------------------------------------------------

def test_list_sheets_reports_every_sheet(xlsx_path):
    assert list_sheets(xlsx_path) == ["Sayfa1", "Sayfa2"]


def test_list_sheets_accepts_a_plain_string_path(xlsx_path):
    """A real API inconsistency, found live: detect_schema/import_dataset
    both coerce str -> Path internally but list_sheets did not, so calling
    it with a plain string (the natural thing to do) raised AttributeError."""
    assert list_sheets(str(xlsx_path)) == ["Sayfa1", "Sayfa2"]


def test_list_sheets_on_a_csv_returns_empty():
    assert list_sheets("whatever.csv") == []


def test_detect_schema_respects_the_requested_sheet(xlsx_path):
    schema = detect_schema(xlsx_path, sheet="Sayfa2")
    assert [c["column"] for c in schema["columns"]] == ["X"]


def test_import_dataset_respects_the_requested_sheet(xlsx_path):
    result = import_dataset(xlsx_path, "excel_test", sheet="Sayfa1")
    assert result["row_count"] == 2
    listed = list_imported_datasets()["datasets"][0]
    assert listed["columns"] == ["Sehir", "Nufus"]


# --- dataset_name validation (this is a real table-name SQL surface) ------

@pytest.mark.parametrize("bad_name", [
    "x; DROP TABLE anything; --", "1cannot_start_with_digit", "has space",
    "", None, "a" * 65, "_import_registry",
])
def test_bad_dataset_names_are_rejected(csv_path, bad_name):
    with pytest.raises(ValueError):
        import_dataset(csv_path, bad_name)


def test_valid_dataset_names_are_accepted(csv_path):
    result = import_dataset(csv_path, "valid_name_123")
    assert result["dataset_name"] == "valid_name_123"


# --- import + query --------------------------------------------------------

def test_import_then_query_round_trip(csv_path):
    import_dataset(csv_path, "hastane")

    result = query_imported_dataset("hastane", "SELECT Il FROM hastane ORDER BY Il")

    assert result["row_count"] == 5
    assert [r["Il"] for r in result["rows"]][0] == "Ankara"


def test_null_values_serialize_as_json_null_not_nan(csv_path):
    """The exact bug found live: frame.where(frame.notna(), None) is a
    no-op on a float64 column - pandas coerces the None straight back to
    np.nan, and json.dumps then emits a bare (invalid) NaN token that a
    browser's JSON.parse rejects. astype(object) first is what actually
    fixes it."""
    import_dataset(csv_path, "hastane")

    result = query_imported_dataset("hastane", "SELECT Il, Yatak_Sayisi FROM hastane WHERE Il = 'Mardin'")

    assert result["rows"][0]["Yatak_Sayisi"] is None
    serialized = json.dumps(result, ensure_ascii=False)
    assert "NaN" not in serialized
    assert json.loads(serialized) == result


def test_query_injection_attempt_is_rejected(csv_path):
    import_dataset(csv_path, "hastane")

    with pytest.raises(ValueError, match="single SELECT"):
        query_imported_dataset("hastane", "SELECT * FROM hastane; DROP TABLE hastane;")


def test_query_against_unknown_dataset_lists_what_exists(csv_path):
    import_dataset(csv_path, "hastane")

    with pytest.raises(ValueError, match="hastane"):
        query_imported_dataset("does_not_exist", "SELECT 1")


def test_query_cannot_reach_the_registry_table(csv_path):
    """The registry itself must not be a queryable 'dataset' - it is
    bookkeeping, not something a user uploaded."""
    import_dataset(csv_path, "hastane")

    with pytest.raises(ValueError):
        query_imported_dataset("_import_registry", "SELECT * FROM _import_registry")


def test_reimporting_the_same_name_replaces_it(csv_path, tmp_path):
    import_dataset(csv_path, "hastane")

    smaller = tmp_path / "smaller.csv"
    smaller.write_text("Il,Deger\nTest,1\n", encoding="utf-8")
    import_dataset(smaller, "hastane")

    result = query_imported_dataset("hastane", "SELECT * FROM hastane")
    assert result["row_count"] == 1


def test_row_cap_is_enforced(tmp_path, monkeypatch):
    import src.agent.data_import as di

    monkeypatch.setattr(di, "MAX_ROWS_PER_IMPORT", 3)
    p = tmp_path / "many_rows.csv"
    p.write_text("x\n" + "\n".join(str(i) for i in range(10)), encoding="utf-8")

    result = import_dataset(p, "many_rows")

    assert result["row_count"] == 3
    assert result["truncated_to"] == 3


# --- discovery --------------------------------------------------------------

def test_list_imported_datasets_empty_before_any_import():
    assert list_imported_datasets() == {"datasets": []}


def test_list_imported_datasets_after_import(csv_path):
    import_dataset(csv_path, "hastane")

    datasets = list_imported_datasets()["datasets"]

    assert len(datasets) == 1
    assert datasets[0]["dataset_name"] == "hastane"
    assert datasets[0]["row_count"] == 5
    assert "Il" in datasets[0]["columns"]


def test_list_imported_datasets_shows_every_import(csv_path, xlsx_path):
    import_dataset(csv_path, "hastane")
    import_dataset(xlsx_path, "excel_data", sheet="Sayfa1")

    names = {d["dataset_name"] for d in list_imported_datasets()["datasets"]}

    assert names == {"hastane", "excel_data"}


# --- TUIK live refresh, registry-scoped only --------------------------------

def test_refresh_rejects_an_unregistered_series():
    with pytest.raises(ValueError, match="not a registered TUIK series"):
        refresh_tuik_series("MADE_UP_SERIES_CODE")


def test_refresh_unregistered_series_lists_the_real_ones():
    """A bare rejection would cost the model a round guessing again; the
    list lets it correct itself in the same turn."""
    with pytest.raises(ValueError) as excinfo:
        refresh_tuik_series("NOT_REAL")

    message = str(excinfo.value)
    assert "TUIK_UNEMPLOYMENT_RATE" in message
    assert "TUIK_LIFE_EXPECTANCY" in message


def test_source_label_overrides_the_reported_file_name(csv_path):
    """The web upload endpoint saves files under a random id to avoid
    collisions; without source_label, every uploaded dataset would report
    that meaningless server-side name instead of what the user uploaded."""
    result = import_dataset(csv_path, "hastane", source_label="orijinal_isim.csv")

    assert result["file_name"] == "orijinal_isim.csv"
    listed = list_imported_datasets()["datasets"][0]
    assert listed["source_file"] == "orijinal_isim.csv"


def test_without_source_label_the_path_name_is_used(csv_path):
    result = import_dataset(csv_path, "hastane")
    assert result["file_name"] == csv_path.name
