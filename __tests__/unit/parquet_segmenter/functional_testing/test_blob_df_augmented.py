import pickle
import datetime
from zoneinfo import ZoneInfo

import pandas as pd
from pandas.api import types as ptypes
from parquet_segmenter.testing_utils import to_int_scalar

from parquet_segmenter.functional_testing.blob_df import (
    build_blob_dataframe,
    augment_with_faker_and_timestamps,
    ChunkSize,
)


def test_augment_with_faker_and_timestamps_fields_and_size():
    # Build a small deterministic blob dataframe
    df = build_blob_dataframe(
        total_df_size=10 * ChunkSize.ONE_KB,
        batch_size=ChunkSize.ONE_KB,
        seed=42,
        top_outliers=0,
    )

    # Sanity: expect at least one row
    assert len(df) >= 1

    initial = datetime.datetime(2020, 1, 1, 0, 0, 0)

    merged = augment_with_faker_and_timestamps(
        df,
        initial_date=initial,
        faker_seed=123,
        tz="UTC",
    )

    # Left-side fake columns exist
    for col in ("first_name", "last_name", "email", "company", "timestamp"):
        assert col in merged.columns

    # 'row_size' temporary column must be dropped
    assert "row_size" not in merged.columns

    # Timestamps should be timezone-aware and equal to initial + i microseconds
    tzinfo = ZoneInfo("UTC")
    for i, ts in enumerate(merged["timestamp"].tolist()):
        assert isinstance(ts, datetime.datetime)
        assert ts.tzinfo is not None
        # Check timezone is UTC (or equivalent)
        assert ts.tzinfo == tzinfo
        expected = (initial.replace(tzinfo=tzinfo) + datetime.timedelta(microseconds=i))
        assert ts == expected

    # Validate the 'size' column equals serialized left-row size + blob_size
    assert "size" in merged.columns

    # Compute the serialized size of the fake left-side columns directly
    # from the returned DataFrame using the same set of columns the
    # implementation generates in `left_df`.
    left_cols = [
        "first_name",
        "last_name",
        "dob",
        "age",
        "ssn",
        "address",
        "latitude",
        "longitude",
        "city",
        "country",
        "time_zone",
        "email",
        "company",
        "job",
        "timestamp",
    ]

    # Relax ordering: require expected left-side fields to exist somewhere in
    # the merged DataFrame. The implementation may reorder or add extra
    # fields; this test checks presence, not exact position.
    required_left_cols = {"first_name", "last_name", "email", "timestamp"}
    assert required_left_cols.issubset(set(merged.columns))

    # Also ensure the full set of expected left columns is present (allowing
    # additional fields and not enforcing exact ordering).
    assert set(left_cols).issubset(set(merged.columns))
    row_sizes = merged[left_cols].apply(
        lambda r: len(pickle.dumps(r.to_dict(), protocol=pickle.HIGHEST_PROTOCOL)),
        axis=1,
    )

    for i, val in enumerate(
        merged.reset_index(drop=True)[["blob_size", "size"]].itertuples(index=False)
    ):
        blob_size = to_int_scalar(val.blob_size)
        size = to_int_scalar(val.size)
        assert size == blob_size + int(row_sizes.iloc[i])

    # Timestamps: verify timezone-awareness and that the set of timestamps
    # corresponds to the monotonic sequence initial + i microseconds (order
    # within the DataFrame is not enforced above).
    tzinfo = ZoneInfo("UTC")
    ts_list = list(merged["timestamp"].tolist())
    assert all(isinstance(t, datetime.datetime) for t in ts_list)
    assert all((t.tzinfo is not None) for t in ts_list)
    # Sorted timestamps should equal initial + i microseconds
    sorted_ts = sorted(ts_list)
    for i, ts in enumerate(sorted_ts):
        expected = initial.replace(tzinfo=tzinfo) + datetime.timedelta(microseconds=i)
        assert ts == expected

    # Column dtype validations
    # Integer-like columns
    assert ptypes.is_integer_dtype(merged["blob_size"])
    assert ptypes.is_integer_dtype(merged["size"])
    assert ptypes.is_integer_dtype(merged["age"]) or merged["age"].apply(lambda v: isinstance(v, int)).all()

    # Float-like columns: accept numeric dtype or string values that can be
    # converted to floats without producing NaN.
    lat_num = pd.to_numeric(merged["latitude"], errors="coerce")
    lon_num = pd.to_numeric(merged["longitude"], errors="coerce")
    assert lat_num.notna().all()
    assert lon_num.notna().all()

    # String-like columns: accept pandas string dtype or object dtype with str values
    string_cols = [
        "first_name",
        "last_name",
        "email",
        "company",
        "job",
        "ssn",
        "address",
        "city",
        "country",
        "time_zone",
    ]
    for col in string_cols:
        assert ptypes.is_string_dtype(merged[col]) or merged[col].apply(lambda v: isinstance(v, str)).all()

    # DOB is a date object per Faker API
    import datetime as _dt

    assert merged["dob"].apply(lambda v: isinstance(v, _dt.date)).all()
