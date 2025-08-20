import pandas as pd

from parquet_segmenter.functional_testing.blob_df import build_blob_dataframe, ChunkSize, OutlierStrategy


def test_blob_df_single_outlier():
    # For small total (1 MB) and default outlier_mb_rate=10, we expect top_outliers=1
    df = build_blob_dataframe(total_df_size=ChunkSize.ONE_MB, seed=42)
    assert isinstance(df, pd.DataFrame)
    assert "is_outlier" in df.columns
    assert df["is_outlier"].sum() == 1


def test_blob_df_deterministic():
    df1 = build_blob_dataframe(total_df_size=5 * ChunkSize.ONE_MB, seed=123)
    df2 = build_blob_dataframe(total_df_size=5 * ChunkSize.ONE_MB, seed=123)
    # sizes and outlier flags should be identical for the same seed
    assert df1["blob_size"].tolist() == df2["blob_size"].tolist()
    assert df1["is_outlier"].tolist() == df2["is_outlier"].tolist()


def test_blob_df_outlier_count_large():
    # For 30 MB and outlier_mb_rate=10 we expect floor(30/10)=3 outliers
    df = build_blob_dataframe(total_df_size=30 * ChunkSize.ONE_MB, seed=7)
    assert df["is_outlier"].sum() == 3


def test_two_outliers_same_batch_nonadjacent():
    # Ensure two top outliers are spread across distinct batches and are not adjacent
    top = 2
    df = build_blob_dataframe(total_df_size=10 * ChunkSize.ONE_MB, top_outliers=top, seed=42)
    out_idx = [i for i, v in enumerate(df["is_outlier"]) if v]
    assert len(out_idx) >= top

    # Recompute batch_rows the same way as the builder does (approx)
    small_bins = tuple(map(lambda x: x * ChunkSize.ONE_KB * x, [5, 8, 9, 10, 14, 16]))
    avg_small = int(sum(small_bins) / len(small_bins))
    batch_rows = max(1, ChunkSize.ONE_MB // max(1, avg_small))

    batches = [idx // batch_rows for idx in out_idx[:top]]
    # both should be in different batches
    assert len(set(batches)) == top

    # ensure they're not adjacent (there is at least one small sample between them)
    a, b = sorted(out_idx[:top])
    assert abs(a - b) > 1


def test_multiple_intermediate_cluster_in_batch():
    # Verify that multiple intermediate blobs can cluster within the same batch
    df = build_blob_dataframe(total_df_size=8 * ChunkSize.ONE_MB, seed=123)

    # Recompute batch_rows to partition df into batches
    small_bins = tuple(map(lambda x: x * ChunkSize.ONE_KB * x, [5, 8, 9, 10, 14, 16]))
    avg_small = int(sum(small_bins) / len(small_bins))
    batch_rows = max(1, ChunkSize.ONE_MB // max(1, avg_small))

    # Check each batch for clustered intermediate blobs
    found = False
    for b_start in range(0, len(df), batch_rows):
        window = df["is_intermediate"].iloc[b_start : b_start + batch_rows]
        if window.sum() >= 2:
            found = True
            break

    assert found, "expected at least one batch to contain multiple intermediate blobs"


def test_cluster_single_batch_contiguous():
    # cluster all top_outliers into a single batch contiguously
    top = 3
    df = build_blob_dataframe(
        total_df_size=10 * ChunkSize.ONE_MB, 
        top_outliers=top, 
        outlier_strategy=OutlierStrategy.CONTIGUOUS, 
        seed=99
    )
    out_idx = [i for i, v in enumerate(df["is_outlier"]) if v]
    assert len(out_idx) >= top
    # check that at least `top` outliers contain a contiguous block
    found_block = False
    for i in range(len(out_idx) - top + 1):
        if out_idx[i + top - 1] - out_idx[i] == top - 1:
            found_block = True
            break
    assert found_block, "expected a contiguous block of outliers in a single batch"


def test_cluster_two_batches_distribution():
    # cluster into two batches; expect top_outliers distributed across two batches
    top = 4
    df = build_blob_dataframe(
        total_df_size=20 * ChunkSize.ONE_MB, 
        top_outliers=top, 
        outlier_strategy=OutlierStrategy.MULTI_CLUSTER,
        cluster_count=2,
        seed=7
    )
    out_idx = [i for i, v in enumerate(df["is_outlier"]) if v]
    assert len(out_idx) >= top

    # Recompute batch_rows
    small_bins = tuple(map(lambda x: x * ChunkSize.ONE_KB * x, [5, 8, 9, 10, 14, 16]))
    avg_small = int(sum(small_bins) / len(small_bins))
    batch_rows = max(1, ChunkSize.ONE_MB // max(1, avg_small))

    batches = [idx // batch_rows for idx in out_idx[:top]]
    assert len(set(batches)) >= 2


def test_single_batch_strategy():
    # Test SINGLE_BATCH strategy - all outliers in one batch, spread within
    top = 3
    df = build_blob_dataframe(
        total_df_size=15 * ChunkSize.ONE_MB, 
        top_outliers=top, 
        outlier_strategy=OutlierStrategy.SINGLE_BATCH, 
        seed=42
    )
    out_idx = [i for i, v in enumerate(df["is_outlier"]) if v]
    assert len(out_idx) >= top
    
    # Recompute batch_rows using the same defaults as the function
    small_bins = tuple(map(lambda x: x * ChunkSize.ONE_KB * x, [5, 8, 9, 10, 14, 16]))
    avg_small = int(sum(small_bins) / len(small_bins))
    batch_rows = max(1, ChunkSize.ONE_MB // max(1, avg_small))

    batches = [idx // batch_rows for idx in out_idx[:top]]
    assert len(set(batches)) == 1, "All outliers should be in the same batch"


def test_spread_strategy_default():
    # Test that SPREAD is the default strategy and spreads outliers across batches
    top = 3
    df = build_blob_dataframe(
        total_df_size=15 * ChunkSize.ONE_MB, 
        top_outliers=top, 
        seed=42
    )  # Uses default OutlierStrategy.SPREAD
    out_idx = [i for i, v in enumerate(df["is_outlier"]) if v]
    assert len(out_idx) >= top
    
    # Recompute batch_rows using the same defaults as the function
    small_bins = tuple(map(lambda x: x * ChunkSize.ONE_KB * x, [5, 8, 9, 10, 14, 16]))
    avg_small = int(sum(small_bins) / len(small_bins))
    batch_rows = max(1, ChunkSize.ONE_MB // max(1, avg_small))

    batches = [idx // batch_rows for idx in out_idx[:top]]
    # With enough data and outliers, they should be spread across different batches
    assert len(set(batches)) >= 2, "Outliers should be spread across different batches"
