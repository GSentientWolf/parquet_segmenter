"""
Proposed API improvement for blob_df.py outlier placement

Current confusing API:
    build_blob_dataframe(..., 
                        spread_top_outliers=True,     # conflicts with below
                        cluster_batches=2,            # ignored when spread=True  
                        contiguous_within_batch=True) # ignored when spread=True

Proposed clean API:
    build_blob_dataframe(..., outlier_strategy=OutlierStrategy.SPREAD)
    build_blob_dataframe(..., outlier_strategy=OutlierStrategy.SINGLE_BATCH)
    build_blob_dataframe(..., outlier_strategy=OutlierStrategy.CONTIGUOUS)
    build_blob_dataframe(..., outlier_strategy=OutlierStrategy.MULTI_CLUSTER, cluster_count=3)
"""

from enum import StrEnum

class OutlierStrategy(StrEnum):
    """Strategy for placing large outlier blobs in the DataFrame."""
    
    # Spread outliers across different batches (current spread_top_outliers=True)
    SPREAD = "spread"
    
    # Place all outliers in a single random batch, spread within batch  
    SINGLE_BATCH = "single"
    
    # Place all outliers contiguously in a single batch
    CONTIGUOUS = "contiguous"
    
    # Distribute outliers across multiple clusters
    MULTI_CLUSTER = "multi"

# Example usage:
def build_blob_dataframe_new_api(
    *,
    total_df_size: int,
    # ... other params stay the same
    top_outliers: int = 1,
    outlier_strategy: OutlierStrategy = OutlierStrategy.SPREAD,
    cluster_count: int = 1,  # Only used with MULTI_CLUSTER strategy
    seed: Optional[int] = None,
    # ... rest of params
):
    """
    Clear, unambiguous API:
    - outlier_strategy controls HOW outliers are placed
    - cluster_count only matters for MULTI_CLUSTER strategy
    - No conflicting parameters
    """
    pass

print("This approach gives you:")
print("✅ Single clear parameter for outlier placement")
print("✅ No conflicting boolean combinations") 
print("✅ Self-documenting strategy names")
print("✅ Easy to extend with new strategies")
print("✅ Backwards compatibility possible with deprecation warnings")
