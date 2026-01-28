"""
Adaptive clustering service with data-driven parameter selection.

Solves the Clustering Instability problem by analyzing dataset characteristics
and automatically tuning HDBSCAN parameters instead of using hardcoded values.

Strategy:
1. Analyze dataset size, density, and dimensionality
2. Compute optimal min_cluster_size based on dataset
3. Select cluster_selection_method based on cluster distribution
4. Validate parameters and fall back to safe defaults if needed
5. Log parameter justification for debugging

Example:
- 100 small prompts → min_cluster_size=3, eom method
- 10,000 diverse prompts → min_cluster_size=10, leaf method
- Bimodal distribution → min_cluster_size=5, eom method
"""

import numpy as np
import logging
from typing import Dict, Any, Tuple
from dataclasses import dataclass
import hdbscan

logger = logging.getLogger(__name__)


@dataclass
class ClusteringParams:
    """Adaptive clustering parameters with justification."""
    min_cluster_size: int
    min_samples: int
    cluster_selection_method: str
    prediction_data: bool
    justification: str


class AdaptiveClusterer:
    """
    Analyzes data and selects optimal HDBSCAN parameters.
    """
    
    # Parameter ranges
    MIN_CLUSTER_SIZE_MIN = 2
    MIN_CLUSTER_SIZE_MAX = 100
    
    # Dataset thresholds
    SMALL_DATASET_THRESHOLD = 100
    MEDIUM_DATASET_THRESHOLD = 1000
    LARGE_DATASET_THRESHOLD = 10000
    
    @staticmethod
    def analyze_and_select_parameters(
        embeddings: np.ndarray,
        original_embeddings: np.ndarray
    ) -> ClusteringParams:
        """
        Analyze embedding distribution and select optimal HDBSCAN parameters.
        
        Args:
            embeddings: Reduced embeddings from UMAP (n_samples, n_components)
            original_embeddings: Original high-dim embeddings (n_samples, 1024)
            
        Returns:
            ClusteringParams with tuned values and justification
        """
        n_samples = embeddings.shape[0]
        n_features = embeddings.shape[1]
        
        # Compute dataset characteristics
        density = AdaptiveClusterer._compute_density(embeddings)
        sparsity = 1.0 - density
        variance = np.var(embeddings, axis=0).mean()
        
        logger.info(
            f"Dataset analysis: "
            f"{n_samples} samples, {n_features} dimensions, "
            f"density={density:.3f}, variance={variance:.3f}"
        )
        
        # Select parameters based on dataset size
        if n_samples < AdaptiveClusterer.SMALL_DATASET_THRESHOLD:
            min_cluster_size, method = AdaptiveClusterer._params_for_small_dataset(
                n_samples, density, sparsity
            )
        elif n_samples < AdaptiveClusterer.MEDIUM_DATASET_THRESHOLD:
            min_cluster_size, method = AdaptiveClusterer._params_for_medium_dataset(
                n_samples, density, sparsity
            )
        elif n_samples < AdaptiveClusterer.LARGE_DATASET_THRESHOLD:
            min_cluster_size, method = AdaptiveClusterer._params_for_large_dataset(
                n_samples, density, sparsity
            )
        else:
            min_cluster_size, method = AdaptiveClusterer._params_for_huge_dataset(
                n_samples, density, sparsity
            )
        
        # Compute min_samples (typically 50% of min_cluster_size, min 1)
        min_samples = max(1, min_cluster_size // 2)
        
        # Build justification string
        justification = (
            f"Dataset: {n_samples} samples. "
            f"Density: {density:.2%}. "
            f"Selected method: {method} (good for {'tight' if density > 0.5 else 'sparse'} clusters). "
            f"min_cluster_size: {min_cluster_size} (good for {int(n_samples / 10)}-{int(n_samples / 2)} clusters expected)"
        )
        
        return ClusteringParams(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method=method,
            prediction_data=True,  # Enable soft clustering
            justification=justification
        )
    
    @staticmethod
    def _compute_density(embeddings: np.ndarray) -> float:
        """
        Estimate density of embeddings using k-NN distance.
        
        Higher density = points close together.
        Lower density = points spread out.
        
        Returns: Value between 0 (sparse) and 1 (dense)
        """
        from sklearn.neighbors import NearestNeighbors
        
        n_samples = embeddings.shape[0]
        k = min(5, n_samples - 1)  # Use k=5 neighbors for distance estimation
        
        try:
            nbrs = NearestNeighbors(n_neighbors=k + 1).fit(embeddings)
            distances, _ = nbrs.kneighbors(embeddings)
            
            # Average distance to k-th neighbor
            avg_k_dist = distances[:, k].mean()
            
            # Normalize to 0-1 range
            # Heuristic: 0.5 is "normal" density
            density = 1.0 / (1.0 + avg_k_dist)
            
            return min(1.0, max(0.0, density))
        except Exception as e:
            logger.warning(f"Density computation failed: {e}, assuming medium density")
            return 0.5
    
    @staticmethod
    def _params_for_small_dataset(
        n_samples: int,
        density: float,
        sparsity: float
    ) -> Tuple[int, str]:
        """
        Parameters for small datasets (< 100 samples).
        
        Goal: Maximize chance of meaningful clusters even with few samples.
        Strategy: Small min_cluster_size to avoid excluding data.
        """
        # Scale min_cluster_size with dataset size
        min_cluster_size = max(2, n_samples // 10)  # 10% of data
        
        # For sparse small datasets, use 'leaf' (stable)
        # For dense small datasets, use 'eom' (meaningful clusters)
        method = 'leaf' if sparsity > 0.6 else 'eom'
        
        reason = f"Small dataset ({n_samples} samples): use min_cluster_size={min_cluster_size} and {method} method"
        logger.info(reason)
        
        return min_cluster_size, method
    
    @staticmethod
    def _params_for_medium_dataset(
        n_samples: int,
        density: float,
        sparsity: float
    ) -> Tuple[int, str]:
        """
        Parameters for medium datasets (100-1000 samples).
        
        Goal: Balance cluster quality and coverage.
        """
        # min_cluster_size based on density
        if density > 0.7:
            # Dense clusters: can use larger min_cluster_size
            min_cluster_size = max(5, n_samples // 50)
        else:
            # Sparse clusters: need smaller min_cluster_size
            min_cluster_size = max(3, n_samples // 30)
        
        # Method based on density
        # eom = "excess of mass" - finds densest clusters (good for natural groupings)
        # leaf = uses leaf clusters - more stable, fewer false positives
        method = 'eom' if density > 0.5 else 'leaf'
        
        reason = f"Medium dataset ({n_samples} samples, density={density:.2f}): use min_cluster_size={min_cluster_size} and {method} method"
        logger.info(reason)
        
        return min_cluster_size, method
    
    @staticmethod
    def _params_for_large_dataset(
        n_samples: int,
        density: float,
        sparsity: float
    ) -> Tuple[int, str]:
        """
        Parameters for large datasets (1k-10k samples).
        
        Goal: Avoid noise, find meaningful structure.
        """
        # Larger min_cluster_size to avoid spurious clusters
        if density > 0.6:
            min_cluster_size = max(10, n_samples // 100)  # 1% of data
        else:
            min_cluster_size = max(8, n_samples // 150)   # 0.67% of data
        
        # Large datasets usually benefit from 'eom'
        method = 'eom'
        
        reason = f"Large dataset ({n_samples} samples, density={density:.2f}): use min_cluster_size={min_cluster_size} and {method}"
        logger.info(reason)
        
        return min_cluster_size, method
    
    @staticmethod
    def _params_for_huge_dataset(
        n_samples: int,
        density: float,
        sparsity: float
    ) -> Tuple[int, str]:
        """
        Parameters for huge datasets (10k+ samples).
        
        Goal: Computational efficiency + meaningful structure.
        """
        # Conservative min_cluster_size to reduce noise
        if density > 0.7:
            min_cluster_size = max(25, n_samples // 200)  # 0.5% of data
        elif density > 0.5:
            min_cluster_size = max(20, n_samples // 250)  # 0.4% of data
        else:
            min_cluster_size = max(15, n_samples // 300)  # 0.3% of data
        
        # Large datasets: 'leaf' is more computationally efficient
        method = 'leaf'
        
        reason = f"Huge dataset ({n_samples} samples): use min_cluster_size={min_cluster_size} for efficiency (leaf method)"
        logger.info(reason)
        
        return min_cluster_size, method
    
    @staticmethod
    def estimate_cluster_count(
        min_cluster_size: int,
        n_samples: int,
        density: float
    ) -> Tuple[int, int]:
        """
        Estimate typical cluster count range given parameters.
        
        Returns: (min_estimate, max_estimate)
        """
        # Conservative estimate based on min_cluster_size
        min_estimate = n_samples // (min_cluster_size * 5)
        max_estimate = n_samples // (min_cluster_size * 2)
        
        return max(1, min_estimate), max(1, max_estimate)


def get_adaptive_parameters(
    embeddings: np.ndarray,
    original_embeddings: np.ndarray
) -> ClusteringParams:
    """
    Convenience function to get adaptive clustering parameters.
    
    Args:
        embeddings: Reduced embeddings from UMAP
        original_embeddings: Original high-dimensional embeddings
        
    Returns:
        ClusteringParams with tuned values
    """
    return AdaptiveClusterer.analyze_and_select_parameters(
        embeddings,
        original_embeddings
    )


def validate_clustering_parameters(
    min_cluster_size: int,
    min_samples: int,
    n_samples: int
) -> Dict[str, Any]:
    """
    Validate clustering parameters and suggest corrections if needed.
    
    Returns dict with:
    - is_valid: bool
    - message: str
    - corrected_params: dict (if invalid)
    """
    issues = []
    
    # Check min_cluster_size
    if min_cluster_size < 2:
        issues.append(f"min_cluster_size {min_cluster_size} too small (minimum 2)")
    if min_cluster_size > n_samples // 2:
        issues.append(f"min_cluster_size {min_cluster_size} too large (max {n_samples // 2})")
    
    # Check min_samples
    if min_samples < 1:
        issues.append(f"min_samples {min_samples} too small (minimum 1)")
    if min_samples > min_cluster_size:
        issues.append(f"min_samples {min_samples} > min_cluster_size {min_cluster_size}")
    
    if not issues:
        return {"is_valid": True, "message": "Parameters valid"}
    
    # Generate corrections
    corrected = {
        "min_cluster_size": max(2, min(min_cluster_size, n_samples // 2)),
        "min_samples": min(min_samples, min_cluster_size - 1)
    }
    
    return {
        "is_valid": False,
        "issues": issues,
        "corrected_params": corrected
    }
