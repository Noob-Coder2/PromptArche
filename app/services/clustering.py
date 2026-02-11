
import numpy as np
import pandas as pd
import json
import logging
from uuid import UUID
from typing import Dict, List, Optional, Any

import umap
import hdbscan
from sklearn.metrics.pairwise import cosine_similarity
from app.db.supabase import get_supabase
from app.services.adaptive_clustering import get_adaptive_parameters, validate_clustering_parameters

logger = logging.getLogger(__name__)

# Similarity threshold for matching new clusters to existing ones
CLUSTER_SIMILARITY_THRESHOLD = 0.85


def run_clustering_for_user(user_id: UUID) -> Dict[str, Any]:
    """
    Fetches user's embeddings, runs UMAP + HDBSCAN, and updates clusters.
    Uses incremental upsert strategy to preserve existing cluster labels.
    
    Args:
        user_id: The user's UUID
        
    Returns:
        Status dict with clustering results
    """
    supabase = get_supabase()
    
    # 1. Fetch embeddings
    response = supabase.table("prompts") \
        .select("id, embedding") \
        .eq("user_id", str(user_id)) \
        .not_.is_("embedding", "null") \
        .execute()
    data = response.data
    
    if not data or len(data) < 5:  # Need minimal data for clustering
        return {"status": "skipped", "message": "Not enough data (minimum 5 embeddings required)"}
        
    df = pd.DataFrame(data)
    embeddings = np.array(df['embedding'].tolist())
    ids = df['id'].tolist()
    
    # 2. UMAP - Dimension Reduction
    reducer = umap.UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.1,
        metric='cosine',
        random_state=42
    )
    embedding_reduced = reducer.fit_transform(embeddings)
    
    # 3. Adaptive HDBSCAN - Clustering with data-driven parameters
    # Select parameters based on dataset characteristics
    params = get_adaptive_parameters(embedding_reduced, embeddings)
    
    logger.info(
        f"Using adaptive clustering parameters: "
        f"min_cluster_size={params.min_cluster_size}, "
        f"min_samples={params.min_samples}, "
        f"method={params.cluster_selection_method}"
    )
    logger.info(f"Justification: {params.justification}")
    
    # Validate parameters before use
    validation = validate_clustering_parameters(
        params.min_cluster_size,
        params.min_samples,
        n_samples=len(embeddings)
    )
    
    if not validation["is_valid"]:
        logger.warning(f"Parameter validation failed: {validation['issues']}")
        logger.warning(f"Using corrected parameters: {validation['corrected_params']}")
        params.min_cluster_size = validation["corrected_params"]["min_cluster_size"]
        params.min_samples = validation["corrected_params"]["min_samples"]
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=params.min_cluster_size,
        min_samples=params.min_samples,
        metric='euclidean',
        cluster_selection_method=params.cluster_selection_method,
        prediction_data=params.prediction_data
    )
    cluster_labels = clusterer.fit_predict(embedding_reduced)
    
    # 4. Fetch existing clusters for incremental update
    existing_res = supabase.table("clusters") \
        .select("id, label, description, centroid") \
        .eq("user_id", str(user_id)) \
        .execute()
    existing_clusters = existing_res.data or []
    
    logger.info(f"Found {len(existing_clusters)} existing clusters for user {user_id}")
    
    # Track which existing clusters are still valid
    used_existing_ids = set()
    cluster_map = {}  # new_label -> cluster_uuid
    unique_labels = set(cluster_labels)
    
    # 5. Process each new cluster
    clusters_created = 0
    clusters_updated = 0
    
    for label in unique_labels:
        if label == -1:
            continue  # Handle noise separately
            
        # Calculate centroid in original high-dim space
        mask = (cluster_labels == label)
        cluster_points = embeddings[mask]
        centroid = np.mean(cluster_points, axis=0)
        
        # Try to match to existing cluster using cosine similarity
        matched_cluster = _find_matching_cluster(centroid, existing_clusters, used_existing_ids)
        
        if matched_cluster:
            # Update existing cluster centroid, preserve label/description
            cluster_id = matched_cluster["id"]
            used_existing_ids.add(cluster_id)
            
            supabase.table("clusters").update({
                "centroid": centroid.tolist()
            }).eq("id", cluster_id).execute()
            
            cluster_map[label] = cluster_id
            clusters_updated += 1
            logger.debug(f"Updated existing cluster {matched_cluster['label']}")
        else:
            # Create new cluster
            cluster_data = {
                "user_id": str(user_id),
                "label": f"Cluster {label}",
                "description": "Auto-generated cluster",
                "centroid": centroid.tolist()
            }
            res = supabase.table("clusters").insert(cluster_data).execute()
            if res.data:
                cluster_id = res.data[0]['id']
                cluster_map[label] = cluster_id
                clusters_created += 1
                logger.debug(f"Created new cluster {label}")
    
    # 6. Handle orphaned clusters (no longer have matching prompts)
    orphaned_count = 0
    for existing in existing_clusters:
        if existing["id"] not in used_existing_ids:
            # Could delete or keep - for now, keep but log
            logger.info(f"Orphaned cluster: {existing['label']} (id: {existing['id']})")
            orphaned_count += 1
    
    # 7. Batch update prompts with cluster assignments
    noise_count = 0
    assigned_count = 0
    
    # Collect noise and assigned prompt IDs
    noise_ids = []
    # Group assigned prompts by cluster UUID for batch updates
    cluster_assignments: Dict[str, list] = {}  # cluster_uuid -> [prompt_ids]
    
    for idx, prompt_id in enumerate(ids):
        label = cluster_labels[idx]
        
        if label == -1:
            noise_ids.append(prompt_id)
            noise_count += 1
        else:
            cluster_uuid = cluster_map.get(label)
            if cluster_uuid:
                if cluster_uuid not in cluster_assignments:
                    cluster_assignments[cluster_uuid] = []
                cluster_assignments[cluster_uuid].append(prompt_id)
                assigned_count += 1
    
    # Batch update: mark noise prompts as unclustered (single DB call)
    if noise_ids:
        try:
            supabase.table("prompts").update({
                "cluster_id": None,
                "metadata": {"unclustered": True}
            }).in_("id", noise_ids).execute()
            logger.info(f"Batch-marked {len(noise_ids)} prompts as unclustered")
        except Exception as e:
            logger.warning(f"Failed to batch-mark noise prompts: {e}")
    
    # Batch update: assign prompts to clusters (one call per cluster)
    for cluster_uuid, prompt_ids in cluster_assignments.items():
        try:
            supabase.table("prompts").update({
                "cluster_id": cluster_uuid,
                "metadata": {"unclustered": False}
            }).in_("id", prompt_ids).execute()
        except Exception as e:
            logger.warning(f"Failed to batch-assign prompts to cluster {cluster_uuid}: {e}")
    
    total_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    
    logger.info(
        f"Clustering complete for user {user_id}: "
        f"{total_clusters} clusters ({clusters_created} new, {clusters_updated} updated), "
        f"{assigned_count} prompts assigned, {noise_count} noise points"
    )
    
    return {
        "status": "success",
        "clusters_found": total_clusters,
        "clusters_created": clusters_created,
        "clusters_updated": clusters_updated,
        "prompts_assigned": assigned_count,
        "noise_points": noise_count,
        "orphaned_clusters": orphaned_count
    }


def _find_matching_cluster(
    centroid: np.ndarray,
    existing_clusters: List[Dict],
    used_ids: set
) -> Optional[Dict]:
    """
    Find an existing cluster that matches the new centroid.
    
    Args:
        centroid: The new cluster centroid
        existing_clusters: List of existing cluster dicts with 'id', 'centroid'
        used_ids: Set of already-matched cluster IDs to exclude
        
    Returns:
        Matching cluster dict or None
    """
    best_match = None
    best_similarity = 0.0
    
    for existing in existing_clusters:
        if existing["id"] in used_ids:
            continue
            
        existing_centroid = existing.get("centroid")
        if not existing_centroid:
            continue
            
        try:
            # Compute cosine similarity
            existing_arr = np.array(existing_centroid).reshape(1, -1)
            new_arr = centroid.reshape(1, -1)
            similarity = cosine_similarity(new_arr, existing_arr)[0][0]
            
            if similarity >= CLUSTER_SIMILARITY_THRESHOLD and similarity > best_similarity:
                best_similarity = similarity
                best_match = existing
        except Exception as e:
            logger.warning(f"Error computing similarity for cluster {existing['id']}: {e}")
            continue
    
    if best_match:
        logger.debug(f"Matched cluster {best_match['label']} with similarity {best_similarity:.3f}")
    
    return best_match


def get_unclustered_prompts(user_id: str, limit: int = 100) -> List[Dict]:
    """
    Get prompts that were marked as noise/unclustered.
    Useful for understanding what patterns aren't being captured.
    
    Args:
        user_id: The user's UUID
        limit: Maximum prompts to return
        
    Returns:
        List of unclustered prompt dicts
    """
    supabase = get_supabase()
    try:
        res = supabase.table("prompts") \
            .select("id, content, created_at") \
            .eq("user_id", user_id) \
            .is_("cluster_id", "null") \
            .limit(limit) \
            .execute()
        return res.data or []
    except Exception as e:
        logger.error(f"Failed to get unclustered prompts: {e}")
        return []

