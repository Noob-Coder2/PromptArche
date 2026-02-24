
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
    
    # 1. Cleanup existing duplicates to ensure clean state
    _cleanup_duplicate_clusters(supabase, user_id)
    
    # 2. Fetch ALL embeddings (paginated to avoid Supabase row limits)
    PAGE_SIZE = 1000
    data = []
    offset = 0
    while True:
        response = supabase.table("prompts") \
            .select("id, embedding") \
            .eq("user_id", str(user_id)) \
            .not_.is_("embedding", "null") \
            .range(offset, offset + PAGE_SIZE - 1) \
            .execute()
        batch = response.data or []
        data.extend(batch)
        if len(batch) < PAGE_SIZE:
            break  # Last page
        offset += PAGE_SIZE
    
    logger.info(f"Fetched {len(data)} prompts with embeddings for user {user_id}")
    
    if not data or len(data) < 5:  # Need minimal data for clustering
        return {"status": "skipped", "message": "Not enough data (minimum 5 embeddings required)"}
        
    df = pd.DataFrame(data)
    
    # Parse vectors if they are strings (Supabase return format)
    if not df.empty:
         df['embedding'] = df['embedding'].apply(_safe_parse_embedding)
    
    # Ensure all items are numpy arrays of floats
    embeddings = np.stack(df['embedding'].values)
    ids = df['id'].tolist()
    
    logger.info(f"Starting clustering for user {user_id} with {len(embeddings)} prompts")
    
    # 3. UMAP - Dimension Reduction
    reducer = umap.UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.1,
        metric='cosine',
        random_state=42
    )
    embedding_reduced = reducer.fit_transform(embeddings)
    
    # 4. Adaptive HDBSCAN - Clustering with data-driven parameters
    # Select parameters based on dataset characteristics
    params = get_adaptive_parameters(embedding_reduced, embeddings)
    
    logger.info(
        f"Using adaptive clustering parameters: "
        f"min_cluster_size={params.min_cluster_size}, "
        f"min_samples={params.min_samples}, "
        f"method={params.cluster_selection_method}"
    )
    
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
    
    # 5. Fetch existing clusters for incremental update
    existing_res = supabase.table("clusters") \
        .select("id, label, description, centroid") \
        .eq("user_id", str(user_id)) \
        .execute()
    existing_clusters = existing_res.data or []
    
    logger.info(f"Found {len(existing_clusters)} existing clusters for user {user_id}")
    
    # Determine the next available cluster index
    next_cluster_index = _get_next_cluster_index(existing_clusters)
    logger.info(f"Next new cluster will be named 'Cluster {next_cluster_index}'")
    
    # Track which existing clusters are still valid
    used_existing_ids = set()
    cluster_map = {}  # new_label (HDBSCAN id) -> cluster_uuid
    unique_labels = set(cluster_labels)
    
    # 6. Process each new cluster
    clusters_created = 0
    clusters_updated = 0
    
    # Sort labels to keep processing deterministic
    sorted_labels = sorted([l for l in unique_labels if l != -1])
    
    for label in sorted_labels:
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
            
            try:
                supabase.table("clusters").update({
                    "centroid": centroid.tolist()
                }).eq("id", cluster_id).execute()
                
                cluster_map[label] = cluster_id
                clusters_updated += 1
                logger.debug(f"Updated existing cluster {matched_cluster['label']}")
            except Exception as e:
                logger.error(f"Failed to update cluster {cluster_id}: {e}")
        else:
            # Create new cluster with SEQUENTIAL numbering
            new_label_str = f"Cluster {next_cluster_index}"
            next_cluster_index += 1
            
            cluster_data = {
                "user_id": str(user_id),
                "label": new_label_str,
                "description": "Auto-generated cluster",
                "centroid": centroid.tolist()
            }
            try:
                res = supabase.table("clusters").insert(cluster_data).execute()
                if res.data:
                    cluster_id = res.data[0]['id']
                    cluster_map[label] = cluster_id
                    clusters_created += 1
                    logger.info(f"Created new cluster '{new_label_str}'")
            except Exception as e:
                logger.error(f"Failed to create cluster '{new_label_str}': {e}")
    
    # 7. Handle orphaned clusters (no longer have matching prompts)
    orphaned_count = 0
    for existing in existing_clusters:
        if existing["id"] not in used_existing_ids:
            # Could delete or keep - for now, just log
            # logger.info(f"Orphaned cluster: {existing['label']} (id: {existing['id']})")
            orphaned_count += 1
    
    # 8. Batch update prompts with cluster assignments
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
    
    # Helper to chunk updates and prevent SQL limits
    def chunked_update(ids_list, update_data, batch_size=100):
        total = len(ids_list)
        for i in range(0, total, batch_size):
            batch = ids_list[i : i + batch_size]
            try:
                supabase.table("prompts").update(update_data).in_("id", batch).execute()
                logger.debug(f"Updated batch {i//batch_size + 1}: {len(batch)} prompts")
            except Exception as e:
                logger.error(f"Failed to update batch starting at index {i}: {e}")

    # Batch update: mark noise prompts as unclustered
    if noise_ids:
        logger.info(f"Marking {len(noise_ids)} prompts as noise (unclustered)")
        chunked_update(noise_ids, {"cluster_id": None})
    
    # Batch update: assign prompts to clusters
    logger.info(f"Assigning {assigned_count} prompts to {len(cluster_assignments)} clusters")
    for cluster_uuid, prompt_ids in cluster_assignments.items():
        chunked_update(prompt_ids, {"cluster_id": cluster_uuid})
    
    total_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    
    logger.info(
        f"Clustering complete for user {user_id}: "
        f"{total_clusters} clusters found ({clusters_created} new, {clusters_updated} updated), "
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
            
        existing_centroid_raw = existing.get("centroid")
        if not existing_centroid_raw:
            continue
            
        try:
            existing_arr = _safe_parse_embedding(existing_centroid_raw)
            if existing_arr.size == 0:
                 continue
                 
            existing_arr = existing_arr.reshape(1, -1)
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


def _get_next_cluster_index(existing_clusters: List[Dict]) -> int:
    """
    Determine the next sequential cluster index (e.g., 13 for "Cluster 12").
    Parses "Cluster X" labels from the existing list.
    """
    max_index = -1
    for cluster in existing_clusters:
        label = cluster.get("label", "")
        if label.startswith("Cluster "):
            try:
                # Extract number part "Cluster 12" -> 12
                # Handle potential edge cases where label might be "Cluster 12 (old)"
                parts = label.split(" ")
                if len(parts) >= 2 and parts[1].isdigit():
                    idx = int(parts[1])
                    if idx > max_index:
                        max_index = idx
            except ValueError:
                continue
    
    return max_index + 1


def _cleanup_duplicate_clusters(supabase, user_id: UUID) -> None:
    """
    Finds clusters with duplicate names (e.g. "Cluster 0") and renames older ones.
    Keeps the most recently created/updated one as the canonical "Cluster X".
    """
    try:
        # Fetch all clusters with basic info
        res = supabase.table("clusters") \
            .select("id, label, created_at") \
            .eq("user_id", str(user_id)) \
            .execute()
        
        clusters = res.data or []
        if not clusters:
            return

        # Group by label
        by_label = {}
        for c in clusters:
            lbl = c["label"]
            if lbl not in by_label:
                by_label[lbl] = []
            by_label[lbl].append(c)

        # Check for duplicates
        duplicates_found = 0
        for label, group in by_label.items():
            if len(group) > 1:
                # Sort by created_at DESC (assuming newer is better/canonical)
                # If created_at is missing or same, fallback to ID sort
                group.sort(key=lambda x: x.get("created_at", "") or "", reverse=True)
                
                # Keep the first one (newest), rename the rest
                canonical = group[0]
                others = group[1:]
                
                for other in others:
                    new_label = f"{label}_dup_{other['id'][:4]}"
                    logger.info(f"Renaming duplicate cluster '{label}' ({other['id']}) to '{new_label}'")
                    
                    supabase.table("clusters").update({
                        "label": new_label,
                        "description": f"Duplicate of {canonical['id']} (auto-renamed)"
                    }).eq("id", other['id']).execute()
                    
                    duplicates_found += 1
        
        if duplicates_found > 0:
            logger.info(f"Cleaned up {duplicates_found} duplicate clusters for user {user_id}")
            
    except Exception as e:
        logger.error(f"Error cleaning up duplicate clusters: {e}")


def _safe_parse_embedding(value: Any) -> np.ndarray:
    """
    Robustly parse an embedding/centroid from various formats:
    - JSON string "[0.1, 0.2]"
    - List of strings ["0.1", "0.2"]
    - List of floats [0.1, 0.2]
    - Numpy array
    Returns empty array on failure.
    """
    if value is None:
        return np.array([])
        
    try:
        # If it's already a numpy array, just ensure type
        if isinstance(value, np.ndarray):
            # Check if it contains strings
            if value.dtype.kind in {'U', 'S'}: # Unicode or String
                # Try to convert content to float
                return value.astype(float)
            return value.astype(float)
            
        # If it's a string, try JSON then string parsing
        if isinstance(value, (str, np.str_)):
            s = str(value).strip()
            if not s: return np.array([])
            try:
                # Try JSON first
                return np.array(json.loads(s), dtype=float)
            except (json.JSONDecodeError, TypeError):
                # Fallback to string parsing (comma separated)
                s = s.strip("[]")
                if not s: return np.array([])
                return np.fromstring(s, sep=",")
                
        # If it's a list (could be list of floats or strings)
        if isinstance(value, list):
            # Convert to numpy array of floats
            return np.array(value, dtype=float)
            
        return np.array([])
    except Exception as e:
        # logger.warning(f"Failed to parse embedding: {e}") # Reduce noise
        return np.array([])

