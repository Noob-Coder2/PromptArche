
import numpy as np
import pandas as pd
import json
import logging
from uuid import UUID

import umap
import hdbscan
from app.db.supabase import get_supabase

logger = logging.getLogger(__name__)

def run_clustering_for_user(user_id: UUID):
    """
    Fetches user's embeddings, runs UMAP + HDBSCAN, and updates clusters.
    """
    supabase = get_supabase()
    
    # 1. Fetch data
    # We need embeddings.
    response = supabase.table("prompts").select("id, embedding").eq("user_id", str(user_id)).not_.is_("embedding", "null").execute()
    data = response.data
    
    if not data or len(data) < 5: # Need minimal data for clustering
        return {"status": "skipped", "message": "Not enough data"}
        
    df = pd.DataFrame(data)
    embeddings = np.array(df['embedding'].tolist())
    ids = df['id'].tolist()
    
    # 2. UMAP - Dimension Reduction
    # Reduce to 5 dimensions for HDBSCAN (good balance)
    # n_neighbors=15, min_dist=0.1 are standard defaults
    reducer = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.1, metric='cosine', random_state=42)
    embedding_reduced = reducer.fit_transform(embeddings)
    
    # 3. HDBSCAN - Clustering
    # min_cluster_size depends on dataset size, keep it small for personal data
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=2, metric='euclidean', cluster_selection_method='eom')
    cluster_labels = clusterer.fit_predict(embedding_reduced)
    
    # 4. Result Processing
    # Label -1 is noise.
    unique_labels = set(cluster_labels)
    
    # Clear old clusters? Or update?
    # Simple approach: Delete old clusters for user, create new.
    # Note: Cascading delete on 'insights' will clean them up.
    # But 'prompts' have FK to cluster. Set null first?
    # The schema says: fk_cluster ... on delete set null. So safe to delete clusters.
    
    supabase.table("clusters").delete().eq("user_id", str(user_id)).execute()
    
    cluster_map = {} # label_id -> valid_uuid
    
    for label in unique_labels:
        if label == -1:
            continue
            
        # Calculate centroid (in original high-dim space or reduced? usually high-dim is better for representation)
        mask = (cluster_labels == label)
        cluster_points = embeddings[mask]
        centroid = np.mean(cluster_points, axis=0).tolist()
        
        # Create Cluster in DB
        # Placeholder info for label/description
        cluster_data = {
            "user_id": str(user_id),
            "label": f"Cluster {label}",
            "description": "Auto-generated cluster", 
            "centroid": centroid
        }
        res = supabase.table("clusters").insert(cluster_data).execute()
        if res.data:
            cluster_uuid = res.data[0]['id']
            cluster_map[label] = cluster_uuid
            
    # 5. Update Prompts with Cluster ID
    for idx, prompt_id in enumerate(ids):
        label = cluster_labels[idx]
        if label != -1:
            c_uuid = cluster_map.get(label)
            if c_uuid:
                supabase.table("prompts").update({"cluster_id": c_uuid}).eq("id", prompt_id).execute()
        else:
            # Clear cluster id if it was set (though we deleted clusters, so it's null already)
            pass
            
    return {"status": "success", "clusters_found": len(unique_labels) - (1 if -1 in unique_labels else 0)}
