import torch

def compute_salience_scores(feature_weights, biomarker_names=None):
    """
    Normalizes and ranks the attention parameter weights to output
    a human-readable clinical feature importance profile.
    """
    raw_weights = feature_weights.detach().cpu().numpy().flatten()
    # Apply soft absolute scaling to normalize metrics between 0 and 1
    normalized_scores = np.abs(raw_weights) / np.sum(np.abs(raw_weights))
    
    if biomarker_names is None:
        biomarker_names = [f"Biomarker_{i+1}" for i in range(len(normalized_scores))]
        
    ranking = sorted(zip(biomarker_names, normalized_scores), key=lambda x: x[1], reverse=True)
    return ranking
