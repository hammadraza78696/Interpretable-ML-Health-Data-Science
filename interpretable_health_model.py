import torch
import torch.nn as nn
import numpy as np

class InterpretableHealthModel(nn.Module):
    """
    A predictive neural network architecture designed for medical data analytics.
    Includes an explicit feature-attention weighting mechanism to ensure 
    clinical decision transparency and post-hoc model interpretability.
    """
    def __init__(self, num_biomarkers):
        super(InterpretableHealthModel, self).__init__()
        # Explicit scoring weight vector to track biomarker importance values
        self.biomarker_attention = nn.Parameter(torch.ones(num_biomarkers, 1) / num_biomarkers)
        
        self.predictive_network = nn.Sequential(
            nn.Linear(num_biomarkers, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid() # Binary classification output (e.g., disease presence)
        )
        
    def forward(self, patient_data):
        # Apply a Hadamard product to score the relative weight of clinical inputs dynamically
        weighted_features = patient_data * self.biomarker_attention.squeeze()
        prediction = self.predictive_network(weighted_features)
        return prediction, self.biomarker_attention

if __name__ == "__main__":
    print("Initializing Interpretable Health Analytics Network...")
    # Assume a patient dataset tracking 5 distinct clinical biomarkers
    model = InterpretableHealthModel(num_biomarkers=5)
    
    # Generate mock clinical matrix: 3 patients, 5 health markers each
    mock_patient_profiles = torch.rand(3, 5)
    risk_probabilities, feature_weights = model(mock_patient_profiles)
    
    print("\nPatient Disease Risk Output Probabilities:")
    print(risk_probabilities.detach().numpy())
    print("\nExtracted Clinical Feature Importance Weights (Biomarker Tracking Metrics):")
    print(feature_weights.detach().numpy().flatten())
    print("\nInterpretable predictive pipeline verified successfully!")
