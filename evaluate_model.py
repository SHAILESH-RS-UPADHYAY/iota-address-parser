"""
Evaluation Script for the Address Parser NER Model.

Performs a proper train/test split and computes:
- Per-entity Precision, Recall, F1-Score
- Overall (micro-averaged) Precision, Recall, F1-Score

Usage:
    python evaluate_model.py
"""

import json
import spacy
from spacy.training.example import Example
from spacy.scorer import Scorer
import random

def load_data(filepath):
    """Load data from JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    return [(item[0], item[1]) for item in data]

def evaluate_model(model_dir="address_ner_model", data_path="training_data.json", test_ratio=0.2):
    """
    Evaluate the trained NER model using a holdout test set.
    
    Args:
        model_dir: Path to the trained SpaCy model directory.
        data_path: Path to the full dataset JSON.
        test_ratio: Fraction of data reserved for testing (default 20%).
    """
    print("=" * 50)
    print("  ADDRESS PARSER - MODEL EVALUATION REPORT")
    print("=" * 50)
    
    # Load the trained model
    try:
        nlp = spacy.load(model_dir)
    except OSError:
        print(f"Error: Model not found at '{model_dir}'. Run train_model.py first.")
        return
        
    # Load full dataset
    all_data = load_data(data_path)
    random.seed(42)  # Fixed seed for reproducibility
    random.shuffle(all_data)
    
    # Train/Test split
    split_index = int(len(all_data) * (1 - test_ratio))
    test_data = all_data[split_index:]
    
    print(f"\n  Total samples:    {len(all_data)}")
    print(f"  Training samples: {split_index}")
    print(f"  Test samples:     {len(test_data)}")
    print(f"  Test ratio:       {test_ratio * 100:.0f}%")
    print("-" * 50)
    
    # Convert test data to SpaCy Example objects
    examples = []
    for text, annotations in test_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        # Run the model's prediction on the input
        example.predicted = nlp(text)
        examples.append(example)
    
    # Score the predictions
    scorer = Scorer()
    scores = scorer.score(examples)
    
    # Extract NER-specific metrics
    ents_p = scores.get("ents_p", 0) * 100
    ents_r = scores.get("ents_r", 0) * 100
    ents_f = scores.get("ents_f", 0) * 100
    
    # Per-entity metrics
    ents_per_type = scores.get("ents_per_type", {})
    
    print(f"\n  {'OVERALL METRICS':^46}")
    print(f"  {'-' * 46}")
    print(f"  {'Metric':<20} {'Score':>10}")
    print(f"  {'-' * 46}")
    print(f"  {'Precision':<20} {ents_p:>9.2f}%")
    print(f"  {'Recall':<20} {ents_r:>9.2f}%")
    print(f"  {'F1-Score':<20} {ents_f:>9.2f}%")
    print(f"  {'-' * 46}")
    
    if ents_per_type:
        print(f"\n  {'PER-ENTITY BREAKDOWN':^46}")
        print(f"  {'-' * 46}")
        print(f"  {'Entity':<10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print(f"  {'-' * 46}")
        
        for entity_type in ["STREET", "CITY", "STATE", "ZIP"]:
            if entity_type in ents_per_type:
                metrics = ents_per_type[entity_type]
                p = metrics.get("p", 0) * 100
                r = metrics.get("r", 0) * 100
                f = metrics.get("f", 0) * 100
                print(f"  {entity_type:<10} {p:>9.2f}% {r:>9.2f}% {f:>9.2f}%")
        
        print(f"  {'-' * 46}")
    
    # Show some sample predictions vs ground truth
    print(f"\n  {'SAMPLE PREDICTIONS (first 5 test samples)':^46}")
    print(f"  {'-' * 46}")
    
    for i, (text, annotations) in enumerate(test_data[:5]):
        doc = nlp(text)
        predicted = {ent.label_: ent.text for ent in doc.ents}
        
        ground_truth = {}
        for start, end, label in annotations["entities"]:
            ground_truth[label] = text[start:end]
        
        match = "[OK]" if predicted == ground_truth else "[WARN]"
        
        print(f"\n  [{i+1}] {match} Input: \"{text}\"")
        print(f"      Expected: {ground_truth}")
        print(f"      Got:      {predicted}")
    
    print(f"\n{'=' * 50}")
    print(f"  Evaluation complete.")
    print(f"{'=' * 50}")

if __name__ == "__main__":
    evaluate_model()
