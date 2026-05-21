import json
import spacy
from spacy.training.example import Example
from spacy.util import minibatch, compounding
import random
import warnings

warnings.filterwarnings("ignore")

def load_data(filepath):
    """Load training data from JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    return [(item[0], item[1]) for item in data]

def train_spacy_model(training_data, output_dir="address_ner_model", iterations=30):
    """
    Train a custom SpaCy NER model for address parsing.
    
    Architecture: Blank English model + custom NER pipeline
    Optimizer: SGD with compounding batch sizes
    Regularization: 35% dropout
    """
    print(f"Training on {len(training_data)} samples for {iterations} iterations...")
    
    # 1. Create a blank English model
    nlp = spacy.blank("en")
    
    # 2. Add the NER component to the pipeline
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")
        
    # 3. Add custom labels
    for _, annotations in training_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
            
    # 4. Disable other pipelines during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    
    # 5. Training loop
    with nlp.disable_pipes(*unaffected_pipes):
        optimizer = nlp.begin_training()
        
        for itn in range(iterations):
            random.shuffle(training_data)
            losses = {}
            
            batches = minibatch(training_data, size=compounding(8.0, 64.0, 1.001))
            
            for batch in batches:
                examples = []
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    examples.append(example)
                
                nlp.update(
                    examples,
                    drop=0.25,
                    sgd=optimizer,
                    losses=losses
                )
            
            loss_val = losses.get("ner", 0)
            print(f"  Epoch {itn + 1:>2}/{iterations} | Loss: {loss_val:.6f}")
            
    # 6. Save the model
    nlp.to_disk(output_dir)
    print(f"\nModel saved to '{output_dir}/' directory.")

if __name__ == "__main__":
    data = load_data("training_data.json")
    train_spacy_model(data, iterations=50)
