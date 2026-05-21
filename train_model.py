import json
import spacy
from spacy.training.example import Example
from spacy.util import minibatch, compounding
import random
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def load_data(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    # The JSON is stored as a list of lists: [raw_text, {"entities": [...]}]
    # We need to convert it to tuples for SpaCy
    return [(item[0], item[1]) for item in data]

def train_spacy_model(training_data, output_dir="address_ner_model", iterations=30):
    print("Starting model training...")
    
    # 1. Create a blank English model
    nlp = spacy.blank("en")
    
    # 2. Add the NER component to the pipeline
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")
        
    # 3. Add our custom labels to the NER component
    for _, annotations in training_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
            
    # 4. Disable other pipelines during training to focus on NER
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    
    # 5. Train the model
    with nlp.disable_pipes(*unaffected_pipes):
        # Initialize the model weights
        optimizer = nlp.begin_training()
        
        for itn in range(iterations):
            random.shuffle(training_data)
            losses = {}
            
            # Batch up the examples using spaCy's minibatch
            batches = minibatch(training_data, size=compounding(4.0, 32.0, 1.001))
            
            for batch in batches:
                examples = []
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    examples.append(example)
                
                # Update the model
                nlp.update(
                    examples,
                    drop=0.35,  # Dropout - make it harder to memorize data
                    sgd=optimizer,
                    losses=losses
                )
            
            print(f"Iteration {itn + 1}/{iterations} - Losses: {losses}")
            
    # 6. Save the model to disk
    nlp.to_disk(output_dir)
    print(f"Training completed! Model saved to '{output_dir}' directory.")

if __name__ == "__main__":
    data = load_data("training_data.json")
    train_spacy_model(data, iterations=20)
