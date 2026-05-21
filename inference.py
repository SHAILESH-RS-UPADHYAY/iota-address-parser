import spacy
import json

def parse_address(address_string, model_dir="address_ner_model"):
    """
    Loads the custom NER model and extracts address components.
    """
    try:
        nlp = spacy.load(model_dir)
    except OSError:
        print(f"Error: Model not found at '{model_dir}'. Please run train_model.py first.")
        return None

    # Process the string
    doc = nlp(address_string)
    
    # Extract entities into a dictionary
    parsed_address = {
        "Street": None,
        "City": None,
        "State": None,
        "Zip": None
    }
    
    for ent in doc.ents:
        # Standardize capitalization based on label if needed
        label = ent.label_.capitalize()
        if label in parsed_address:
            parsed_address[label] = ent.text
            
    return parsed_address

if __name__ == "__main__":
    print("========================================")
    print(" IOTA ANALYTICS - ADDRESS PARSER ENGINE")
    print("========================================")
    print("Type 'exit' or 'quit' to close the program.\n")
    
    while True:
        try:
            # Let the interviewer type whatever address they want!
            user_input = input("Enter an address to parse -> ")
            
            if user_input.strip().lower() in ['exit', 'quit']:
                print("Exiting...")
                break
                
            if not user_input.strip():
                continue
                
            print("\nProcessing...")
            result = parse_address(user_input)
            
            if result:
                print("\n[Parsed Output]:")
                print(json.dumps(result, indent=2))
            print("-" * 40 + "\n")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
