import json
import random
import csv
import os

def load_addresses_from_csv(filepath):
    """Load structured addresses from a CSV file."""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                "Street": row["Street"].strip(),
                "City": row["City"].strip(),
                "State": row["State"].strip(),
                "Zip": row["Zip"].strip()
            })
    return data

def generate_training_data(data_rows, num_samples=2000):
    """
    Generate SpaCy NER training data using Programmatic Weak Supervision.
    
    Takes structured address rows (Street, City, State, Zip) and reverse-engineers
    them into messy, unstructured strings while tracking exact character offsets
    for each entity label.
    
    Supports 7 formatting variations to make the model robust against real-world input.
    """
    training_data = []
    
    for _ in range(num_samples):
        row = random.choice(data_rows)
        
        street = row["Street"]
        city = row["City"]
        state = row["State"]
        zip_code = row["Zip"]
        
        # Randomly choose a formatting style (7 variations)
        style = random.choice([1, 2, 3, 4, 5, 6, 7])
        
        # Style 3, 6, 7 use lowercase to train on case-insensitive input
        if style in [3, 6, 7]:
            street = street.lower()
            city = city.lower()
            state = state.lower()
            zip_code = zip_code.lower()
            
        entities = []
        raw_text = ""
        
        if style == 1:
            # "Street, City, State Zip"
            entities.append((len(raw_text), len(raw_text) + len(street), "STREET"))
            raw_text += street + ", "
            entities.append((len(raw_text), len(raw_text) + len(city), "CITY"))
            raw_text += city + ", "
            entities.append((len(raw_text), len(raw_text) + len(state), "STATE"))
            raw_text += state + " "
            entities.append((len(raw_text), len(raw_text) + len(zip_code), "ZIP"))
            raw_text += zip_code
            
        elif style == 2:
            # "Street City State Zip" (no commas)
            entities.append((len(raw_text), len(raw_text) + len(street), "STREET"))
            raw_text += street + " "
            entities.append((len(raw_text), len(raw_text) + len(city), "CITY"))
            raw_text += city + " "
            entities.append((len(raw_text), len(raw_text) + len(state), "STATE"))
            raw_text += state + " "
            entities.append((len(raw_text), len(raw_text) + len(zip_code), "ZIP"))
            raw_text += zip_code
            
        elif style == 3:
            # lowercase "street, city, state zip"
            entities.append((len(raw_text), len(raw_text) + len(street), "STREET"))
            raw_text += street + ", "
            entities.append((len(raw_text), len(raw_text) + len(city), "CITY"))
            raw_text += city + ", "
            entities.append((len(raw_text), len(raw_text) + len(state), "STATE"))
            raw_text += state + " "
            entities.append((len(raw_text), len(raw_text) + len(zip_code), "ZIP"))
            raw_text += zip_code
            
        elif style == 4:
            # "Street, City State Zip" (comma only after street)
            entities.append((len(raw_text), len(raw_text) + len(street), "STREET"))
            raw_text += street + ", "
            entities.append((len(raw_text), len(raw_text) + len(city), "CITY"))
            raw_text += city + " "
            entities.append((len(raw_text), len(raw_text) + len(state), "STATE"))
            raw_text += state + " "
            entities.append((len(raw_text), len(raw_text) + len(zip_code), "ZIP"))
            raw_text += zip_code
            
        elif style == 5:
            # "Street City, State Zip" (comma only after city)
            entities.append((len(raw_text), len(raw_text) + len(street), "STREET"))
            raw_text += street + " "
            entities.append((len(raw_text), len(raw_text) + len(city), "CITY"))
            raw_text += city + ", "
            entities.append((len(raw_text), len(raw_text) + len(state), "STATE"))
            raw_text += state + " "
            entities.append((len(raw_text), len(raw_text) + len(zip_code), "ZIP"))
            raw_text += zip_code
            
        elif style == 6:
            # lowercase no commas "street city state zip"
            entities.append((len(raw_text), len(raw_text) + len(street), "STREET"))
            raw_text += street + " "
            entities.append((len(raw_text), len(raw_text) + len(city), "CITY"))
            raw_text += city + " "
            entities.append((len(raw_text), len(raw_text) + len(state), "STATE"))
            raw_text += state + " "
            entities.append((len(raw_text), len(raw_text) + len(zip_code), "ZIP"))
            raw_text += zip_code
            
        elif style == 7:
            # lowercase no spaces after comma "street city,state zip"
            entities.append((len(raw_text), len(raw_text) + len(street), "STREET"))
            raw_text += street + " "
            entities.append((len(raw_text), len(raw_text) + len(city), "CITY"))
            raw_text += city + ","
            entities.append((len(raw_text), len(raw_text) + len(state), "STATE"))
            raw_text += state + " "
            entities.append((len(raw_text), len(raw_text) + len(zip_code), "ZIP"))
            raw_text += zip_code
            
        training_data.append((raw_text, {"entities": entities}))
            
    return training_data

if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(__file__), "us_addresses.csv")
    
    print(f"Loading structured addresses from {csv_path}...")
    structured_data = load_addresses_from_csv(csv_path)
    print(f"Loaded {len(structured_data)} unique real US addresses.")
    
    print("Generating synthetic training data with 6 format variations...")
    data = generate_training_data(structured_data, num_samples=2500)
    
    with open("training_data.json", "w") as f:
        json.dump(data, f, indent=2)
        
    print(f"Successfully generated {len(data)} training samples and saved to training_data.json")
    print(f"Format variations: comma-separated, space-separated, lowercase, mixed punctuation, missing spaces")
