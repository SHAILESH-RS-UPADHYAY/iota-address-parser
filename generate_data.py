import json
import random

# Mock structured dataset (In a real scenario, this comes from Kaggle/OpenAddresses CSV)
# Columns: Street, City, State, Zip
mock_structured_data = [
    {"Street": "123 Main St", "City": "New York", "State": "NY", "Zip": "10001"},
    {"Street": "456 Elm Street", "City": "Los Angeles", "State": "CA", "Zip": "90001"},
    {"Street": "789 Pine Ave", "City": "Seattle", "State": "WA", "Zip": "98101"},
    {"Street": "321 Maple Dr", "City": "Austin", "State": "TX", "Zip": "73301"},
    {"Street": "555 Cedar Blvd", "City": "Chicago", "State": "IL", "Zip": "60601"},
    {"Street": "999 Oak Lane", "City": "Miami", "State": "FL", "Zip": "33101"},
    {"Street": "777 Birch Court", "City": "Denver", "State": "CO", "Zip": "80201"},
    {"Street": "888 Spruce Way", "City": "Boston", "State": "MA", "Zip": "02101"},
    {"Street": "222 Walnut St", "City": "San Francisco", "State": "CA", "Zip": "94101"},
    {"Street": "444 Cherry Ave", "City": "Portland", "State": "OR", "Zip": "97201"}
]

# We need to format the data into SpaCy's expected format:
# ("Raw String", {"entities": [(start_char, end_char, "LABEL")]})

def generate_training_data(data_rows, num_samples=50):
    training_data = []
    
    for _ in range(num_samples):
        # Pick a random row
        row = random.choice(data_rows)
        
        street = row["Street"]
        city = row["City"]
        state = row["State"]
        zip_code = row["Zip"]
        
        # Randomly choose a formatting style to make the model robust
        style = random.choice([1, 2, 3])
        
        if style == 3:
            street = street.lower()
            city = city.lower()
            state = state.lower()
            zip_code = zip_code.lower()
            
        entities = []
        raw_text = ""
        
        if style == 1 or style == 3:
            # Format: "Street, City, State Zip"
            entities.append((len(raw_text), len(raw_text) + len(street), "STREET"))
            raw_text += street + ", "
            
            entities.append((len(raw_text), len(raw_text) + len(city), "CITY"))
            raw_text += city + ", "
            
            entities.append((len(raw_text), len(raw_text) + len(state), "STATE"))
            raw_text += state + " "
            
            entities.append((len(raw_text), len(raw_text) + len(zip_code), "ZIP"))
            raw_text += zip_code
            
        elif style == 2:
            # Format: "Street City State Zip"
            entities.append((len(raw_text), len(raw_text) + len(street), "STREET"))
            raw_text += street + " "
            
            entities.append((len(raw_text), len(raw_text) + len(city), "CITY"))
            raw_text += city + " "
            
            entities.append((len(raw_text), len(raw_text) + len(state), "STATE"))
            raw_text += state + " "
            
            entities.append((len(raw_text), len(raw_text) + len(zip_code), "ZIP"))
            raw_text += zip_code
            
        training_data.append((raw_text, {"entities": entities}))
            
    return training_data

if __name__ == "__main__":
    print("Generating synthetic training data...")
    data = generate_training_data(mock_structured_data, num_samples=100)
    
    # Save to JSON for the training script to consume
    with open("training_data.json", "w") as f:
        json.dump(data, f, indent=4)
        
    print(f"Successfully generated {len(data)} training samples and saved to training_data.json")
