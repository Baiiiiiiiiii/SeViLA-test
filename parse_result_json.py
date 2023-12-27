import json

# Load the original JSON data
input_file_path = '/home/eric/temp/SeViLA-test/result/best_blip_no_frameidx_test/result/test_epochbest.json'
# Save the result to a new JSON file
output_file_path = '/home/eric/temp/SeViLA-test/result/best_blip_no_frameidx_test/result/test_clean.json'  # Specify the desired output file path


with open(input_file_path, 'r') as file:
    original_data = json.load(file)

# Initialize a dictionary to store the organized data
organized_data = {}

# Iterate over the original data and organize it into the new format
for item in original_data:
    # Extract information from the original data
    qid = item["qid"]
    category, _ = qid.split('_')[:2]
    prediction = item["prediction"]

    # Create a new dictionary with the organized format
    new_item = {"question_id": qid, "answer": prediction}

    # Append the new item to the corresponding category in the organized data dictionary
    if category not in organized_data:
        organized_data[category] = []
    organized_data[category].append(new_item)

# Convert the organized data dictionary to JSON format
output_data = json.dumps(organized_data)

with open(output_file_path, 'w') as file:
    file.write(output_data)

print(f"Conversion completed. Result saved to {output_file_path}")
