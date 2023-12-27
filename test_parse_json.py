import json

file_path = "/home/eric/temp/SeViLA-test/lavis/models/dvc_results.json"
output_file_path = "/home/eric/temp/SeViLA-test/lavis/models/extracted_data.txt"

with open(file_path, 'r') as file:
    data = json.load(file)

# Extract "timestamp" and "sentence" for each entry
extracted_data = []
for person, entries in data["results"].items():
    for entry in entries:
        start_time, end_time = entry["timestamp"]
        sentence = entry["sentence"]
        extracted_data.append(f"Time {start_time:.1f}-{end_time:.1f} : {sentence}")

# Create a string with the extracted data
result_string = " ".join(extracted_data)

# Print the result string
print(result_string)

# Save the result string to a file
with open(output_file_path, 'w') as output_file:
    output_file.write(result_string)