import json

file_path = "/home/eric/temp/SeViLA-test/pdvc_dvc_results.json"

# Read the JSON file
with open(file_path, 'r') as file:
    data = json.load(file)


file_path = "/home/eric/temp/SeViLA-test/dvc_results_indent.json"
# Write the JSON file with indentation
with open(file_path, 'w') as file:
    json.dump(data, file, indent=4)