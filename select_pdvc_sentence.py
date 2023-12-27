import json

# Your JSON data
file_path = "/home/eric/temp/SeViLA-test/pdvc_dvc_results.json"



# Load JSON data
with open(file_path, 'r') as file:
    data = json.load(file)

# Extract information for each video
video_info_dict = {}

for video_id, entries in data["results"].items():
    # Sort entries by sentence_score and take the top 4
    top_entries = sorted(entries, key=lambda x: x["sentence_score"], reverse=True)
    
    # Create a set to track unique sentences for each video
    unique_sentences = set()
    
    # Create a list to store up to 4 unique sentences for the video
    added_sentences = 0

    # Create a string for the video
    video_string = "Caption:\n"
    for entry in top_entries:
        start_time, end_time = entry["timestamp"]
        sentence = entry["sentence"]
        
        # Check if the sentence is not already added
        if sentence not in unique_sentences and added_sentences < 8:
            video_string += f"Time {start_time:.1f}-{end_time:.1f} sec : {sentence}\n"
            # video_string += f"{sentence}\n"
            unique_sentences.add(sentence)
            added_sentences += 1
            

    # Store the string in the dictionary
    video_info_dict[video_id] = video_string

# Print or use the dictionary as needed
print(video_info_dict["XU2BY"])    
    

import torch
# # Specify the file path for saving the .pt file
output_file_path = 'pdvc_video_cap_dict_Time_8_sentence.pt'
# Save the dictionary as a .pt file
torch.save(video_info_dict, output_file_path)
    
    