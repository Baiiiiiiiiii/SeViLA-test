import pandas as pd

# Read the CSV file into a DataFrame
file_path = '/home/eric/temp/SeViLA-test/Video_Keyframe_IDs.csv'  # Replace with the actual file path
df = pd.read_csv(file_path)

# Create a dictionary from 'question_id' and 'video_id' columns
id_dict = dict(zip(df['question_id'], df['video_id']))

# Print or use the dictionary as needed
# print(id_dict)


new_dict={}
import torch
pt = torch.load('/home/eric/temp/SeViLA-test/pdvc_video_cap_dict_noTime.pt')

for k, v in id_dict.items():
    new_dict[k]=pt[v]
    
print(new_dict)


output_file_path = 'pdvc_qid2cap_dict_noTime.pt'
# Save the dictionary as a .pt file
torch.save(new_dict, output_file_path)