import torch

# Define the path to your checkpoint file
checkpoint_path = '/home/eric/temp/SeViLA/result/star_ft/checkpoint_best.pth'
# checkpoint_path = "/home/eric/temp/SeViLA/sevila_checkpoints/sevila_pretrained.pth"

# Load the checkpoint
checkpoint = torch.load(checkpoint_path)

# Access the model from the checkpoint
model_state_dict = checkpoint['model']

# Access other information saved in the checkpoint, such as optimizer state, epoch, etc.
# optimizer_state = checkpoint['optimizer_state']
# epoch = checkpoint['epoch']

# Print the model architecture
# Print parameter names

    
# Load weights from the second model
model1 = torch.load("/home/eric/temp/SeViLA/sevila_checkpoints/sevila_pretrained.pth")

# Load weights from the first model
# model2 = torch.load('/home/eric/temp/SeViLA/result/star_ft/checkpoint_best.pth')


model2 = torch.load('/home/eric/temp/SeViLA/best.pth')


model_state_dict={"model":model2}

for param_name in model_state_dict.keys():
    print(param_name)

torch.save(model_state_dict, "best_model.pth")

# # for param_name in model1.keys():
# #     print(param_name)
# # Combine weights (e.g., averaging, summation, or other custom logic)
# combined_weights = {}
# for name, param  in model1['model'].items():
#     model2['model'][name] = param

model3 = torch.load('best_model.pth')


for param_name in model3.keys():
    print(param_name)

# print(model2.keys())
    
# # model2["model"] = combined_weights
# torch.save(model2["model"], "best.pth")

# for param_name in model2.keys():
#     print(param_name)