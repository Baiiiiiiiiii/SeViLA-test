import torch

# checkpoint_path1 = "/home/eric/temp/SeViLA/result/star_ft_ViT_feat/checkpoint_best.pth"
# checkpoint_path2 = "/home/eric/temp/SeViLA/result/star_ft/checkpoint_best.pth"
# # checkpoint_path = "/home/eric/temp/SeViLA-test/star.pth"
# # checkpoint_path = "/home/eric/temp/SeViLA-test/only_video.pt"
# # video_feat = torch.load(checkpoint_path)["model"]
# pt1 = torch.load(checkpoint_path1)
# pt2 = torch.load(checkpoint_path2)
# # video_feat = video_feat.detach().cpu().squeeze(0)
# print(pt1["model"])
# print(pt2["model"])


checkpoint_path1 = "/home/eric/temp/SeViLA/result/star_ft_ViT_feat/checkpoint_best.pth"
checkpoint_path2 = "/home/eric/temp/SeViLA/result/star_ft/checkpoint_best.pth"
output_checkpoint_path = "/home/eric/temp/SeViLA/result/star_ft_ViT_feat/combined_checkpoint_best.pth"


# Load the checkpoints
checkpoint1 = torch.load(checkpoint_path1)
checkpoint2 = torch.load(checkpoint_path2)

# Create a new checkpoint dictionary
combined_checkpoint = {}

# Copy the state_dict of the first model
combined_checkpoint["model"] = checkpoint1["model"]

# Update the state_dict with the second model's parameters
for key, value in checkpoint2["model"].items():
    if key in combined_checkpoint["model"]:
        # If the key exists in both models, update the parameters
        combined_checkpoint["model"][key] = value
    else:
        # If the key only exists in the second model, add it to the combined model
        combined_checkpoint["model"][key] = value

# Save the combined checkpoint
torch.save(combined_checkpoint, output_checkpoint_path)
# for k, v in video_feat.items():
    # print(k)
# print(video_feat["visual_proj.weight"])
# print(video_feat["adapter_query.weight"].size())
# print(video_feat["temporal_emb.weight"].size())

# dict = {"Interaction_T1_4":video_feat, "Interaction_T1_1":video_feat}
# dict = {"Interaction_T1_4":torch.rand(1,768), "Interaction_T1_1":torch.rand(1,768)}
# torch.save(dict,"/home/eric/temp/SeViLA-test/vis_feat_test.pt")
# print(dict)












# checkpoint_path = "/home/eric/temp/SeViLA-test/vis_feat_test.pt"
# # Load the checkpoint

# checkpoint = torch.load(checkpoint_path)
# print(checkpoint["Interaction_T1_1"].size())
# print(checkpoint["Interaction_T1_1"].squeeze(0).mean(dim=0).size())

# for key, value in checkpoint.items():
# #     print(key)
# #     print(value.squeeze(0).flatten().size())
#     # if value.dim()==3:
#     print(value.squeeze(0).flatten().size())
#     checkpoint[key] = value.squeeze(0).mean(dim=0)
        

# torch.save(checkpoint,"/home/eric/temp/SeViLA-test/vis_feat_test.pt")
# print(checkpoint["Interaction_T1_4"].size())
# print(checkpoint)



# dict = {"Interaction_T1_4":torch.rand(6272,768), "Interaction_T1_1":torch.rand(6272,768)}
# print(dict["Interaction_T1_4"].size())
# torch.save(dict,"/home/eric/temp/SeViLA-test/vis_feat_test.pt")





























# video_feat= torch.load(checkpoint_path)
# print(video_feat["Interaction_T1_4"].size())
# tensor = video_feat["Interaction_T1_4"].squeeze(0)

# if tensor.dim() == 3:
#     # Squeeze the tensor to remove the first dimension
#     tensor = tensor.squeeze(0)
# print(tensor.size())

# # print(video_feat.size())
# # for key in checkpoint.keys():
#     # print(key)
# # id = checkpoint["ids"][0]
# # video_feat = checkpoint["video_features"][0]
# dict = {"Interaction_T1_4":video_feat, "Interaction_T1_1":video_feat}
# torch.save(dict,"/home/eric/temp/SeViLA-test/vis_feat_test.pt")
# print(dict)




# checkpoint_path = "/home/eric/temp/SeViLA-test/vis_feat_test.pt"
# # Load the checkpoint
# checkpoint = torch.load(checkpoint_path)
# print(checkpoint)





# import torch

# # Your input tensor
# tensor = torch.randn(1, 2, 3)

# # Check if the tensor has three dimensions
# if tensor.dim() == 3:
#     # Squeeze the tensor to remove the first dimension
#     tensor = tensor.squeeze(0)

# print(tensor.size())



# # Define the path to your checkpoint file
# checkpoint_path = '/home/eric/temp/SeViLA/result/star_ft/checkpoint_best.pth'
# # 

# # Load the checkpoint
# checkpoint = torch.load(checkpoint_path)

# # Access the model from the checkpoint
# model_state_dict = checkpoint['model']

# # Access other information saved in the checkpoint, such as optimizer state, epoch, etc.
# # optimizer_state = checkpoint['optimizer_state']
# # epoch = checkpoint['epoch']

# # Print the model architecture
# # Print parameter names

    
# # Load weights from the second model
# model1 = torch.load("/home/eric/temp/SeViLA/sevila_checkpoints/sevila_pretrained.pth")

# # Load weights from the first model
# # model2 = torch.load('/home/eric/temp/SeViLA/result/star_ft/checkpoint_best.pth')


# model2 = torch.load('/home/eric/temp/SeViLA/best.pth')


# model_state_dict={"model":model2}

# for param_name in model_state_dict.keys():
#     print(param_name)

# torch.save(model_state_dict, "best_model.pth")

# # # for param_name in model1.keys():
# # #     print(param_name)
# # # Combine weights (e.g., averaging, summation, or other custom logic)
# # combined_weights = {}
# # for name, param  in model1['model'].items():
# #     model2['model'][name] = param

# model3 = torch.load('best_model.pth')


# for param_name in model3.keys():
#     print(param_name)

# # print(model2.keys())
    
# # # model2["model"] = combined_weights
# # torch.save(model2["model"], "best.pth")

# # for param_name in model2.keys():
# #     print(param_name)