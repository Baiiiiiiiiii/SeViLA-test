import torch

checkpoint_path = "/home/eric/temp/SeViLA-test/qid_vid_0.pt"

checkpoint = torch.load(checkpoint_path)
# print(checkpoint["Interaction_T1_1"].size())
# print(checkpoint["Interaction_T1_1"].squeeze(0).mean(dim=0).size())

qids = checkpoint["qids"]
vid_feats = checkpoint["video_features"]
print(vid_feats[0].size())

assert len(qids)==len(vid_feats)
dict = {}
for idx in range(len(qids)):
    vid_feat = vid_feats[idx].unsqueeze(0)
    dict[qids[idx]]=vid_feat
    
# torch.save(dict,"/home/eric/temp/SeViLA-test/vis_feat_test.pt")
print(dict["Sequence_T2_2448"].size())
print(len(dict))




checkpoint_path2 = "/home/eric/temp/SeViLA-test/qid_vid_1.pt"

checkpoint = torch.load(checkpoint_path2)
# print(checkpoint["Interaction_T1_1"].size())
# print(checkpoint["Interaction_T1_1"].squeeze(0).mean(dim=0).size())

qids = checkpoint["qids"]
vid_feats = checkpoint["video_features"]
print(vid_feats[0].size())

assert len(qids)==len(vid_feats)
# dict = {}
for idx in range(len(qids)):
    vid_feat = vid_feats[idx].unsqueeze(0)
    dict[qids[idx]]=vid_feat

print(len(dict))
torch.save(dict,"/home/eric/temp/SeViLA-test/vis_feat_test_ego400.pt")
# print(dict["Sequence_T2_2448"].size())






import torch

def process_checkpoint(checkpoint_path, existing_dict=None):
    checkpoint = torch.load(checkpoint_path)
    qids = checkpoint["qids"]
    vid_feats = checkpoint["video_features"]
    
    if existing_dict is None:
        dict_result = {}
    else:
        dict_result = existing_dict

    assert len(qids) == len(vid_feats)

    for idx in range(len(qids)):
        vid_feat = vid_feats[idx].unsqueeze(0)
        dict_result[qids[idx]] = vid_feat

    return dict_result

# List of checkpoint paths
checkpoint_paths = [
    "/home/eric/temp/SeVIla-test/qid_vid_0.pt",
    "/home/eric/temp/SeVIla-test/qid_vid_1.pt",
    "/home/eric/temp/SeVIla-test/qid_vid_2.pt",
    "/home/eric/temp/SeVIla-test/qid_vid_3.pt",
    "/home/eric/temp/SeVIla-test/qid_vid_4.pt"
]

# Initialize an empty dictionary to store the results
result_dict = None

# Iterate over the checkpoint paths
for checkpoint_path in checkpoint_paths:
    result_dict = process_checkpoint(checkpoint_path, existing_dict=result_dict)

# Save the final result
torch.save(result_dict, "/home/eric/temp/SeVIla-test/vis_feat_test_ego400.pt")









    # print(value)
#     print(value.squeeze(0).flatten().size())
    # if value.dim()==3:
    # print(value.squeeze(0).flatten().size())
    # checkpoint[key] = value.squeeze(0).mean(dim=0)
# checkpoint_path = "/home/eric/temp/SeViLA-test/egovlpv2_val.pt"
# checkpoint_path = "/home/eric/temp/SeViLA-test/only_video.pt"
# video_feat = torch.load(checkpoint_path)
# video_feat = video_feat.detach().cpu().squeeze(0)
# print(video_feat.detach().cpu().size())
# dict = {"Interaction_T1_4":video_feat, "Interaction_T1_1":video_feat}
# dict = {"Interaction_T1_4":torch.rand(2048,768), "Interaction_T1_1":torch.rand(2048,768)}
# torch.save(dict,"/home/eric/temp/SeViLA-test/vis_feat_test.pt")
# print(dict)