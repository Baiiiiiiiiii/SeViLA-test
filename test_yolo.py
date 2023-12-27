# from ultralytics import YOLO

# # Load a pretrained YOLOv8n model
# model = YOLO('yolov8n.pt')
# print(model.model.model[21])
# img = '/home/eric/temp/InternVideo/Data/InternVid/viclip/TJZ0P.mp4/000181.png'
# # Run inference on 'bus.jpg' with arguments
# results = model.predict(source=img, embed=[10, 15, 20])
# for r in results:
#     for i, x in enumerate(r.embeddings):
#         print(f'{r.path}: embedding {i} shape is {x.shape}')


# # print(result[0].size())
# # print(result[1])
# # print(len(result))
# # tensors = model.embed(embed=[1, 15, 18])
# # print(len(tensors))
# embedding = model.embed(img)
# print(embedding[0].shape)



from ultralytics import YOLO
from PIL import Image
import torchvision
# model = YOLO()
# print(model.embed(source="/home/eric/miniconda3/envs/sev/lib/python3.8/site-packages/ultralytics/assets/bus.jpg", embed=[15, 18, 21]).shape)  # shape(2,448)



test_img = '/home/eric/miniconda3/envs/sev/lib/python3.8/site-packages/ultralytics/assets/bus.jpg'
pil_img = Image.open(test_img)
tensor_img = torchvision.transforms.functional.pil_to_tensor(pil_img)[None, :] / 255.0

yolo = YOLO('yolov8n-seg.pt')
yolo.model.model = yolo.model.model[:-1]  # drop only last layer (Segment(...))

res = yolo.model(tensor_img)
print(res)

res = yolo.model(tensor_img)
model = YOLO()
result = model.embed(source='/home/eric/miniconda3/envs/sev/lib/python3.8/site-packages/ultralytics/assets/bus.jpg')
print(result[0].shape)
# for i, r in enumerate(results):
    # for i, x in enumerate(r):
    # print(f'{i}: embedding {i} shape is {r.shape}')