# from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
# import torch
# from PIL import Image

# model = VisionEncoderDecoderModel.from_pretrained("./nlpconnect/vit-gpt2-image-captioning")
# feature_extractor = ViTImageProcessor.from_pretrained("./nlpconnect/vit-gpt2-image-captioning")
# tokenizer = AutoTokenizer.from_pretrained("./nlpconnect/vit-gpt2-image-captioning")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)



# max_length = 16
# num_beams = 4
# gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
# def predict_step(image_paths):
#   images = []
#   for image_path in image_paths:
#     i_image = Image.open(image_path)
#     if i_image.mode != "RGB":
#       i_image = i_image.convert(mode="RGB")

#     images.append(i_image)

#   pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
#   pixel_values = pixel_values.to(device)

#   output_ids = model.generate(pixel_values, **gen_kwargs)

#   preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
#   preds = [pred.strip() for pred in preds]
#   return preds


# predict_step(['1.jpg']) # ['a woman in a hospital bed with a woman in a hospital bed']


from PIL import Image
import matplotlib.pyplot as plt 
import numpy as np

plt.figure(figsize=(30,30))

img = Image.open("image_data/676.jpg")
plt.subplot(3, 1, 1)
plt.imshow(img)
plt.colorbar()
width,height = img.size

print(width,height)

img_black = img.convert("1")
plt.subplot(3, 1, 2)
plt.imshow(img_black)
plt.colorbar()

# plt.rcParams["image.cmap"] = "gray"
# plt.imshow(img_black)
# print(img_black)
img_01 = np.array(img)
a = img_01/255
img_01 = (img_01-img_01.min()/(img_01.max() - img_01.min()))
plt.subplot(3, 1, 3)
plt.imshow(img)
plt.colorbar()
plt.show()