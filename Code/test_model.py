# import requests
# import torch
# from PIL import Image
# from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation


# # load Mask2Former fine-tuned on COCO panoptic segmentation
# processor = AutoImageProcessor.from_pretrained("./pre_model/mask2former-swin-base-ade-semantic/")
# model = Mask2FormerForUniversalSegmentation.from_pretrained("./pre_model/mask2former-swin-base-ade-semantic/")

# # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# # image = Image.open(requests.get(url, stream=True).raw)
# image_path = "./000000039769.jpg"
# image = Image.open(image_path).convert('RGB')
# inputs = processor(images=image, return_tensors="pt")

# with torch.no_grad():
#     outputs = model(**inputs)

# # model predicts class_queries_logits of shape `(batch_size, num_queries)`
# # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
# class_queries_logits = outputs.class_queries_logits
# masks_queries_logits = outputs.masks_queries_logits
# features = outputs.last_hidden_state

# # you can pass them to processor for postprocessing
# result = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
# # we refer to the demo notebooks for visualization (see "Resources" section in the Mask2Former docs)
# predicted_panoptic_map = result["segmentation"]


# from transformers import AutoImageProcessor, Swinv2Model
# import torch
# from PIL import Image

# image_path = "./000000039769.jpg"
# image = Image.open(image_path).convert('RGB')

# image_processor = AutoImageProcessor.from_pretrained("./pre_model/swinv2-base-patch4-window12-192-22k/")
# model = Swinv2Model.from_pretrained("./pre_model/swinv2-base-patch4-window12-192-22k/")

# inputs = image_processor(image, return_tensors="pt")

# with torch.no_grad():
#     outputs = model(**inputs)

# last_hidden_states = outputs.last_hidden_state
# list(last_hidden_states.shape)

from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline, TrainingArguments, Trainer


# model_name = "deepset/roberta-base-squad2"
model_name = "../huggingface/model/roberta-base-squad2"
max_length = 384 # The maximum length of a feature (question and context) 问题和文本的长度
doc_stride = 128 # The authorized overlap between two part of the context when splitting it is needed.

# a) Get predictions
# nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
QA_input = {
    'question': 'Why is model conversion important?',
    'context': 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
}
# output = nlp(QA_input)
# print(output)

# b) Load model & tokenizer
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenized_example = tokenizer(QA_input['question'], QA_input['context'], 
                            max_length=max_length,
                            truncation="only_second",
                            return_overflowing_tokens=True,
                            stride=doc_stride
                            )
print(tokenized_example)

output = model(tokenized_example)
print(output)


