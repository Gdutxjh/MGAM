import json
import os

def caption_generate():
    json_path = "PathVQA_data/decla"
    split = ["train.json", "val.json", "test.json"]
    split_dict = ["train_dict.json", "val_dict.json", "test_dict.json"]
    json_dir = "PathVQA_data/split_caption"

    for i in range(len(split)):
        json_data = json.load(open(os.path.join(json_path, split[i]), "r"))
        json_caption = []
        json_dict = {}
        for j in range(len(json_data)):
            # "img_id": "val_0000", 
            # "question": " What does candida organism have?", 
            # "pred": "the candida organism has [MASK] color.", 
            # "answer": "pseudohyphae and budding yeasts"
            ann = json_data[j]
            img_id = ann["img_id"]
            answer = ann["answer"]
            caption = ann["pred"]
            caption = caption.replace("[MASK]", answer)
            if answer == "no":
                continue

            json_caption.append({
                "img_id": img_id,
                "question": ann["question"],
                "pred": ann["pred"],
                "caption": caption,
                "answer": ann["answer"]
            })
            if img_id not in json_dict:
                json_dict[img_id] = caption
            else:
                json_dict[img_id] = json_dict[img_id] + " " + caption
        result_json = os.path.join(json_dir, split[i])
        json.dump(json_caption, open(result_json, "w"))
        print("save to %s" % result_json)
        result_dict = os.path.join(json_dir, split_dict[i])
        json_dict2list = []
        json_dict2list.append(json_dict)
        json.dump(json_dict2list, open(result_dict, "w"))
        print("save to %s" % result_dict)

def create_pathvqa_caption():
    json_path = "PathVQA_data/caption"
    split = ["train_dict.json", "val_dict.json", "test_dict.json"]
    json_dir = "PathVQA_data/caption/json"
    split_list = ["train_list.json", "val_list.json", "test_list.json"]

    for i in range(len(split)):
        json_data = json.load(open(os.path.join(json_path, split[i]), "r"))
        json_list = []
        for img_id, caption in json_data[0].items():
            json_list.append({
                "img_id": img_id,
                "caption": caption,
                "type": img_id.split("_")[0]
            })
        result_file = os.path.join(json_dir, split_list[i])
        json.dump(json_list, open(result_file, "w"))
        print("save to %s" % result_file)

def get_close_dataset():
    json_path = "PathVQA_data/decla/train.json"
    pathvqa_dataset = json.load(open(json_path, "r"))
    # {"img_id": "train_0001", 
    # "question": " What form a submucosal nodule composed of tumor cells embedded in dense fibrous tissue?", 
    # "pred": "the [MASK] form a submucosal nodule", 
    # "answer": "carcinoid tumors"}
    close_dataset = []
    for i in range(len(pathvqa_dataset)):
        answer = pathvqa_dataset[i]["answer"]
        if answer in ["yes", "no"]:
            close_dataset.append({
                "img_id": pathvqa_dataset[i]["img_id"],
                "question": pathvqa_dataset[i]["question"],
                "pred": pathvqa_dataset[i]["pred"],
                "answer": pathvqa_dataset[i]["answer"]
            })
    json_dir = "PathVQA_data/decla/train_close.json"
    json.dump(close_dataset, open(json_dir, "w"))
    print("save to %s " % json_dir)

if __name__ == '__main__':
    # caption_generate()
    # create_pathvqa_caption()
    get_close_dataset()