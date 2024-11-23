import json
import os

def create_woyn_data():
    json_root = "PathVQA_data"
    split = ["train.json", "val.json", "test.json"]
    output = ["train_wo.json", "val_wo.json", "test_wo.json"]

    for i in range(len(split)):
        # {'img_id': 'train_0001', 
        # 'question': ' What form a submucosal nodule composed of tumor cells embedded in dense fibrous tissue?', 
        # 'pred': 'the [MASK] form a submucosal nodule', 
        # 'answer': 'carcinoid tumors'}
        json_path = os.path.join(json_root, split[i])
        dataset = json.load(open(json_path, 'r'))
        json_data = []
        for j in range(len(dataset)):
            answer = dataset[j]["answer"] 
            if answer == "yes" or answer == "no":
                continue
            json_data.append({
                "img_id": dataset[j]["img_id"],
                "question": dataset[j]["question"],
                "answer": dataset[j]["answer"]
            })
        result_file = os.path.join(json_root, output[i])
        json.dump(json_data, open(result_file, "w"))
        print("save to %s" % result_file)

if __name__ == "__main__":
    create_woyn_data()
