import pickle
import json
file_path = '/root/nas/my_work/document/qid2a.pkl'
# json_path = '/root/nas/my_work/document/id2ans.json'
with open(file_path, 'rb') as file:
    data = pickle.load(file)
    
# with open(json_path, 'w') as json_file:
#     json.dump(data, json_file, indent = 4)
print(len(data))
# for key, value in data.items():
#     if value == "positively charged":
#         print(key)
