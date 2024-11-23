from transformers import AutoTokenizer,  AutoModelForSeq2SeqLM
from transformers import BartTokenizer, BartForConditionalGeneration
import json

def generate_D(tokenizer, model, question, answer):
    SEP = ". "

    prompt = f'{question}{SEP}{answer}'
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids
    output_ids = model.generate(input_ids)
    responses = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return responses

def main():
    tokenizer = AutoTokenizer.from_pretrained('domenicrosati/QA2D-t5-base')
    model = AutoModelForSeq2SeqLM.from_pretrained('domenicrosati/QA2D-t5-base')
    
    # data = json.load(open("slake/train_en_spread.json"))
    # data = json.load(open('../med-data/VQA_RAD/train_spread.json'))
    data = json.load(open('../med-data/PathVQA/split/declaration/train_declar.json'))
    ddate = []
    for i in range(len(data)):
        row = data[i]
        question = row["sent"]
        # answer = row["answer"]
        label = row["label"]
        for key, value in label.items():
            answer = key
        
        if str(answer).lower() == "no":
            continue
        
        declar = generate_D(tokenizer, model, question, answer)
        # if answer in declar:
        #     declar = declar.replace(answer, "[MASK]")
        #     row["declaration"] = declar
        #     ddate.append(row)
        row["declaration"] = declar[0]
        ddate.append(row)
        print("\r[%d/%d]" % (i, len(data)), end=" ")
        
    json.dump(ddate, open("predata/PathVQA/train_declar_wno.json", "w"), indent=2)

def bert_QA2D():
    tokenizer = BartTokenizer.from_pretrained("MarkS/bart-base-qa2d")
    model = BartForConditionalGeneration.from_pretrained("MarkS/bart-base-qa2d")

    input_text = "question: Are they normal? answer: No"
    input = tokenizer(input_text, return_tensors='pt')
    output = model.generate(input.input_ids)
    result = tokenizer.batch_decode(output, skip_special_tokens=True)
    print(result)



if __name__ == "__main__":
    # main()
    bert_QA2D()
