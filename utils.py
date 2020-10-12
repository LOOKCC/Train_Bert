import os
import torch

def get_data(file_path):
    if not os.path.exists(file_path):
        print('file not exisits.')
        exit(0)
    print('loading ', file_path)
    text = []
    labels = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            words, label = line.split(',')
            text.append(words)
            labels.append(float(label))
    return text, labels


def encode_fn(text_list, tokenizer):
    all_input_ids = []    
    for text in text_list:
        input_ids = tokenizer.encode(
                        text,                      
                        add_special_tokens = True,   
                        padding='max_length',      
                        return_tensors = 'pt'       
                   )
        all_input_ids.append(input_ids)    
    all_input_ids = torch.cat(all_input_ids, dim=0)
    return all_input_ids