import os
import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return accuracy_score(labels_flat, pred_flat)

def get_data(file_path):
    if not os.path.exists(file_path):
        print('file not exisits.')
        exit(0)
    print('loading ', file_path)
    text = []
    labels = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            words, label = line.split('\t')
            text.append(words)
            labels.append(int(label))
            #if float(label) > 0:
            #    labels.append(1)
            #if float(label) <= 0:
            #    labels.append(0)
    return text, labels

def get_deception_data():
    fout = open('./deception/deception.txt', 'w')
    Truthful = os.listdir('deception/Transcription/Truthful/') 
    Truthful = sorted(Truthful)
    
    Deceptive = os.listdir('deception/Transcription/Deceptive/') 
    Deceptive = sorted(Deceptive)

    for file_name in Truthful:
        path = 'deception/Transcription/Truthful/' + file_name
        with open(path, 'r') as f:
            text = f.readline()
        fout.write(text + '\t1\n')
       
    for file_name in Deceptive:
        path = 'deception/Transcription/Deceptive/' + file_name
        with open(path, 'r') as f:
            text = f.readline()
        fout.write(text + '\t0\n')
 
    fout.close()

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



if __name__ == '__main__':
    get_deception_data()
