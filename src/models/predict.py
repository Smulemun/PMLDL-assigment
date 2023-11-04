import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data.preprocess import put_mask, put_mask_with_classifier, get_toxicity
from data.postprocess import postprocess

# prefix to use for 
PREFIX = 'Detoxify text: '

def detoxificate_text(texts, toxic_words, tokenizer, model, device='cpu'):
    '''Function to detoxify a given text'''
    # putting mask in sentences 
    masked_text = [put_mask(x, toxic_words) for x in texts]
    # tokenize texts
    test_input = tokenizer(masked_text, padding='max_length', max_length=128, truncation=True, return_tensors='pt')
    test_input.to(device)
    model.to(device)
    input_ids = test_input.input_ids
    with torch.no_grad():
        outputs = model(**test_input)
    test_input.to('cpu')
    for i in range(len(texts)):
        # find masks in the text and replace them with non-toxic words
        mask_idxs = torch.where(test_input['input_ids'][i] == tokenizer.mask_token_id)
        mask_token_logits = outputs.logits[i, mask_idxs[0]]
        top_tokens = torch.topk(mask_token_logits, 100, dim=1).indices.tolist()
        for j in range(len(top_tokens)):
            for token in top_tokens[j]:
                if tokenizer.decode([token]) not in toxic_words:
                    input_ids[i][mask_idxs[0][j]] = token
                    break

    # decode the tokenized texts
    non_toxic_text = postprocess(tokenizer.batch_decode(input_ids, skip_special_tokens=True))

    return non_toxic_text

def detoxificate_text_with_classifier(texts, tokenizer, masked_model, classifier, device='cpu'):
    '''Function to detoxify a given text using a classifier'''
    # putting mask in sentences 
    masked_text = [put_mask_with_classifier(x, tokenizer, classifier) for x in texts]
    # tokenize texts
    test_input = tokenizer(masked_text, padding='max_length', max_length=128, truncation=True, return_tensors='pt')
    test_input.to(device)
    masked_model.to(device)
    input_ids = test_input.input_ids
    with torch.no_grad():
        outputs = masked_model(**test_input)
    test_input.to('cpu')
    for i in range(len(texts)):
        # find masks in the text and replace them with non-toxic words
        mask_idxs = torch.where(test_input['input_ids'][i] == tokenizer.mask_token_id)
        mask_token_logits = outputs.logits[i, mask_idxs[0]]
        top_tokens = torch.topk(mask_token_logits, 100, dim=1).indices.tolist()
        for j in range(len(top_tokens)):
            for token in top_tokens[j]:
                if get_toxicity(tokenizer.decode([token]), tokenizer, classifier) < 0.8:
                    input_ids[i][mask_idxs[0][j]] = token
                    break

    # decode the tokenized texts
    non_toxic_text = postprocess(tokenizer.batch_decode(input_ids, skip_special_tokens=True))

    return non_toxic_text

def detoxificate_style_transfer(texts, model, tokenizer, device='cpu'):
    '''Function to detoxify a given text using style transfer'''
    # adding prefix to the sentences
    texts = [PREFIX + x for x in texts]
    # tokenize texts
    test_input = tokenizer(texts, padding='max_length', max_length=128, truncation=True, return_tensors='pt')
    test_input.to(device)
    model.to(device)
    input_ids = test_input.input_ids
    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, max_length=128)
    # decode the tokenized texts
    non_toxic_text = postprocess(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    return non_toxic_text