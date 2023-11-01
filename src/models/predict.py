import torch
from ..data.preprocess import put_mask, put_mask_with_classifier, get_toxicity
from ..data.postprocess import postprocess

def detoxificate_text(text, toxic_words, tokenizer, model):
    test_input = tokenizer(put_mask(text, toxic_words), padding='max_length', max_length=128, truncation=True, return_tensors='pt')
    input_ids = test_input.input_ids
    with torch.no_grad():
        output = model(**test_input)
    mask_idxs = torch.where(test_input['input_ids'][0] == tokenizer.mask_token_id)
    mask_token_logits = output.logits[0, mask_idxs[0]]
    top_tokens = torch.topk(mask_token_logits, 100, dim=1).indices.tolist()
    for i in range(len(top_tokens)):
        for token in top_tokens[i]:
            if tokenizer.decode([token]) not in toxic_words:
                input_ids[0][mask_idxs[0][i]] = token
                break

    non_toxic_text = postprocess([tokenizer.decode(input_ids[0], skip_special_tokens=True)])[0]

    return non_toxic_text

def detoxificate_text_with_classifier(text, tokenizer, masked_model, classifier):
    test_input = tokenizer(put_mask_with_classifier(text, tokenizer, classifier), padding='max_length', max_length=128, truncation=True, return_tensors='pt')
    input_ids = test_input.input_ids
    with torch.no_grad():
        output = masked_model(**test_input)
    mask_idxs = torch.where(test_input['input_ids'][0] == tokenizer.mask_token_id)
    mask_token_logits = output.logits[0, mask_idxs[0]]
    top_tokens = torch.topk(mask_token_logits, 100, dim=1).indices.tolist()
    for i in range(len(top_tokens)):
        for token in top_tokens[i]:
            if get_toxicity(tokenizer.decode([token]), tokenizer, classifier) < 0.8:
                input_ids[0][mask_idxs[0][i]] = token
                break

    non_toxic_text = postprocess([tokenizer.decode(input_ids[0], skip_special_tokens=True)])[0]

    return non_toxic_text