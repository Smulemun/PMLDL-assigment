import torch

def put_mask(text, words):
    text = text.lower()
    # replacing toxic words with [MASK]
    for word in words:
        text = text.replace(word, '[MASK]')
    return text

def get_toxicity(word, tokenizer, model, device='cpu'):
    # using classifier to get the toxicity of a given word
    try:
        encoded = tokenizer(word, add_special_tokens=False, max_length=1, truncation=True).input_ids[0]
    except Exception as e:
        return 0
    encoded = torch.tensor(encoded).to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        toxic_chance = model(encoded)
    return toxic_chance.item()

def put_mask_with_classifier(text, tokenizer, model, device='cpu'):
    text = text.lower().split()
    # replacing words that were classified as toxic with [MASK]
    for i in range(len(text)):
        toxic_chance = get_toxicity(text[i], tokenizer, model, device)
        if toxic_chance > 0.8:
            text[i] = "[MASK]"
    return ' '.join(text)