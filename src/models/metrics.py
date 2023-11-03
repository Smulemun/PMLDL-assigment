from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import numpy as np
import torch

# loading models used for metric calculation
semantic_similarity_model = SentenceTransformer('distilbert-base-nli-mean-tokens')

style_accuracy_model_name = 'SkolkovoInstitute/roberta_toxicity_classifier'
style_accuracy_tokenizer = RobertaTokenizer.from_pretrained(style_accuracy_model_name)
style_accuracy_model = RobertaForSequenceClassification.from_pretrained(style_accuracy_model_name)

fluency_model_name = "cointegrated/roberta-large-cola-krishna2020"
fluency_tokenizer = RobertaTokenizer.from_pretrained(fluency_model_name)
fluency_model = RobertaForSequenceClassification.from_pretrained(fluency_model_name)

def semantic_similarity(pred, target):
    # Semantic similarity shows how similar two texts are based on their meaning (1 - same meaning, 0 - different meaning)
    pred_embedding = semantic_similarity_model.encode(pred)
    target_embedding = semantic_similarity_model.encode(target)
    cos_sim_score = cos_sim(pred_embedding, target_embedding)
    return np.mean([cos_sim_score[i][i] for i in range(len(pred))])

def style_accuracy(pred, device='cuda'):
    # Style accuracy shows how toxic the text is (1 - non-toxic, 0 - toxic)
    style_accuracy_model.to(device)
    with torch.no_grad():
        encoded = style_accuracy_tokenizer(pred, return_tensors='pt', padding=True).to(device)
        logits = style_accuracy_model(**encoded).logits
        result = torch.softmax(logits, dim=1)[:, 0]
    return np.mean(result.cpu().numpy())

def fluency(pred, device='cuda'):
    # Fluency shows how gramatically correct the given text is accordinng to the english grammar (1 - correct, 0 - incorrect)
    fluency_model.to(device)
    with torch.no_grad():
        encoded = fluency_tokenizer(pred, return_tensors='pt', padding=True).to(device)
        logits = fluency_model(**encoded).logits
        result = torch.softmax(logits, dim=1)[:, 0]
    return np.mean(result.cpu().numpy())

def j_metric(similarity, accuracy, fluency):
    # J metric is a combination of three metrics listed above
    return similarity * accuracy * fluency
