from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import numpy as np
import torch

semantic_similarity_model = SentenceTransformer('distilbert-base-nli-mean-tokens')

style_accuracy_model_name = 'SkolkovoInstitute/roberta_toxicity_classifier'
style_accuracy_tokenizer = RobertaTokenizer.from_pretrained(style_accuracy_model_name)
style_accuracy_model = RobertaForSequenceClassification.from_pretrained(style_accuracy_model_name)

fluency_model_name = "cointegrated/roberta-large-cola-krishna2020"
fluency_tokenizer = RobertaTokenizer.from_pretrained(fluency_model_name)
fluency_model = RobertaForSequenceClassification.from_pretrained(fluency_model_name)

def semantic_similarity(pred, target):
    pred_embedding = semantic_similarity_model.encode(pred)
    target_embedding = semantic_similarity_model.encode(target)
    cos_sim_score = cos_sim(pred_embedding, target_embedding)
    return np.mean([cos_sim_score[i][i] for i in range(len(pred))])

def style_accuracy(pred):

    # 1 - non-toxic
    # 0 - toxic

    with torch.no_grad():
        encoded = style_accuracy_tokenizer(pred, return_tensors='pt', padding=True)
        logits = style_accuracy_model(**encoded).logits
        result = torch.softmax(logits, dim=1)[:, 0]
    return np.mean(result.numpy())

def fluency(pred):

    # 1 - fluent
    # 0 - non-fluent
    
    with torch.no_grad():
        encoded = fluency_tokenizer(pred, return_tensors='pt', padding=True)
        logits = fluency_model(**encoded).logits
        result = torch.softmax(logits, dim=1)[:, 0]
    return np.mean(result.numpy())

def j_metric(similarity, accuracy, fluency):
    return similarity * accuracy * fluency
