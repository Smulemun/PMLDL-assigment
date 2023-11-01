from transformers import TrainingArguments, Trainer
from src.models.metrics import semantic_similarity, style_accuracy, fluency, j_metric
from ..data.postprocess import postprocess
import numpy as np
from tqdm import tqdm
import torch

def train(model_type, model, tokenizer, train_dataset, val_dataset, data_collator, batch_size, epochs, seed):
    
    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        decoded_preds = postprocess(tokenizer.batch_decode(preds, skip_special_tokens=True))
        decoded_labels = postprocess(tokenizer.batch_decode(labels, skip_special_tokens=True))

        result = {}
        sim = semantic_similarity(decoded_preds, decoded_labels)
        result['SIM'] = sim
        acc = style_accuracy(decoded_preds)
        result['ACC'] = acc
        flnc = fluency(decoded_preds)
        result['FLNC'] = flnc
        j = j_metric(sim, acc, flnc)
        result['J'] = j
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    if model_type == 'maskedlm':
        training_args = TrainingArguments(
            output_dir="../models",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            save_total_limit=2,
            evaluation_strategy = "steps",
            learning_rate=2e-5,
            weight_decay=0.01,
            seed=seed,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )
        trainer.train()
        model.save_pretrained("../models/bert_maskedlm")
        tokenizer.save_pretrained("../models/bert_maskedlm")
    elif model_type == 'seq2seq':
        pass

def train_classifier(epoch, model, optimizer, criterion, train_dataloader, device):
    model.to(device)
    model.train()
    progress_bar = tqdm(train_dataloader)
    for batch in progress_bar:
        optimizer.zero_grad()
        x, y = batch
        x, y = x.to(device), y.reshape(-1, 1).to(device)
        y_hat = model(x)
        loss = criterion(y_hat, y.float())
        loss.backward()
        optimizer.step()

        acc = ((y_hat > 0.7) == y).sum().item() / len(y)

        progress_bar.set_description(f'Epoch: {epoch}, Loss: {loss.item():.4f}, Acc: {acc:.4f}')

def evaluate_classifier(epoch, model, criterion, eval_loader, device):
    model.to(device)
    model.eval()
    progress_bar = tqdm(eval_loader)
    with torch.no_grad():
        for batch in progress_bar:
            x, y = batch
            x, y = x.to(device), y.reshape(-1, 1).to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y.float())

            acc = ((y_hat > 0.7) == y).sum().item() / len(y)

            progress_bar.set_description(f'\tEpoch: {epoch}, Loss: {loss.item():.4f}, Acc: {acc:.4f}')