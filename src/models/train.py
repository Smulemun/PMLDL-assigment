from transformers import TrainingArguments, Trainer, Seq2SeqTrainingArguments, Seq2SeqTrainer
from src.models.metrics import semantic_similarity, style_accuracy, fluency, j_metric
from ..data.postprocess import postprocess
import numpy as np
from tqdm import tqdm
import torch

def train(model_type, model, tokenizer, train_dataset, val_dataset, data_collator, batch_size, epochs, seed):
    '''Function to train a given model'''
    if model_type == 'maskedlm' or model_type == 'maskedlm_with_classifier':
        # training maskedlm model
        training_args = TrainingArguments(
            output_dir="../models",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            save_total_limit=2,
            evaluation_strategy = "epoch",
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
        )
        trainer.train()
        # and saving it after the training
        model.save_pretrained("../models/bert_maskedlm")
        tokenizer.save_pretrained("../models/bert_maskedlm")

    elif model_type == 'seq2seq':
        # training sequence to sequence model
        training_args = Seq2SeqTrainingArguments(
            output_dir="../models/",
            evaluation_strategy = "epoch",
            learning_rate=2e-5,
            weight_decay=0.01,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            predict_with_generate=True,
            save_total_limit=2,
            seed=seed,
        )
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        trainer.train()
        # and saving it after the training
        model.save_pretrained("../models/detoxificator")
        tokenizer.save_pretrained("../models/detoxificator")

    

def train_classifier(epoch, model, optimizer, criterion, train_dataloader, device):
    '''Function to train a given classifier'''
    # training the classifier
    model.to(device)
    model.train()
    progress_bar = tqdm(train_dataloader)
    accs = []
    losses = []
    for batch in progress_bar:
        optimizer.zero_grad()
        x, y = batch
        x, y = x.to(device), y.reshape(-1, 1).to(device)
        y_hat = model(x)
        loss = criterion(y_hat, y.float())
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        accs.append(((y_hat > 0.8) == y).sum().item() / len(y))

        progress_bar.set_description(f'Epoch: {epoch}, Loss: {np.mean(losses):.5f}, Acc: {np.mean(accs):.5f}')

def evaluate_classifier(epoch, model, criterion, eval_loader, device):
    '''Function to validate a given classifier'''
    # validating the classifier
    model.to(device)
    model.eval()
    progress_bar = tqdm(eval_loader)
    accs = []
    losses = []
    with torch.no_grad():
        for batch in progress_bar:
            x, y = batch
            x, y = x.to(device), y.reshape(-1, 1).to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y.float())

            losses.append(loss.item())
            accs.append(((y_hat > 0.8) == y).sum().item() / len(y))

            progress_bar.set_description(f'\tEpoch: {epoch}, Loss: {np.mean(losses):.5f}, Acc: {np.mean(accs):.5f}')
    return np.mean(losses)