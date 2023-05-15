
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader




import matplotlib.pyplot as plt

import numpy as np
import transformers

from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from transformers import TrainingArguments, Trainer
from scipy.stats import pearsonr
import argparse
import os


def make_dataset(train, val, test):
    data = load_dataset('csv', data_files={'train': train,
                                           'val': val,
                                           'test': test})
    return data


def tokenize(observation, tokenizer):
    return tokenizer(observation['Utterance'], padding='max_length', max_length=512)


def make_loaders(tokenized_data, batch_size=8):
    trainloader = DataLoader(tokenized_data['train'], shuffle=True, batch_size=batch_size)
    valloader = DataLoader(tokenized_data['val'], batch_size=batch_size)
    testloader = DataLoader(tokenized_data['test'], batch_size=batch_size)
    return trainloader, valloader, testloader

"""
def train(roberta, trainloader, valloader, optimizer, num_epochs, device,finetune_filename):
    roberta.train()
    running_losses = []
    losses = []
    best_validation_loss = -99999
    val_losses = []

    for epoch in range(num_epochs):

        running_loss = 0
        roberta.train()
        for batch in trainloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = roberta(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
            running_loss += loss.item()
        running_loss /= len(trainloader)
        running_losses.append(running_loss)
        
        del batch
        # early stopping
        roberta.eval()

        val_loss = 0
        with torch.no_grad():
            for batch in valloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = roberta(**batch)
                val_loss += (outputs.loss).item()
            val_loss /= len(valloader)
        print('epoch',epoch,'validation loss:',val_loss)

        if val_loss < best_validation_loss or epoch<=4:
            best_validation_loss = val_loss
            
            roberta.save_pretrained(finetune_filename)  # Save
            val_losses.append(val_loss)
        else:
            del roberta
            roberta = transformers.AutoModelForSequenceClassification.from_pretrained(finetune_filename, num_labels=1, problem_type='regression').to(device)
            break



    return roberta, losses, running_losses, val_losses

def loss_plots(losses,running_losses,val_losses,language):
    # loss plots
    plt.figure(1)
    plt.plot(range(len(losses)), losses)
    plt.xlabel('batch')
    plt.ylabel('Loss')
    plt.title('Training loss over each batch')
    plt.savefig('plots/'+language+'_'+str(lr)+'_train_loss_per_batch.png')

    plt.figure(2)
    plt.plot(range(len(running_losses)), running_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training loss over each epoch')
    plt.savefig('plots/'+language+'_'+str(lr)+'_train_loss_per_epoch.png')

    plt.figure(3)
    plt.plot(range(len(val_losses)), val_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation loss over each epoch')
    plt.savefig('plots/'+language+'_'+str(lr)+'_val_loss_per_epoch.png')
"""
def evaluate(roberta, loader):
    roberta.eval()
    all_predictions = torch.Tensor().cpu()
    all_labels = torch.Tensor().cpu()
    all_predictions.requires_grad_(False)
    all_labels.requires_grad_(False)
    loss = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = roberta(**batch)
            predictions = outputs.logits
            loss += (outputs.loss).item()
            labels = batch['labels']
            all_predictions = torch.cat((all_predictions, predictions.cpu()))
            all_labels = torch.cat((all_labels, labels.cpu()))
            del predictions
            del labels
            del outputs
    loss /= len(loader)
    all_predictions = np.squeeze(all_predictions.cpu().numpy())
    all_labels = np.squeeze(all_labels.cpu().numpy())
    correlation_object = pearsonr(all_predictions, all_labels)
    correlation = correlation_object[0]
    print('predictions',all_predictions)
    print('labels',all_labels)
    return correlation, loss


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions_all = predictions.flatten()
    labels_all = labels.flatten()

    # get mean squared error and corr
    y_pred = predictions_all
    y_true = labels_all
    mse = ((y_true-y_pred)**2).mean()
    corr = pearsonr(y_true,y_pred)
    # return as dictionary
    metrics = {'mean_sq_error': mse,
               'correlation':corr}
    return metrics

if __name__ == "__main__":
    language = ''
    batch_size = 32
    epochs = 50

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    

    #tokenize
    tokenizer = None
    model_name = ''
    for language in ['english','spanish','chinese','japanese','all']:
        #read in data
        train_filename = 'data/'+language + "_train_4-14.csv"
        val_filename = 'data/'+ language + "_val_4-14.csv"
        test_filename = 'data/'+ language + "_test_4-14.csv"
        print(train_filename)




        
        for lr in [5e-6,1e-6]:
            if language == 'english':
                model_name = "roberta-base"
                tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            elif language == 'spanish':
                model_name = "bertin-project/bertin-roberta-base-spanish"
                tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            elif language == 'chinese':
                model_name = "hfl/chinese-roberta-wwm-ext"
                tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            elif language == 'japanese':
                model_name = "rinna/japanese-roberta-base"
                tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=False)
                # docs say to do this as a result of a bug
                tokenizer.do_lower_case = True
            elif language == 'all':
                model_name = "xlm-roberta-base"
                tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

            data = make_dataset(train_filename, val_filename, test_filename)

            tokenized_data = (data.map(tokenize, fn_kwargs={'tokenizer': tokenizer}, batched=True)).remove_columns(
                ['Utterance'])

            tokenized_data.set_format('torch')

            
            del data
            finetune_filename = 'politeness_roberta_' + language + '_batch_size_' + str(batch_size) + '_lr_' + str(lr)


            print('----------------------------------------TRAINING',language,lr)

            #num_labels=1 makes it a regression task
            roberta = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1,
                                                                                      problem_type='regression').to(device)

            optimizer = torch.optim.AdamW(roberta.parameters(), lr=lr)
            # number of epochs was from earlier, one of the args from argparser

            # batch size 64, learning rate start with 1e-3, epochs save model state at every epoch
            # look up how to save model state, save at each epoch, early stop based on when validation loss stops decreasing
            #os.mkdir('/nlp/data/mpressi/'+finetune_filename)
            training_args = TrainingArguments(
                "models/" + language + '/' + str(lr),
                evaluation_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=51,
                logging_strategy="epoch",
                learning_rate=lr,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=48,
                gradient_accumulation_steps=1,
                num_train_epochs=50,
                report_to="none",
            )

            trainer = Trainer(
                model=roberta,
                args=training_args,
                train_dataset=tokenized_data["train"],
                eval_dataset=tokenized_data["val"],
                tokenizer=tokenizer,
                compute_metrics=compute_metrics
            )
            try:
                train_result = trainer.train()
                metrics = {}

                metrics["train_eval_metrics"] = trainer.state.log_history
                test_metrics = trainer.predict(tokenized_data["test"]).metrics
                metrics["test_metrics"] = test_metrics

                trainer.save_metrics("all", metrics)
            except RuntimeError as e:
                print(language,lr,e)
            
            del roberta

            #Command to run in terminal: python standardize_lang.py --filename <filename> --outfile <outfile>
