import numpy as np
import pandas as pd
import os
import torch
import random
import torch.nn as nn
from transformers import BertTokenizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast
import matplotlib.pyplot as plt
from nltk import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import re
from sklearn.metrics import accuracy_score
import string
import time
import datetime

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def clean(tweet):
    tweet = re.sub(r"@[A-Za-z0-9ğüşöçıİĞÜŞÖÇ_]+",' ',tweet)
    tweet = re.sub(r"#[A-Za-z0-9ğüşöçıİĞÜŞÖÇ]+",' ',tweet)
    tweet = re.sub(r'https?://[A-Za-z0-9ğüşöıçİĞÜŞÖÇ./]+', ' ', tweet)
    tweet = re.sub(r" +", ' ', tweet)
    exclude = set(string.punctuation)
    tweet = ''.join(ch for ch in tweet if ch not in exclude)
    return tweet

def removeNum(tweet):
    tweet = re.sub(r"[0-9]+",' ',tweet)
    return tweet

def plot_sentence_embeddings_length(text_list, tokenizer):
    tokenized_texts = list(map(lambda t: tokenizer.tokenize(t), text_list))
    tokenized_texts_len = list(map(lambda t: len(t), tokenized_texts))
    fig, ax = plt.subplots(figsize=(8, 5));
    ax.hist(tokenized_texts_len, bins=40);
    ax.set_xlabel("Length of Comment Embeddings");
    ax.set_ylabel("Number of Comments");
    plt.show()
    return

def plotMostCommons(df):
    #for clickabait sentiments
    df_clickbait=df[df["clickbait"]==1]

    #for only unigrams
    token_list=[]
    exclude = set(string.punctuation)
    exclude.add('’')
    stop_words=stopwords.words("turkish")

    for i,r in df_clickbait.iterrows():
        text=''.join(ch for ch in df_clickbait["tr_headline"][i] if ch not in exclude) #remove punctuations from the text in order not to distort frequencies
        #text=''.join(ch for ch in df_clickbait["headline"][i]) #with punctuation
        tokens=word_tokenize(text)
        tokens=[tok.lower() for tok in tokens if tok not in stop_words] #remove stopwords from the text in order not to distort frequencies
        token_list.extend(tokens)

    frequencies=Counter(token_list)
    frequencies_sorted=sorted(frequencies.items(), key=lambda k: k[1],reverse=True)
    top_15=dict(frequencies_sorted[0:15])

    plt.rcdefaults()
    fig, ax = plt.subplots()

    # Example data
    ngram = top_15.keys()
    y_pos = np.arange(len(ngram))
    performance = top_15.values()


    ax.barh(y_pos, performance, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ngram)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Counts')
    ax.set_title('Top-15 Most Common Unigrams in Clickbait Headlines')

    plt.show()

    #for negative sentiments
    df_nonclickbait=df[df["clickbait"]==0]
    exclude.add('‘')
    #for only unigrams
    token_list=[]

    for i,r in df_nonclickbait.iterrows():
        text=''.join(ch for ch in df_nonclickbait["tr_headline"][i] if ch not in exclude) #remove punctuations from the text in order not to distort frequencies
        tokens=word_tokenize(text)
        tokens=[tok.lower() for tok in tokens if tok not in stop_words] #remove stopwords from the text in order not to distort frequencies
        token_list.extend(tokens)

    frequencies=Counter(token_list)
    frequencies_sorted=sorted(frequencies.items(), key=lambda k: k[1],reverse=True)
    top_15=dict(frequencies_sorted[0:15])

    plt.rcdefaults()
    fig, ax = plt.subplots()

    # Example data
    ngram = top_15.keys()
    y_pos = np.arange(len(ngram))
    performance = top_15.values()


    ax.barh(y_pos, performance, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ngram)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Counts')
    ax.set_title('Top-15 Most Common Unigrams in Non Clickbait Headlines')

    plt.show()

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

if __name__ == "__main__":
    df = pd.read_csv('tr_clickbait_dataset.csv',  encoding="utf-8-sig")
    df.dropna(inplace=True)

    device = torch.device("cuda")

    # check class distribution
    #print(df['clickbait'].value_counts(normalize = True))

    data_clean = []
    for headline in df['headline']:
        headlined = str(headline)
        headlined = re.sub(r"Diken",' ',headlined)
        #remove three dots!!!!!!!!
        cleaned_headline = clean(headlined)
        cleanedd_headline = removeNum(cleaned_headline)
        data_clean.append(cleanedd_headline)

    x = pd.Series(data_clean)
    x = x.tolist()
    df['tr_headline'] = x
    del df['headline']
    df.reset_index(drop=True, inplace=True)

    #plotMostCommons(df)

    train_text, temp_text, train_labels, temp_labels = train_test_split(df['tr_headline'], df['clickbait'],
                                                                        random_state=42,
                                                                        test_size=0.3,
                                                                        stratify=df['clickbait'])

    # we will use temp_text and temp_labels to create validation and test set
    val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels,
                                                                    random_state=42,
                                                                    test_size=0.5,
                                                                    stratify=temp_labels)

    tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased', do_lower_case=True)

    #plot_sentence_embeddings_length(train_text.values,tokenizer)

    max_seq_len = 64

    # tokenize and encode sequences in the training set
    tokens_train = tokenizer.batch_encode_plus(
        train_text.tolist(),
        max_length = max_seq_len,
        padding=True,
        truncation=True,
        return_token_type_ids=False
    )

    # tokenize and encode sequences in the validation set
    tokens_val = tokenizer.batch_encode_plus(
        val_text.tolist(),
        max_length = max_seq_len,
        padding=True,
        truncation=True,
        return_token_type_ids=False
    )
    # for train set
    train_seq = torch.tensor(tokens_train['input_ids'])
    train_mask = torch.tensor(tokens_train['attention_mask'])
    train_y = torch.tensor(train_labels.tolist())

    # for validation set
    val_seq = torch.tensor(tokens_val['input_ids'])
    val_mask = torch.tensor(tokens_val['attention_mask'])
    val_y = torch.tensor(val_labels.tolist())

    #define a batch size
    batch_size = 32 #CAN BE CHANGED????

    # wrap tensors
    train_data = TensorDataset(train_seq, train_mask, train_y)

    # dataLoader for train set
    train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)

    # wrap tensors
    val_data = TensorDataset(val_seq, val_mask, val_y)

    # dataLoader for validation set
    val_dataloader = DataLoader(val_data, sampler = SequentialSampler(val_data), batch_size=batch_size)

    model = BertForSequenceClassification.from_pretrained(
        "dbmdz/bert-base-turkish-cased",
        num_labels = 2,
        output_attentions = False,
        output_hidden_states = False,
    )

    # Running the model on GPU.
    model.cuda()

    optimizer = AdamW(model.parameters(),
                      lr = 2e-5,
                      eps = 1e-8
                      )

    # Number of training epochs (authors recommend between 2 and 4)
    epochs = 4

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)

    # Set the seed value all over the place to make this reproducible.
    #The “seed” is a starting point for the sequence and the guarantee is that if you start from the same seed you will get the same sequence of numbers.
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # We'll store a number of quantities such as training and validation loss,
    # validation accuracy, and timings.
    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    # For each epoch...
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because
            # accumulating the gradients is "convenient while training RNNs".
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()

            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            loss = outputs.loss
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in val_dataloader:

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using
            # the `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():

                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # Get the "logits" output by the model. The "logits" are the output
                # values prior to applying an activation function like the softmax.
                outputs = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask,
                                labels=b_labels)

            # Accumulate the validation loss.
            loss = outputs.loss
            logits = outputs.logits

            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += flat_accuracy(logits, label_ids)


        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(val_dataloader)

        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

    # Display floats with two decimal places.
    pd.set_option('precision', 2)

    # Create a DataFrame from our training statistics.
    df_stats = pd.DataFrame(data=training_stats)

    # Use the 'epoch' as the row index.
    df_stats = df_stats.set_index('epoch')

    # A hack to force the column headers to wrap.
    #df = df.style.set_table_styles([dict(selector="th",props=[('max-width', '70px')])])

    # Display the table.
    df_stats

    # tokenize and encode sequences in the test set
    tokens_test = tokenizer.batch_encode_plus(
        test_text.tolist(),
        max_length = max_seq_len,
        padding=True,
        truncation=True,
        return_token_type_ids=False
    )

    # Convert to tensors.
    prediction_inputs = torch.tensor(tokens_test['input_ids'])
    prediction_masks = torch.tensor(tokens_test['attention_mask'])
    prediction_labels = torch.tensor(test_labels.tolist())

    # Set the batch size.
    batch_size = 32

    # Create the DataLoader.
    prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    # Put model in evaluation mode
    model.eval()

    # Tracking variables
    predictions , true_labels = [], []

    # Predict
    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)

        logits = outputs.logits

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)

    print('    DONE.')

    # Combine the predictions for each batch into a single list of 0s and 1s.
    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    # Combine the correct labels for each batch into a single list.
    flat_true_labels = [item for sublist in true_labels for item in sublist]

    print("Accuracy of BERT is:",accuracy_score(flat_true_labels, flat_predictions))

    print(classification_report(flat_true_labels, flat_predictions))

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()

    output_dir = './model_save/'

    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving model to %s" % output_dir)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Good practice: save your training arguments together with the trained model
    # torch.save(args, os.path.join(output_dir, 'training_args.bin'))


