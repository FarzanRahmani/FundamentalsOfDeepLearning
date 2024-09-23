import os

import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import get_linear_schedule_with_warmup

from .dataset import SentenceDataset

# with Early Stopping to prevent overfitting
def train_parsbert(model_name, cache_dir, device: torch.device, label_dict, train_sentences, train_labels,
                    val_sentences, val_labels, base_path, epochs=12, early_stop_patience=6):
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_dict),
                                                                cache_dir=cache_dir, ignore_mismatched_sizes=True)
    model = model.to(device)
    results_csv_path = f"{base_path}/stats/{model_name.split('/')[-1]}_train.csv"
    f = open(results_csv_path, "w")
    f.write("epoch,train_loss,train_accuracy,val_loss,val_accuracy\n")

    train_dataset = SentenceDataset(
        sentences=train_sentences.to_list(),
        labels=train_labels.to_list(),
        tokenizer=tokenizer,
        label_dict=label_dict
    )
    val_dataset = SentenceDataset(
        sentences=val_sentences.to_list(),
        labels=val_labels.to_list(),
        tokenizer=tokenizer,
        label_dict=label_dict
    )

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    best_val_loss = float('inf')
    best_epoch = 0
    no_improvement_counter = 0

    # Learning rate scheduler setup
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        description = f"Training Epoch {epoch + 1}"
        progress_bar = tqdm(train_dataloader, desc=description, colour='green')
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            loss = outputs.loss
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            _, predicted = torch.max(outputs.logits, dim=1)
            train_total += batch['labels'].size(0)
            train_correct += (predicted == batch['labels']).sum().item()
            progress_bar.set_postfix({"Loss": loss.item()})

        train_average_loss = train_loss / len(train_dataloader)
        train_accuracy = train_correct / train_total
        print(f"\nTrain Loss: {train_average_loss:.4f} | Train Accuracy: {train_accuracy:.4f}")

        # Evaluate the model
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            description = f"Validation Epoch {epoch + 1}"
            progress_bar = tqdm(val_dataloader, desc=description, colour='yellow')
            for batch in progress_bar:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)

                loss = outputs.loss
                val_loss += loss.item()

                _, predicted = torch.max(outputs.logits, dim=1)
                val_total += batch['labels'].size(0)
                val_correct += (predicted == batch['labels']).sum().item()

            val_average_loss = val_loss / len(val_dataloader)
            val_accuracy = val_correct / val_total
            print(f"Validation Loss: {val_average_loss:.4f} | Validation Accuracy: {val_accuracy:.4f}")
            print('*' * 50)
        f.write(
            f"{epoch + 1},{train_average_loss:.4f},{train_accuracy:.4f},{val_average_loss:.4f},{val_accuracy:.4f}\n")

        # Early stopping and saving the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improvement_counter = 0
            print(f"Saving new best model at epoch {epoch + 1}")
            output_dir = f"{base_path}/models/{model_name}/best"
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        else:
            no_improvement_counter += 1
            if no_improvement_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch + 1}. Best epoch: {best_epoch + 1}")
                break

    f.close()

# L2 regularization with AdamW (better training with better ثبات)
def train_parsbert_with_l2(model_name, cache_dir, device, label_dict, train_sentences, train_labels, val_sentences,
                            val_labels, base_path, epochs=12, weight_decay=1e-3):
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_dict),
                                                                cache_dir=cache_dir, ignore_mismatched_sizes=True)
    # ignore_mismatched_sizes -> ignore the mismatched sizes between the model and the pretrained model
    # e.g. digikala: 2 class -> we have 7 class

    model.to(device)
    results_csv_path = os.path.join(base_path, "stats", f"{model_name.split('/')[-1]}_train.csv")

    with open(results_csv_path, "w") as f:
        f.write("epoch,train_loss,train_accuracy,val_loss,val_accuracy\n")

        train_dataset = SentenceDataset(sentences=train_sentences.to_list(), labels=train_labels.to_list(),
                                        tokenizer=tokenizer, label_dict=label_dict)
        val_dataset = SentenceDataset(sentences=val_sentences.to_list(), labels=val_labels.to_list(),
                                    tokenizer=tokenizer, label_dict=label_dict)

        train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=8)

        optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=weight_decay) # with lambda parameter
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        best_val_loss = float('inf')
        for epoch in range(epochs):
            model.train()
            train_loss, train_correct, train_total = 0, 0, 0
            progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}", colour='green')
            for batch in progress_bar:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)

                loss = outputs.loss
                train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                _, predicted = torch.max(outputs.logits, dim=1)
                train_total += batch['labels'].size(0)
                train_correct += (predicted == batch['labels']).sum().item()

            train_average_loss = train_loss / len(train_dataloader)
            train_accuracy = train_correct / train_total
            print(f"Epoch {epoch + 1}: Train Loss: {train_average_loss:.4f} | Train Accuracy: {train_accuracy:.4f}")

            model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc=f"Validation Epoch {epoch + 1}", colour='yellow'):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)

                    loss = outputs.loss
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.logits, dim=1)
                    val_total += batch['labels'].size(0)
                    val_correct += (predicted == batch['labels']).sum().item()

                val_average_loss = val_loss / len(val_dataloader)
                val_accuracy = val_correct / val_total
                print(f"Epoch {epoch + 1}: Val Loss: {val_average_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")
                print('*' * 50)

            f.write(
                f"{epoch + 1},{train_average_loss:.4f},{train_accuracy:.4f},{val_average_loss:.4f},{val_accuracy:.4f}\n")

            if val_average_loss < best_val_loss: # best result is the lowest validation loss
                best_val_loss = val_average_loss
                output_dir = os.path.join(base_path, "models", model_name, "best")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                print(
                    f"Epoch {epoch + 1}: New best model saved with val_loss {best_val_loss:.4f} & val_acc {val_accuracy:.4f}")


def test_parsbert(model_name, cache_dir, device: torch.device, label_dict, test_sentences, test_labels, base_path):
    # Load the best model
    model_path = f"{base_path}/models/{model_name}/best"
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=len(label_dict),
                                                                cache_dir=cache_dir, ignore_mismatched_sizes=True)
    model = model.to(device)
    results_csv_path = f"{base_path}/stats/{model_name.split('/')[-1]}_test.csv"
    f = open(results_csv_path, "w")
    f.write("test_loss,test_accuracy,precision,recall,f1\n")

    test_dataset = SentenceDataset(
        sentences=test_sentences,
        labels=test_labels,
        tokenizer=tokenizer,
        label_dict=label_dict
    )

    # Assuming `test_dataset` is an instance of `SentenceDataset` and already defined
    test_dataloader = DataLoader(test_dataset, batch_size=8)

    # Test the model
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    predictions, true_labels = [], []

    with torch.no_grad():
        progress_bar = tqdm(test_dataloader, desc="Testing", colour='blue')
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            loss = outputs.loss
            test_loss += loss.item()

            _, predicted = torch.max(outputs.logits, dim=1)
            test_total += batch['labels'].size(0)
            test_correct += (predicted == batch['labels']).sum().item()

            # Collect the predictions and true labels for each batch
            predictions.extend(predicted.view(-1).cpu().numpy())
            true_labels.extend(batch['labels'].view(-1).cpu().numpy())

        # Calculate the average loss and accuracy over all test data
        test_average_loss = test_loss / len(test_dataloader)
        test_accuracy = test_correct / test_total
        print(f"Test Loss: {test_average_loss:.4f} | Test Accuracy: {test_accuracy:.4f}")

        # Compute precision, recall, and F1 score
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='macro') 
        # average='macro': calculate metrics for each label, and find their unweighted mean (each class is equally weighted and has same value)
        
        print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}")
        print('*' * 50)
        f.write(f"{test_average_loss:.4f},{test_accuracy:.4f},{precision:.4f},{recall:.4f},{f1:.4f}\n")
        f.close()


def predict_parsbert(model_name, cache_dir, device, label_dict, text, base_path):
    # Load the model and tokenizer
    model_path = f"{base_path}/models/{model_name}/best"
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, cache_dir=cache_dir)
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode

    # Tokenize the input text
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Convert logits to probabilities (optional)
    probabilities = torch.softmax(logits, dim=1)

    # Get the predicted label index
    predicted_label_index = logits.argmax(dim=1).item()

    # Map the predicted label index to its corresponding label name
    predicted_label_name = {v: k for k, v in label_dict.items()}[predicted_label_index]

    return predicted_label_name, probabilities[0][predicted_label_index].item()
