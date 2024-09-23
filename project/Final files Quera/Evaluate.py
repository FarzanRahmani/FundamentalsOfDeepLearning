# gets a csv, and cleans it, and evaluates it: returns accuracy, precision, recall, f1
def evaluate_bert(model_name, cache_dir, device, label_dict, csv_path, base_path, batch_size=16, use_url=False):
    df = pd.read_csv(csv_path)
    df['text'] = df['text'].apply(clean_text)
    df['label'] = df['label'].apply(lambda x: x.upper())
    test_sentences, test_labels = df['text'], df['label']

    test_dataset = SentenceDataset(
        sentences=test_sentences,
        labels=test_labels,
        tokenizer=AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir),
        label_dict=label_dict
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    # Load the best model
    if use_url:
        model_path = model_name
    else:
        model_path = f"{base_path}/models/{model_name}/best"
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=len(label_dict), cache_dir=cache_dir, ignore_mismatched_sizes=True)
    model = model.to(device)

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
        print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}")
