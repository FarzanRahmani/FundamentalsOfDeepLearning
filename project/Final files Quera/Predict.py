def predict_bert(model_name, cache_dir, device, label_dict, text, base_path, use_url=False):
    # Load the model and tokenizer
    if use_url:
        model_path = model_name
    else:
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
