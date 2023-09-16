import json
import os
import glob
import logging
import gc
import string
import pickle
import nltk
import torch
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer, pipeline
from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW
from prettytable import PrettyTable
from transformers import BertConfig, BertForSequenceClassification  # Dodane importy

logging.basicConfig(level=logging.INFO)

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Download NLTK packages
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

def load_model(model_name, model_type="classification"):
    try:
        trust_remote_code = True  # Dodane na potrzeby zaufanego kodu
        if model_type == "classification":
            model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        elif model_type == "generation":
            model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Dodane
        logging.info("Model successfully loaded")
        return model, tokenizer
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return None, None

def initialize_and_train_model(train_data, train_labels, val_data, val_labels, model_name="bert-base-uncased", num_labels=2, epochs=2):
    """
    Initialize and train a BertForSequenceClassification model.
    :param train_data: Training data (PyTorch tensor)
    :param train_labels: Training labels (PyTorch tensor)
    :param val_data: Validation data (PyTorch tensor)
    :param val_labels: Validation labels (PyTorch tensor)
    :param model_name: the name of the model or path
    :param num_labels: the number of labels for the classification task
    :param epochs: number of training epochs
    :return: trained model
    """
    # Initialize config
    config = BertConfig.from_pretrained(model_name, num_labels=num_labels)

    # Initialize model
    model = BertForSequenceClassification(config)
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # Create DataLoaders
    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Training loop
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            input_data, labels = batch
            outputs = model(input_data, labels=labels)
            loss = outputs[0]
            loss.backward()
            optimizer.step()

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_data, labels = batch
                outputs = model(input_data, labels=labels)
                val_loss += outputs[0].item()

        print(f"Epoch {epoch+1}, Validation loss: {val_loss / len(val_loader)}")

    return model

def load_training_data(folder_path):
    data = []
    data_files = glob.glob(os.path.join(folder_path, 'valid*.json'))
    for file in data_files:
        try:
            with open(file, 'r') as f:
                data.extend(json.load(f))
        except Exception as e:
            logging.error(f"Failed to load training data from {file}: {e}")
    return data

def load_data_chunk(folder_path, filename, start=0, chunk_size=10):
    """
    Load a chunk of data starting from 'start' index with 'chunk_size' elements.
    """
    file_path = os.path.join(folder_path, filename)
    data_chunk = []
    with open(file_path, 'r') as f:
        data = json.load(f)
        data_chunk = data[start:start + chunk_size]
    return data_chunk

def train_model_in_chunks(train_data, train_labels, chunk_size=10):
    """
    Train a model using chunks of data.
    """
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    for i in range(0, len(train_data), chunk_size):
        train_data_chunk = train_data[i:i + chunk_size]
        train_labels_chunk = train_labels[i:i + chunk_size]
        model.fit(train_data_chunk, train_labels_chunk)
    return model

def tokenize_and_clean(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalnum()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def evaluate_classification_model(model, test_data, test_labels):
    if model is None:
        logging.error("Model is None. Cannot evaluate.")
        return 0
    predictions = model.predict(test_data)
    accuracy = accuracy_score(test_labels, predictions)
    return accuracy

def save_model_to_safetensor(model, path):
    try:
        torch.save(model.state_dict(), path, _use_new_zipfile_serialization=False, pickle_protocol=2)
        logging.info(f"Model saved to {path}")
    except Exception as e:
        logging.error(f"Failed to save model: {e}")

def load_model_from_safetensor(model, path):
    try:
        model.load_state_dict(torch.load(path))
        logging.info(f"Model loaded from {path}")
        return model
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return None

def save_classification_model(model, file_path="model.pkl"):
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)

def load_classification_model(file_path="model.pkl"):
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    return model

def analyze_sentiment(text, model, tokenizer):
    sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return sentiment_analyzer(text)[0]

def process_data(data, classification_model, classification_tokenizer, generation_model, generation_tokenizer, batch_size=50):
    processed_data = []
    for i in range(0, len(data), batch_size):
        batch_data = data[i:i+batch_size]
        for item in batch_data:
            text = item.get('text', '')
            sentiment = analyze_sentiment(text, classification_model, classification_tokenizer)
            processed_data.append({
                'original_text': text,
                'sentiment_label': sentiment['label'],
            })
    return processed_data

def train_classification_model(train_data, train_labels):
    if not train_data or all(not d for d in train_data):
        logging.error("Training data is empty or contains only stop words. Cannot train the model.")
        return None
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    model.fit(train_data, train_labels)
    return model

def generate_text(prompt, model, tokenizer):
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cpu")
    outputs = model.generate(inputs.input_ids, max_length=500, num_return_sequences=1, do_sample=True, top_p=0.95, temperature=0.7)
    generated_text = tokenizer.decode(outputs[0])
    return generated_text


if __name__ == '__main__':
    with torch.no_grad():
        # Display model names
        print(f"Loading classification model: bert-base-uncased")
        print(f"Loading generation model: Deci/DeciLM-6b")

        # Dummy data for demonstration
        train_data = torch.tensor([])  # Your training data tensor
        train_labels = torch.tensor([])  # Your training labels tensor
        val_data = torch.tensor([])  # Your validation data tensor
        val_labels = torch.tensor([])  # Your validation labels tensor

        bert_model = initialize_and_train_model(train_data, train_labels, val_data, val_labels, model_name="bert-base-uncased", num_labels=2, epochs=2)

        # Load models and tokenizers
        classification_model, classification_tokenizer = load_model("bert-base-uncased", "classification")
        generation_model, generation_tokenizer = load_model("Deci/DeciLM-6b", "generation")

        # Load and display data
        print("Loading training data...")
        data = load_training_data('./')
        print(f"Loaded {len(data)} training samples.")

        # Loading and processing training data
        train_data_chunk = load_data_chunk('/', 'train.json', start=0, chunk_size=10)
        cleaned_train_data = [tokenize_and_clean(item.get('text', '')) for item in train_data_chunk]
        train_labels = [item.get('fn_call', '') for item in train_data_chunk]

        # Train model in chunks
        trained_model = train_model_in_chunks(cleaned_train_data, train_labels, chunk_size=10)

        # Loading and processing test data
        test_data_chunk = load_data_chunk('/', 'test.json', start=0, chunk_size=10)
        cleaned_test_data = [tokenize_and_clean(item.get('text', '')) for item in test_data_chunk]
        test_labels = [item.get('fn_call', '') for item in test_data_chunk]

        # Cleaning and tokenizing data
        print("Cleaning and tokenizing data...")
        cleaned_data = [tokenize_and_clean(item.get('text', '')) for item in data]
        labels = [item.get('label', 0) for item in data]

        # Display sample data
        print("Sample cleaned data and labels for training:")
        print(cleaned_data[:5])
        print(labels[:5])

        # Train and evaluate classification model
        print("Training classification model...")
        classification_model_sklearn = train_classification_model(cleaned_data, labels)
        if classification_model_sklearn:
            accuracy = evaluate_classification_model(classification_model_sklearn, cleaned_data, labels)
            print(f"Classification model accuracy: {accuracy}")

        # Remove unnecessary data
        del cleaned_data, labels
        gc.collect()

        # Data processing
        print("Processing data...")
        processed_data = None
        if classification_model_sklearn:
            processed_data = process_data(data, classification_model, classification_tokenizer, generation_model, generation_tokenizer)

        # Remove unnecessary objects and clear memory
        del data
        if processed_data:
            del processed_data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Text generation
        prompt = "hey, why is python3? show simple script."
        print(f"Generating text based on the prompt: {prompt}")

        generated_text_result = generate_text(prompt, generation_model, generation_tokenizer)

        # Display the results
        table = PrettyTable()
        table.field_names = ["Prompt", "Generated Text"]
        table.add_row([prompt, generated_text_result])

        print(table)
