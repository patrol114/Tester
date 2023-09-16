import json
import os
import glob
import logging
import gc

logging.basicConfig(level=logging.INFO)

# Load training data function
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

# Text Tokenization and Cleaning
def tokenize_and_clean(text):
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer

    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalnum()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Model Loading
def load_model(model_name, model_type="classification"):
    from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer
    try:
        trust_remote_code = True
        if model_type == "classification":
            model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        elif model_type == "generation":
            model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        logging.info("Model successfully loaded")
        return model, tokenizer
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return None, None

# Main Execution
if __name__ == '__main__':
    # Load and clean the data
    data = load_training_data('./')
    cleaned_data = [tokenize_and_clean(item.get('text', '')) for item in data]
    labels = [item.get('label', 0) for item in data]

    # Train the classification model
    from sklearn.pipeline import make_pipeline
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import accuracy_score

    model = make_pipeline(CountVectorizer(), MultinomialNB())
    model.fit(cleaned_data, labels)

    # Evaluate the model
    accuracy = accuracy_score(labels, model.predict(cleaned_data))
    logging.info(f"Classification model accuracy: {accuracy}")

    # Optional: Load and use the transformer model
    with gc.collect():  # Trigger garbage collection
        classification_model, classification_tokenizer = load_model("bert-base-uncased", "classification")

    # Clean up
    del model, cleaned_data, labels, classification_model, classification_tokenizer
    gc.collect()
