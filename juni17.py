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
from nltk.corpus import wordnet
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
from sklearn.preprocessing import LabelEncoder
logging.basicConfig(level=logging.INFO)
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
nltk.download('wordnet')

# Download NLTK packages
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
print(nltk.data.path)

def load_model(model_name, model_type="classification"):
    try:
        trust_remote_code = True  # potrzeby zaufanego kodu
        if model_type == "classification":
            model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        elif model_type == "generation":
            model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Dodane
        logging.info("Pomyślnie załadowano model")
        return model, tokenizer
    except Exception as e:
        logging.error(f"Nie udało się załadować modelu: {e}")
        return None, None

def load_training_data(folder_path):
    data = []
    data_files = glob.glob(os.path.join(folder_path, 'valid*.json'))
    for file in data_files:
        try:
            with open(file, 'r') as f:
                data.extend(json.load(f))
        except Exception as e:
            logging.error(f"Nie udało się załadować danych treningowych z pliku {file}: {e}")
    return data

def tokenize_and_clean(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalnum()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    cleaned_text = ' '.join(tokens)
    if not cleaned_text:
        logging.warning("Tekst stał się pusty po tokenizacji i oczyszczeniu.")
    return cleaned_text

def evaluate_classification_model(model, test_data, test_labels):
    if model is None:
        logging.error("Model to None. Nie można ocenić.")
        return 0
    predictions = model.predict(test_data)
    accuracy = accuracy_score(test_labels, predictions)

    # Macierz pomyłek
    cm = confusion_matrix(test_labels, predictions)

    # Precyzja, czułość i F1
    precision = precision_score(test_labels, predictions, average='weighted')
    recall = recall_score(test_labels, predictions, average='weighted')
    f1 = f1_score(test_labels, predictions, average='weighted')

    # Raport klasyfikacji
    class_report = classification_report(test_labels, predictions)

    # Wypisz statystyki
    print(f"Dokładność: {accuracy}")
    print(f"Macierz pomyłek: \n{cm}")
    print(f"Precyzja: {precision}")
    print(f"Czułość: {recall}")
    print(f"Miara F1: {f1}")
    print(f"Raport klasyfikacji: \n{class_report}")

    return accuracy

def save_model_to_safetensor(model, path):
    try:
        torch.save(model.state_dict(), path, _use_new_zipfile_serialization=False, pickle_protocol=2)
        logging.info(f"Model zapisany w {path}")
    except Exception as e:
        logging.error(f"Nie udało się zapisać modelu: {e}")

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

def process_data(data, classification_model, classification_tokenizer, generation_model, generation_tokenizer, batch_size=100):
    processed_data = []
    for i in range(0, len(data), batch_size):
        batch_data = data[i:i+batch_size]
        for item in batch_data:
            text = item.get('text', '')
            if text:  # Dodanie warunku sprawdzającego, czy tekst jest pusty
                sentiment = analyze_sentiment(text, classification_model, classification_tokenizer)
                processed_data.append({
                    'original_text': text,
                    'sentiment_label': sentiment['label'],
                })
            else:
                logging.warning("Pominięto pusty tekst.")
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
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cpu")
    attention_mask = inputs['attention_mask']  # Dodane
    outputs = model.generate(inputs.input_ids, attention_mask=attention_mask, max_length=512, num_return_sequences=2, do_sample=True, top_p=0.85, temperature=0.7)
    generated_text = tokenizer.decode(outputs[0])
    return generated_text


def load_and_transform_data(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Zakładamy, że 'text' i 'context' są cechami, a 'fn_call' to etykieta
        texts = [item['text'] for item in data]
        contexts = [item['context'] for item in data]
        fn_calls = [item['fn_call'] for item in data]

        # Przykładowa przekształcenie tekstu i kontekstu na liczby (tutaj długości stringów)
        # W praktyce można użyć bardziej zaawansowanych technik jak embeddingi
        texts_lengths = [len(text) for text in texts]
        contexts_lengths = [len(context) for context in contexts]

        # Kodowanie etykiet
        label_encoder = LabelEncoder()
        fn_calls_encoded = label_encoder.fit_transform(fn_calls)

        # Tworzenie tensorów
        features = np.array([texts_lengths, contexts_lengths]).T
        features_tensor = torch.tensor(features, dtype=torch.float32)
        labels_tensor = torch.tensor(fn_calls_encoded, dtype=torch.int64)

        return TensorDataset(features_tensor, labels_tensor)

    except Exception as e:
        print(f"Failed to load and transform data: {e}")
        return None

def display_results_and_ask_questions(generated_text_result):
    table = PrettyTable()
    table.field_names = ["Generated Text"]
    table.add_row([generated_text_result + '...'])  # Wyświetlanie tylko pierwszych 50 znaków
    print(table)

    while True:
        user_prompt = input("Wprowadź prompt lub wpisz 'quit' aby zakończyć: ")
        if user_prompt.lower() == 'quit':
            break
        generated_text = generate_text(user_prompt, generation_model, generation_tokenizer)
        table.add_row([generated_text + '...'])
        print(table)

if __name__ == '__main__':
    with torch.no_grad():
        # Wyświetlenie nazw modeli
        print("Ładowanie modelu klasyfikacji: bert-base-cased...")
        print("Ładowanie modelu generacji: gpt2-xl..")

        # Ładowanie modeli i tokenizerów
        classification_model, classification_tokenizer = load_model("bert-base-cased", "classification")
        generation_model, generation_tokenizer = load_model("gpt2-xl", "generation")

        # Ładowanie i wyświetlanie danych
        print("Ładowanie danych treningowych...")
        data = load_training_data('./')
        print(f"Załadowano {len(data)} próbek treningowych.")

        # Wczytanie i przekształcenie danych
        tensor_data = load_and_transform_data('cleaned_train.json')
        if tensor_data:
            train_loader = DataLoader(tensor_data, batch_size=32, shuffle=True)

        # Czyszczenie i tokenizacja
        print("Oczyszczanie danych do tokenizacji...")
        cleaned_data = [tokenize_and_clean(item.get('text', '')) for item in data]
        labels = [item.get('label', 0) for item in data]

        # Wyświetlenie przykładowych danych w tabeli
        table = PrettyTable()
        table.field_names = ["Original Text", "Cleaned Text", "Label"]
        for original, cleaned, label in zip(data[:5], cleaned_data[:5], labels[:5]):
            table.add_row([original.get('text', '')[:50], cleaned[:50], label])
        print("Pierwsze 5 próbek do nauki:")
        print(table)

        # Trenowanie i ewaluacja modelu klasyfikacji
        print("Rozpoczynam trenowanie modelu klasyfikacji...")
        # Dodajemy długość tekstu jako dodatkową cechę
        cleaned_data_lengths = np.array([len(text) for text in cleaned_data]).reshape(-1, 1)
        features = np.concatenate([cleaned_data_lengths], axis=1)

        classification_model_sklearn = train_classification_model(features, labels)
        if classification_model_sklearn:
            accuracy = evaluate_classification_model(classification_model_sklearn, cleaned_data, labels)
            print(f"Dokładność modelu klasyfikacji: {accuracy}")


        # Remove unnecessary data
        del cleaned_data, labels
        gc.collect()

        # Data processing
        print("Przetwarzanie danych...")
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
        prompt = "Hey, Describe python and tell me webtorrent video stream."
        print(f"Wygenerowany text przez prompt: {prompt}")
        generated_text_result = generate_text(prompt, generation_model, generation_tokenizer)

        # Display the results and allow for further queries
        display_results_and_ask_questions(generated_text_result)
