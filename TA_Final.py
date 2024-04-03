import json
import pandas as pd
import string
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from summa import summarizer

# Function to read JSON file and return a DataFrame
def read_json(file_path):
    with open(file_path, mode= "rt") as f:
        data = [json.loads(line) for line in f]
        return pd.DataFrame.from_dict(data)

# Function to keep only 'reviewText' and 'overall' columns
def only_two_variables(data_frame):
    columns_to_keep = ['reviewText', 'overall']
    return data_frame[columns_to_keep]

# Function for preprocessing: lowercasing and removing punctuation
def lowercase_and_no_punctuation(data_frame):
    data_frame['reviewText'] = data_frame['reviewText'].apply(lambda x: ''.join([char.lower() if char not in string.punctuation else ' ' for char in str(x)]))
    return data_frame

# Function to compute IDF
def compute_idf(data_frame):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    tfidf_matrix = tfidf_vectorizer.fit_transform(data_frame['reviewText'])
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(tfidf_matrix)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    idf_scores = tfidf_transformer.idf_
    idf_df = pd.DataFrame({'Word': feature_names, 'IDF': idf_scores})
    return idf_df.sort_values(by='IDF')

# Function for sentence detection
def perform_sentence_detection(data_frame1, data_frame2):
    nlp = spacy.load("en_core_web_sm")
    combined_first_20 = pd.concat([data_frame1.head(10), data_frame2.head(10)], ignore_index=True)
    for i, review_text in enumerate(combined_first_20['reviewText'], start=1):
        doc = nlp(review_text)
        for sent_id, sent in enumerate(doc.sents, start=1):
            print(f"Review ID: {i}, Sentence {sent_id}: {sent.text}")

# Function for word tokenization on selected reviews from both categories
def perform_word_tokenization(data_frame1, data_frame2):
    nlp = spacy.load("en_core_web_sm")
    combined_first_20 = pd.concat([data_frame1.head(10), data_frame2.head(10)], ignore_index=True)
    for index, review_text in enumerate(combined_first_20['reviewText'], start=1):
        doc = nlp(review_text)
        for token in doc:
            print(f"Review ID: {index}, Token: {token.text}, Lemma: {token.lemma_}, POS: {token.pos_}")

# Function for binary classification
def binary_classification(data_frame):
    data_frame['binary_class'] = (data_frame['overall'] == 5).astype(int)
    train_data, test_data, train_labels, test_labels = train_test_split(
        data_frame['reviewText'], data_frame['binary_class'], test_size=0.2, random_state=42)
    tfidf_vectorizer = TfidfVectorizer(max_features=50000)
    train_tfidf_matrix = tfidf_vectorizer.fit_transform(train_data)
    test_tfidf_matrix = tfidf_vectorizer.transform(test_data)
    nb_model = MultinomialNB()
    nb_model.fit(train_tfidf_matrix, train_labels)
    binary_predictions = nb_model.predict(test_tfidf_matrix)
    print("Binary Classification Report:")
    print(classification_report(test_labels, binary_predictions))
    print("Accuracy:", accuracy_score(test_labels, binary_predictions))

    print("Binary Classification Report:")
    print(classification_report(test_labels, binary_predictions))
    print("Accuracy:", accuracy_score(test_labels, binary_predictions))

    train_data, test_data, train_labels, test_labels = train_test_split(
        data_frame1['reviewText'], data_frame1['overall'], test_size=0.2, random_state=42
    )

    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(max_features=50000)
    train_tfidf_matrix = tfidf_vectorizer.fit_transform(train_data)
    test_tfidf_matrix = tfidf_vectorizer.transform(test_data)

    # Train a Multinomial Naïve Bayes model
    nb_model = MultinomialNB()
    nb_model.fit(train_tfidf_matrix, train_labels)

    # Predictions on the test set
    multiclass_predictions = nb_model.predict(test_tfidf_matrix)

    # Evaluate the model
    print("\nMulticlass Classification Report:")
    print(classification_report(test_labels, multiclass_predictions))
    print("Accuracy:", accuracy_score(test_labels, multiclass_predictions))

# Function for summarization
def summarise(data_frame):
    rating_1_reviews = data_frame[data_frame['overall'] == 1.0]['reviewText'].head(1000)
    rating_5_reviews = data_frame[data_frame['overall'] == 5.0]['reviewText'].head(1000)
    summary_1 = summarizer.summarize('\n'.join(rating_1_reviews), ratio=0.01)
    summary_5 = summarizer.summarize('\n'.join(rating_5_reviews), words=300)
    print("Summary for Rating 1.0 reviews (1% length):", summary_1)
    print("Summary for Rating 5.0 reviews (300 words):", summary_5)

# Read and process both JSON files
print("\nReading the Amazon Fashion Dataframe:")
data_frame1 = read_json('AMAZON_FASHION.json')
print(data_frame1)
print("\nReading the Patio Lawn and Garden dataframe:")
data_frame2 = read_json('Patio_Lawn_and_Garden.json')
print(data_frame2)

# Combine the data frames for both categories
print("\n Combining both the dataframes:")
combined_data_frame = pd.concat([data_frame1, data_frame2], ignore_index=True)
print(combined_data_frame)

# Keep only 'reviewText' and 'overall' columns
combined_data_frame = only_two_variables(combined_data_frame)
print("\n", combined_data_frame)

# Preprocessing: lowercasing and removing punctuation
combined_data_frame = lowercase_and_no_punctuation(combined_data_frame)
print(combined_data_frame)

# Compute IDF and display top/bottom 30 words
idf_df_sorted = compute_idf(combined_data_frame)
print("Top 30 words based on IDF:", idf_df_sorted.head(30))
print("Bottom 30 words based on IDF:", idf_df_sorted.tail(30))

# Perform sentence detection
perform_sentence_detection(data_frame1, data_frame2)

# Perform word tokenization on the first 10 reviews from each category
perform_word_tokenization(data_frame1, data_frame2)

# Binary classification
binary_classification(combined_data_frame)

# Summarization
summarise(combined_data_frame)
