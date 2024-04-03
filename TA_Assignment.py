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

data_frame1 = None
data_frame2 = None

def read_json_and_extract(file_path):
    global data_frame1, data_frame2
    with open(file_path, mode="rt") as f:
        data = [json.loads(line) for line in f][:15000]  # Read only first 15000 lines
        return pd.DataFrame.from_dict(data)[['overall', 'reviewText']]



def lowercase_and_no_punctuation(data_frame):
#     print("\nQuestion 1:")
    try:
        # Fill NaN values with empty strings
        data_frame['reviewText'] = data_frame['reviewText'].fillna('')
        selected_reviews = pd.concat([data_frame.head(10000), data_frame.tail(10000)])

        # Apply a function to convert 'reviewText' to lowercase and remove punctuations
        selected_reviews['reviewText'] = selected_reviews['reviewText'].apply(lambda x: ''.join([char.lower() if char not in string.punctuation else ' ' for char in str(x)]))

        print(selected_reviews['reviewText'])
        return selected_reviews

    except Exception as e:
        print(f'Error: {e}')

def compute_idf(data_frame):
    try:
        # Create a TfidfVectorizer
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)

        # Fit and transform the 'reviewText' column
        tfidf_matrix = tfidf_vectorizer.fit_transform(data_frame['reviewText'])

        # Create a TfidfTransformer to get IDF values
        tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
        tfidf_transformer.fit(tfidf_matrix)

        # Get feature names (words) and corresponding IDF scores
        feature_names = tfidf_vectorizer.get_feature_names_out()
        idf_scores = tfidf_transformer.idf_

        # Combine feature names and IDF scores into a DataFrame
        idf_df = pd.DataFrame({'Word': feature_names, 'IDF': idf_scores})

        # Sort DataFrame by IDF scores in ascending order
        idf_df_sorted = idf_df.sort_values(by='IDF', ascending=True)

        # Display top 30 and bottom 30 words based on IDF
        print("\nTop 30 words based on IDF:")
        print(idf_df_sorted.head(30))

        print("\nBottom 30 words based on IDF:")
        print(idf_df_sorted.tail(30))

    except Exception as e:
        print(f'Error: {e}')



def perform_sentence_detection(data_frame):
    print("\nQuestion 2:")
    try:
        # Load spaCy's English language model
        nlp = spacy.load("en_core_web_sm")

        # Take the first 15 review texts
        for i, review_text in enumerate(data_frame['reviewText'][:10], start=1):
            # Process the review text with spaCy
            doc = nlp(review_text)

            # Extract sentences and print review ID along with each sentence
            sentence_id = 0
            for sent in doc.sents:
                sentence_id += 1
                print(f"Review ID: {i}, Sentence {sentence_id}: {sent.text}")

    except Exception as e:
        print(f'Error: {e}')

def perform_word_tokenization(data_frame):
    print("\nQuestion 3:")
    try:
        # Load spaCy's English language model
        nlp = spacy.load("en_core_web_sm")

        # Take the first 10 review texts
        for index, review_text in enumerate(data_frame['reviewText'][:10], start=1):
            # Process the review text with spaCy
            doc = nlp(review_text)

            # Extract token, lemma, and POS tag, and print review ID along with each token
            for token in doc:
                print(f"Review ID: {index}, Token: {token.text}, Lemma: {token.lemma_}, POS: {token.pos_}")

    except Exception as e:
        print(f'Error: {e}')

def binary_classification(data_frame):
    print("\nQuestion 4:")
    try:
        data_frame['binary_class'] = (data_frame['overall'] == 5).astype(int)
        train_data, test_data, train_labels, test_labels = train_test_split(
            data_frame['reviewText'], data_frame['binary_class'], test_size=0.2, random_state=42
        )
        tfidf_vectorizer = TfidfVectorizer(max_features=50000)
        train_tfidf_matrix = tfidf_vectorizer.fit_transform(train_data)
        test_tfidf_matrix = tfidf_vectorizer.transform(test_data)

        nb_model = MultinomialNB()
        nb_model.fit(train_tfidf_matrix, train_labels)

        binary_predictions = nb_model.predict(test_tfidf_matrix)

        # Evaluate the model
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

    except Exception as e:
        print(f'Error: {e}')

def summarise(data_frame):
    print("\nQuestion 5:")
    try:
        rating_1_reviews = data_frame[data_frame['overall'] == 1.0]['reviewText'].head(1000)
        rating_5_reviews = data_frame[data_frame['overall'] == 5.0]['reviewText'].head(1000)

        summary_1 = summarizer.summarize('\n'.join(rating_1_reviews), ratio=0.01)
        summary_5 = summarizer.summarize('\n'.join(rating_5_reviews), words=300)

        # Print or return the summaries
        print("Summary for Rating 1.0 reviews (1% length):")
        print(summary_1)

        print("\nSummary for Rating 5.0 reviews (300 words):")
        print(summary_5)

    except Exception as e:
        print(f'Error: {e}')


# Read and process AMAZON_FASHION.json
print("\nFor Category 1 - Amazon Fashion:")
data_frame1 = read_json_and_extract('AMAZON_FASHION.json')
data_frame1.to_csv('amazon_fashion.csv', index=False)
data_frame1 = pd.read_csv('amazon_fashion.csv')
print(data_frame1)
processed_df1 = lowercase_and_no_punctuation(data_frame1)
compute_idf(processed_df1)
perform_sentence_detection(processed_df1)
perform_word_tokenization(processed_df1)
binary_classification(data_frame1)
summarise(processed_df1)

# Read and process Patio_Lawn_and_Garden.json
print("\nFor Category 2 - Patio Lawn and Garden:")
data_frame2 = read_json_and_extract('Patio_Lawn_and_Garden.json')
data_frame2.to_csv('patio_lawn_and_garden.csv', index=False)
data_frame2 = pd.read_csv('patio_lawn_and_garden.csv')
print(data_frame2)
processed_df2 = lowercase_and_no_punctuation(data_frame2)
compute_idf(processed_df2)
perform_sentence_detection(processed_df2)
perform_word_tokenization(processed_df2)
binary_classification(data_frame2)
summarise(processed_df2)

# Combine the datasets
combined_data_frame = pd.concat([data_frame1, data_frame2], ignore_index=True)
combined_data_frame.to_csv('Combined_Categories_dataset.csv', index=False)
combined_data_frame = pd.read_csv('Combined_Categories_dataset.csv')


print("\nCombined Dataset:")
print(combined_data_frame)
