from sklearn.feature_extraction.text import TfidfVectorizer

# Sample documents
documents = [
    "the sky is blue",
    "the sun is bright",
    "the sun in the blue sky is bright",
    "we can see the shining sun, the bright sun"
]

# Create a TfidfVectorizer object
vectorizer = TfidfVectorizer()

# Fit and transform the documents
tfidf_matrix = vectorizer.fit_transform(documents)

# Display the resulting TF-IDF feature matrix
print(tfidf_matrix.toarray())

# Display feature names (words)
print(vectorizer.get_feature_names_out())
