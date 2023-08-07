import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load your dataset (replace with your data loading logic)
data = pd.read_csv('fake_news_data.csv')

# Split dataset into train and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Preprocess text data (replace with your data preprocessing logic)

# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust the max_features parameter

# Transform text data to TF-IDF features
X_train = tfidf_vectorizer.fit_transform(train_data['text'])
X_test = tfidf_vectorizer.transform(test_data['text'])

# Create and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, train_data['label'])

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(test_data['label'], predictions)
classification_rep = classification_report(test_data['label'], predictions)

print(f"Accuracy: {accuracy}")
print("Classification Report:\n", classification_rep)
