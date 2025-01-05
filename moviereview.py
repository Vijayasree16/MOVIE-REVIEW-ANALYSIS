import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
import string
import threading
import tkinter as tk
from tkinter import messagebox

# Download necessary NLTK data
nltk.download('stopwords')

# Global variables for the model and vectorizer
vectorizer = None
model = None

# 1. Text Preprocessing Function
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()  # Convert to lowercase
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# 2. Function to load and train the model asynchronously
def load_model():
    global df, vectorizer, model
    
    # Load Dataset
    df = pd.read_csv("IMDB Dataset.csv")

    # Check for missing values and drop them
    df.dropna(inplace=True)

    # Convert 'sentiment' column to 'good' or 'bad'
    df['label'] = df['sentiment'].apply(lambda x: 'good' if x == 'positive' else 'bad')

    # Preprocess text
    df['cleaned_review'] = df['review'].apply(preprocess_text)

    # Sample dataset to speed up loading for testing purposes
    df = df.sample(frac=0.1, random_state=42)  # 10% of the data for testing

    # 3. Feature Extraction (TF-IDF Vectorization)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['cleaned_review'])
    y = df['label']

    # 4. Split the Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Model Training
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 6. Evaluation
    y_pred = model.predict(X_test)
    print("Model loaded and trained.")
    print("Accuracy:", accuracy_score(y_test, y_pred))

# Start model loading in a separate thread
load_thread = threading.Thread(target=load_model)
load_thread.start()

# 7. Prediction Function
def predict_review(review):
    if model is None or vectorizer is None:
        return "Model is still loading. Please wait..."
    
    cleaned_review = preprocess_text(review)
    vectorized_review = vectorizer.transform([cleaned_review])
    prediction = model.predict(vectorized_review)
    return prediction[0]

# 8. tkinter GUI setup
def create_gui():
    # Function to handle button click
    def on_predict_click():
        review = review_input.get("1.0", tk.END).strip()  # Get input from text box
        if not review:
            messagebox.showerror("Input Error", "Please enter a review.")
            return

        prediction = predict_review(review)
        if prediction == "Model is still loading. Please wait...":
            result_label.config(text=prediction)
        else:
            result_label.config(text=f"The review is classified as '{prediction}'", fg="#4CAF50" if prediction == "good" else "#f44336")

    # Create tkinter window
    root = tk.Tk()
    root.title("Movie Review Sentiment Classifier")
    root.geometry("500x400")
    root.configure(bg="#f0f0f0")  # Set background color

    # Add a title label
    title_label = tk.Label(root, text="Movie Review Sentiment Classifier", font=("Helvetica", 16, "bold"), bg="#3F51B5", fg="white")
    title_label.pack(pady=20, fill=tk.X)

    # Create input box for the review with a colorful background
    tk.Label(root, text="Enter a movie review:", bg="#f0f0f0", font=("Arial", 12)).pack(pady=5)
    review_input = tk.Text(root, height=8, width=60, bg="#E3F2FD", fg="black", font=("Arial", 10))
    review_input.pack(pady=10)

    # Create a colorful button to predict
    predict_button = tk.Button(root, text="Predict Sentiment", command=on_predict_click, bg="#4CAF50", fg="white", font=("Arial", 12, "bold"))
    predict_button.pack(pady=20)

    # Create label to show result with larger font and bold style
    result_label = tk.Label(root, text="", bg="#f0f0f0", font=("Arial", 14, "bold"))
    result_label.pack(pady=10)

    # Run the tkinter loop
    root.mainloop()

# Start the tkinter GUI
create_gui()
