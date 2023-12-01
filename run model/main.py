'''file that runs model '''
import re
# pylint: disable =E0401
from tensorflow.keras.preprocessing.text import Tokenizer
# pylint: disable =E0401
from tensorflow.keras.preprocessing.sequence import pad_sequences
# pylint: disable =E0401
from tensorflow.keras.models import load_model

# Load the saved model
# Load the saved model
model = load_model('sentiment_analyzer.model')

# Texts you want to analyze
new_texts = ["I appreciate you.", "You are perfect just the way you are", "it was a bawdy play full of vulgar jokes."]

# Preprocess the new texts consistently

# pylint: disable =W0621
def preprocess_text(text):
    '''preprocessor'''
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    return text

preprocessed_texts = [preprocess_text(text) for text in new_texts]

tokenizer = Tokenizer(num_words=3000)
tokenizer.fit_on_texts(new_texts)

# Tokenize and pad the new texts using the same tokenizer from training
sequences = tokenizer.texts_to_sequences(preprocessed_texts)
X_new = pad_sequences(sequences, padding='post', truncating='post', maxlen=None)

# Make predictions
predictions = model.predict(X_new)

# Convert predictions to categories
categories = ['Negative', 'Neutral', 'Positive']
categorical_predictions = [categories[p.argmax()] for p in predictions]

# Display the predictions
for text, prediction in zip(new_texts, categorical_predictions):
    print(f'Text: {text}, Sentiment: {prediction}')

