import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load your pre-trained models (LR, DT, nb) from pickle files
with open('LR_model.pkl', 'rb') as lr_file:
    LR = pickle.load(lr_file)

with open('DT_model.pkl', 'rb') as dt_file:
    DT = pickle.load(dt_file)

with open('nb_model.pkl', 'rb') as nb_file:
    nb = pickle.load(nb_file)

# Load the vectorizer from a pickle file
with open('vectorization.pkl', 'rb') as vectorization_file:
    vectorization = pickle.load(vectorization_file)
# Check if the vectorized_input_data is a sparse matrix
if not scipy.sparse.issparse(vectorized_input_data):
    vectorized_input_data = scipy.sparse.csr_matrix(vectorized_input_data)
# Create a Streamlit web app
st.title('Fake News Detection App')

# Create a text input for users to enter news text
news_text = st.text_area('Enter the news text for analysis:', '')

# Create a button to perform fake news detection
if st.button('Detect Fake News'):
    # Perform predictions using your models
    input_data = [news_text]
    vectorized_input_data = vectorization.transform(input_data)
    
    prediction_lr = LR.predict(vectorized_input_data)
    prediction_dt = DT.predict(vectorized_input_data)
    prediction_nb = nb.predict(vectorized_input_data)
    
    # Display the predictions
    if prediction_lr == 0:
        st.write("Logistic Regression: The News is FAKE")
    else:
        st.write("Logistic Regression: The News is REAL")
    
    if prediction_dt == 0:
        st.write("Decision Tree Classifier: The News is FAKE")
    else:
        st.write("Decision Tree Classifier: The News is REAL")
    
    if prediction_nb == 0:
        st.write("Naive Bayes: The News is FAKE")
    else:
        st.write("Naive Bayes: The News is REAL")
