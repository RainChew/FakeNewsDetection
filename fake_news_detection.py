import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load your pre-trained models (LR, DT, nb) from pickle files
with open('RF_model.pkl', 'rb') as rf_file:
    RF = pickle.load(rf_file)

with open('DT_model.pkl', 'rb') as dt_file:
    DT = pickle.load(dt_file)

with open('nb_model.pkl', 'rb') as nb_file:
    nb = pickle.load(nb_file)

# Load the vectorizer from a pickle file
with open('vectorization.pkl', 'rb') as vectorization_file:
    vectorization = pickle.load(vectorization_file)

# Create a Streamlit web app
st.title('Fake News Detection App')
st.image('fake.jpg', use_column_width=True)  # Add a header image

# Add instructions
st.markdown("Enter a piece of news text in the text area below and click 'Analyze' to check if it's fake or real.")

# Create a text input for users to enter news text
news_text = st.text_area('Enter the news text for analysis:', '')

# Create a button to perform fake news detection
if st.button('Analyze'):
    if not news_text.strip():  # Handle empty input
        st.warning("Please enter news text for analysis.")
    else:

            # Perform predictions using the Naive Bayes model
            input_data = [news_text]
            vectorized_input_data = vectorization.transform(input_data)
            prediction_nb = nb.predict(vectorized_input_data)
            prediction_dt = DT.predict(vectorized_input_data)
            prediction_rf = RF.predict(vectorized_input_data)

            # Display the result and confidence (if available)
            if prediction_nb == 1:
                result = "The News is REAL"
            else:
                result = "The News is FAKE"
            st.write(f"Result: {result}")
            
            # Add model information
            st.info("Model Used: Naive Bayes")

            if prediction_dt == 1:
                result = "The News is REAL"
            else:
                result = "The News is FAKE"
            st.write(f"Result: {result}")
            st.info("Model Used: Decision Tree")

            if prediction_rf == 1:
                result = "The News is REAL"
            else:
                result = "The News is FAKE"
            st.write(f"Result: {result}")

            # Add model information
            st.info("Model Used: Random Forest")
            
