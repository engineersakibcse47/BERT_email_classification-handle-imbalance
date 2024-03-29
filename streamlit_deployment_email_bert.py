import streamlit as st
import tensorflow as tf
import tensorflow_text as text

# Load my model
reloaded_model = tf.keras.models.load_model('./spam_model_2')

# Define the Streamlit app
def main():
    st.title("Spam Email Classification")
    user_input = st.text_input("Enter your sample here:")
    predict_button = st.button("Predict")

    if predict_button:
        if user_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            prediction = reloaded_model.predict([user_input])
            # Display the prediction score
            formatted_score = "{:.4f}".format(prediction[0][0])
            st.write(f"Input: {user_input}")
            st.write(f'Prediction Score: {formatted_score}')

if __name__ == "__main__":
    main()
