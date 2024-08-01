# SNU Chatbot
This is a chatbot built using natural language processing (NLP) techniques to answer frequently asked questions (FAQs) about Shiv Nadar University Chennai (SNU Chennai).

## Dependencies
The chatbot code relies on several external libraries:

nltk\
numpy\
tensorflow\
keras\
Flask

These libraries can be installed using the pip command:
```bash
pip install nltk numpy tensorflow keras Flask
```
Use code with caution.

### Data Files
The chatbot utilizes three data files:

words.pkl: This file stores a pickled list of words used for training the chatbot model.

classes.pkl: This file stores a pickled list of classes (intents) that the chatbot can recognize.

chatbotmodel.h5: This file stores the trained Keras model used for intent classification.

### Code Structure
The chatbot code consists of several functions:

1) clean_up_sentence(sentence): This function preprocesses the user's input by tokenizing the sentence and lemmatizing the words.
2) bag_of_words(sentence): This function creates a bag-of-words representation of the sentence.
3) predict_class(sentence): This function predicts the most likely class (intent) for the given sentence using the trained model.
4) get_response(intents_list, intents_json): This function retrieves a response from the chatbot based on the predicted intent.

\
The code also includes a Flask application that serves the chatbot interface. The / route renders the base HTML template, and the /get route handles user messages and returns chatbot responses.

### Running the Chatbot
1) Download the chatbot code and data files.
2) Install the required libraries using pip install <library_name>.
3) Open a terminal in the chatbot directory and run:
```bash
 python chatbot.py.
```
The chatbot will start running and prompt you to enter messages. Enter a question and press Enter to get a response.

### Deploying the Chatbot
The provided Flask application can be deployed on a web server to create a web-based chatbot interface.

Note: This is a basic example and can be further enhanced with features like conversation history, personalization, and integration with external APIs.
