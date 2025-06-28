from transformers import pipeline
import json

# Load intents from JSON
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Load pre-trained zero-shot classification model
classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

def get_response(user_query):
    """
    Match the user query to the most relevant intent and return a response.
    """
    # Extract all tags and patterns from intents
    tags = [intent['tag'] for intent in intents['intents']]
    patterns = [" ".join(intent['patterns']) if intent['patterns'] else intent['tag'] for intent in intents['intents']]

    # Use the zero-shot classifier to find the best matching tag
    result = classifier(user_query, patterns)
    matched_pattern = result['labels'][0]

    # Find the corresponding tag
    for i, pattern in enumerate(patterns):
        if pattern == matched_pattern:
            matched_tag = tags[i]
            break
    else:
        matched_tag = 'fallback'

    # Get a response for the matched tag
    for intent in intents['intents']:
        if intent['tag'] == matched_tag:
            return intent['responses'][0]

    # Default fallback response
    return "I'm sorry, I didn't understand that."
