from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load FAQ
faq_df = pd.read_csv("faq.csv")
questions = faq_df['Question'].astype(str).tolist()
answers = faq_df['Answer'].astype(str).tolist()

# Vectorize questions
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

# Initialize Flask
app = Flask(__name__)
CORS(app)

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").strip()
    if not user_input:
        return jsonify({"reply": "Please type a question."})

    # Compute similarity
    user_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vec, question_vectors)

    max_index = similarities.argmax()
    max_score = similarities[0][max_index]

    if max_score > 0.3:
        reply = answers[max_index]
    else:
        reply = "I'm sorry, I couldn't understand that. Can you rephrase?"

    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(debug=True)
