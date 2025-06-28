from flask import Flask, request, jsonify, render_template
import json

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('chatbot.html')

@app.route('/get_response', methods=['POST'])
def chat_response():
    user_query = request.json.get('query')
    response = get_response(user_query)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
