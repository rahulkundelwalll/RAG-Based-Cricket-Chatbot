from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import sqlite3
from Split_data_in_chunk import SplitDataInChunkSaveChromaDB
from Rag import RAG_CHATBOT

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Initialize Database
def init_db():
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chats (id INTEGER PRIMARY KEY, user TEXT, question TEXT, answer TEXT)''')
    conn.commit()
    conn.close()

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        session['user'] = request.form['name']
        return redirect(url_for('chat'))
    return render_template('index.html')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if 'user' not in session:
        return redirect(url_for('home'))

    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute("SELECT question, answer FROM chats WHERE user = ?", (session['user'],))
    chat_history = c.fetchall()  # List of (question, answer)
    conn.close()

    return render_template('chat.html', user=session['user'], chat_history=chat_history)


# @app.route('/ask', methods=['POST'])
# def ask():
#     if 'user' not in session:
#         return jsonify({'error': 'User not logged in'})
    
#     query = request.json.get('question')
#     rag = RAG_CHATBOT()
#     response = rag.chatbot(query)
    
#     conn = sqlite3.connect('chat_history.db')
#     c = conn.cursor()
#     c.execute("INSERT INTO chats (user, question, answer) VALUES (?, ?, ?)", (session['user'], query, response))
#     conn.commit()
#     conn.close()
    
#     return jsonify({'answer': response})



# dummy
@app.route('/ask', methods=['POST'])
def ask():
    # if 'user' not in session:
    #     return jsonify({'error': 'User not logged in'})
    
    query = request.json.get('question')
    rag = RAG_CHATBOT()
    response = rag.chatbot(query)
    
    # conn = sqlite3.connect('chat_history.db')
    # c = conn.cursor()
    # c.execute("INSERT INTO chats (user, question, answer) VALUES (?, ?, ?)", (session['user'], query, response))
    # conn.commit()
    # conn.close()
    
    return jsonify({'answer': response})

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
