from flask import Flask, render_template, request, redirect, url_for, session
from flask_socketio import SocketIO
from utils.logging_config import configure_logging
from request_handlers import inference_handler
from dotenv import load_dotenv
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Secret key for session management

socket_io = SocketIO(app, always_connect=True, logger=False, engineio_logger=False)
socket_io = inference_handler.add_as_websocket_handler(socket_io)

configure_logging(app)

# Sample user data
users = {
    'user1': 'banaan',
    'user2': 'password2',
}

@app.route('/')
def index():
    if 'username' in session:
        return render_template('index.html', username=session['username'])
    return redirect(url_for('login'))

@app.route('/kali')
def kali_page():
    if 'username' in session:
        return render_template('kali.html', username=session['username'])
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            session['username'] = username
            return redirect(url_for('index'))
        return "Invalid credentials", 401
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

def run_app():
    return socket_io.run(app=app, host='0.0.0.0', port=5252, debug=False)

def get_app():
    load_dotenv()
    return app

if __name__ == '__main__':
    load_dotenv()
    run_app()
