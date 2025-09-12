import os
from os import environ
import sys
import requests

# This is required to make imports work consistently across different
# machines. This needs to be executed before other imports
basedir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(basedir, "..")))

from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_socketio import SocketIO
from utils.logging_config import configure_logging
from request_handlers import inference_handler
from dotenv import load_dotenv


app = Flask(__name__)
app.secret_key = os.urandom(24)  # Secret key for session management

socket_io = SocketIO(app, always_connect=True, logger=False, engineio_logger=False)
socket_io = inference_handler.add_as_websocket_handler(socket_io, app)

TRAINING_SERVER_URL = environ.get("TRAINING_SERVER_URL")


configure_logging(app)

# Sample user data
users = {
    'user1': 'banaan',
    'user2': 'password2',
    'maarten' : 'maarten',
    'eddy' : 'eddy',
    'abdul' : 'abdul', 
    'silas' : 'silas',
    'rob' : 'rob',
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

@app.route('/unlabeled')
def unlabeled_page():
    if 'username' in session:
        return render_template('unlabeled_overview.html', username=session['username'])
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


@app.route("/api/labeling_tasks")
def get_labeling_tasks_proxy():
    try:
        print("Requesting labeling tasks from:", f"{TRAINING_SERVER_URL}/labeling_tasks")
        response = requests.get(f"{TRAINING_SERVER_URL}/labeling_tasks")
        response.raise_for_status()
        return jsonify(response.json()["tasks"])
    except requests.RequestException as e:
        print("Error fetching labeling_tasks:", e)
        return jsonify([]), 500

@app.route('/label')
def label_task():
    if 'username' in session:
        dataset_id = request.args.get("dataset")
        if not dataset_id:
             "No dataset_id provided", 400
        return render_template("label.html", dataset_id=dataset_id, username=session['username'])
    return redirect(url_for('login'))


# Proxy endpoint to fetch labeling task data for a specific dataset from the training server
@app.route("/api/labeling_task_data/<dataset_id>")
def get_labeling_task_data_proxy(dataset_id):
    try:
        url = f"{TRAINING_SERVER_URL}/labeling_task_data/{dataset_id}"
        print(f"Fetching labeling task data from: {url}")
        response = requests.get(url)
        response.raise_for_status()
        return jsonify(response.json())
    except requests.RequestException as e:
        print("Error fetching labeling_task_data:", e)
        return jsonify({"error": "Could not fetch labeling_task_data"}), 500


def run_app():
    return socket_io.run(app=app, host='0.0.0.0', port=5252, debug=True)

def get_app():
    load_dotenv()
    return app

if __name__ == '__main__':
    load_dotenv()
    run_app()
