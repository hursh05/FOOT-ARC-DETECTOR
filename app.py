from flask import Flask, render_template, redirect, url_for
import subprocess
import threading
import time
import os

app = Flask(__name__, template_folder='html')  # Specify the template folder

# Function to run Streamlit app
def run_streamlit():
    streamlit_path = os.path.join(os.path.dirname(__file__), 'streamlit.py')
    subprocess.run(["streamlit", "run", streamlit_path])

# Endpoint to trigger Streamlit app start
@app.route('/run_streamlit')
def run_streamlit_endpoint():
    # Start Streamlit in a new thread
    threading.Thread(target=run_streamlit).start()
    # Allow some time for Streamlit to start
    time.sleep(2)
    return redirect(url_for('index'))

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
