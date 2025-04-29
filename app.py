import os
import subprocess
from flask import Flask, render_template, request, jsonify

# Configure Flask to look for templates in the project root's "templates" folder.
template_dir = os.path.join(os.path.dirname(__file__), '..', 'templates')
app = Flask(__name__, template_folder=template_dir)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat_api():
    data = request.get_json()
    query = data.get('query')
    if not query:
        return jsonify({'error': 'No query provided'}), 400

    try:
        # Locate main.py in the current directory.
        main_py_path = os.path.join(os.path.dirname(__file__), 'main.py')

        # Run main.py as a subprocess, feeding the query as input.
        proc = subprocess.run(
            ['python', main_py_path],
            input=query.encode('utf-8'),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60
        )
        stdout = proc.stdout.decode('utf-8')
        stderr = proc.stderr.decode('utf-8')

        if stderr:
            return jsonify({'error': stderr.strip()}), 500

        # Extract the final answer from the output.
        parts = stdout.split("Final Answer:")
        if len(parts) > 1:
            reply = parts[1].strip()
        else:
            reply = stdout.strip()

        return jsonify({'response': reply})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
