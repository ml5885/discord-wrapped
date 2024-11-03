from flask import Flask, send_from_directory

app = Flask(__name__, static_folder='app')

@app.route('/')
def index():
    return send_from_directory('app', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('app', path)

if __name__ == "__main__":
    app.run(debug=True)