from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    """A simple route that returns a welcome message."""
    return "Hello, World! The minimal app is running."

if __name__ == '__main__':
    # This block is for local development and won't be executed by Gunicorn.
    # Railway will use the PORT environment variable provided to Gunicorn.
    app.run(debug=True, port=8080)
