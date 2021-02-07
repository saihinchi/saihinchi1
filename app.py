from flask import Flask
import os

app = Flask(__name__)

@app.route('/')
def index():
    return 'Hello,heroku'

if __name__ == '__main__':
    port = int(os.getenv('PORT',5000))
    app.run(debug=False)