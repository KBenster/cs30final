from flask import Flask
import sqlite3

app = Flask(__name__)

@app.route('/')
def main():
    return "test"

app.run()