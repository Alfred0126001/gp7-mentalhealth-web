# app.py

from flask import Flask, render_template
from simulation import run_simulation

app = Flask(__name__)

@app.route('/')
def index():
    outputs = run_simulation()
    return render_template('index.html', outputs=outputs)

if __name__ == '__main__':
    app.run(debug=True)
