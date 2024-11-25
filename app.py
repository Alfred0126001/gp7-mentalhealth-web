# app.py

from flask import Flask, render_template, request
from simulation import run_simulation

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get user input
        healthy_population = int(request.form.get('healthy_population', 800000))
        mild_cases = int(request.form.get('mild_cases', 20000))
        moderate_cases = int(request.form.get('moderate_cases', 5000))
        severe_cases = int(request.form.get('severe_cases', 3000))
        scenario = request.form.get('scenario', 'peace')

        # Run simulation
        outputs = run_simulation(healthy_population, mild_cases, moderate_cases, severe_cases, scenario)
        return render_template('index.html', outputs=outputs)
    else:
        # Display default page
        return render_template('index.html', outputs=None)

if __name__ == '__main__':
    app.run(debug=True)
