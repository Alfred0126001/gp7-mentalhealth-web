from flask import Flask, request, jsonify, send_file
from simulation import run_simulation

app = Flask(__name__, static_folder="static")

@app.route("/")
def index():
    return send_file("static/index.html")

@app.route("/run_simulation", methods=["POST"])
def simulate():
    data = request.json
    scenario = data.get("scenario", "peace")
    population = data.get("population_data", {
        "healthy": 800000,
        "mild": 20000,
        "moderate": 5000,
        "severe": 3000,
    })

    results, images = run_simulation(scenario=scenario, initial_population=population)
    return jsonify({"results": results, "images": images})
