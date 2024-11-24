from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from simulation import run_simulation  # 假设 simulation.py 包含一个 run_simulation 函数

app = Flask(__name__)

@app.route('/run_simulation', methods=['POST'])
def run_simulation_endpoint():
    try:
        # 获取请求中的参数
        data = request.json
        scenario = data.get("scenario", "peace")  # 默认场景
        population_data = data.get("population_data", {})  # 默认人口分布

        # 调用 simulation.py 的函数
        results = run_simulation(scenario, population_data)

        # 返回结果
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/')
def index():
    return app.send_static_file('index.html')  # 前端静态页面

if __name__ == "__main__":
    app.run(debug=True)