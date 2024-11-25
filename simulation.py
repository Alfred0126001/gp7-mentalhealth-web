import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def run_simulation(scenario='peace', initial_population=None, sim_time=365, total_doctors=1000, output_dir='static'):
    # 初始化默认参数
    if initial_population is None:
        initial_population = {"healthy": 800000, "mild": 20000, "moderate": 5000, "severe": 3000}

    healthy_population = initial_population["healthy"]
    mild_cases = initial_population["mild"]
    moderate_cases = initial_population["moderate"]
    severe_cases = initial_population["severe"]

    # 定义转移矩阵
    if scenario == 'peace':
        T = np.array([
            [0.995, 0.005, 0.00, 0.00],
            [0.1, 0.7, 0.2, 0.00],
            [0.0, 0.1, 0.7, 0.2],
            [0.00, 0.00, 0.0, 1.00]
        ])
    elif scenario == 'crisis':
        T = np.array([
            [0.99, 0.01, 0.00, 0.00],
            [0.10, 0.75, 0.15, 0.0],
            [0.0, 0.10, 0.85, 0.05],
            [0.0, 0.0, 0.05, 0.95]
        ])
    else:
        raise ValueError("Invalid scenario. Choose 'peace' or 'crisis'.")

    # 初始化状态
    S = np.array([healthy_population, mild_cases, moderate_cases, severe_cases], dtype=float)

    # 医生分配
    allocation_mild, allocation_moderate, allocation_severe = 0.7, 0.2, 0.1
    doctors_mild = int(total_doctors * allocation_mild)
    doctors_moderate = int(total_doctors * allocation_moderate)
    doctors_severe = total_doctors - doctors_mild - doctors_moderate

    service_rate_mild, service_rate_moderate, service_rate_severe = 10, 5, 1
    mu_mild = doctors_mild * service_rate_mild
    mu_moderate = doctors_moderate * service_rate_moderate
    mu_severe = doctors_severe * service_rate_severe

    # 初始化记录变量
    queue_mild, queue_moderate, queue_severe = 10000, 5000, 1000
    queue_lengths_mild, queue_lengths_moderate, queue_lengths_severe = [], [], []
    cumulative_cured_mild, cumulative_cured_moderate, cumulative_cured_severe = [0], [0], [0]
    allocation_mild_list, allocation_moderate_list, allocation_severe_list = [], [], []

    # 模拟循环
    for day in range(sim_time):
        S = np.dot(S, T)
        S = np.maximum(S, 0)

        # 更新队列
        queue_mild += max(np.random.poisson(S[1] * (T[1, 2] + T[1, 3])), 0)
        queue_moderate += max(np.random.poisson(S[2] * T[2, 3]), 0)

        # 服务患者
        served_mild = min(queue_mild, mu_mild)
        served_moderate = min(queue_moderate, mu_moderate)
        served_severe = min(queue_severe, mu_severe)

        queue_mild -= served_mild
        queue_moderate -= served_moderate
        queue_severe -= served_severe

        cumulative_cured_mild.append(cumulative_cured_mild[-1] + served_mild)
        cumulative_cured_moderate.append(cumulative_cured_moderate[-1] + served_moderate)
        cumulative_cured_severe.append(cumulative_cured_severe[-1] + served_severe)

        queue_lengths_mild.append(queue_mild)
        queue_lengths_moderate.append(queue_moderate)
        queue_lengths_severe.append(queue_severe)
        allocation_mild_list.append(allocation_mild)
        allocation_moderate_list.append(allocation_moderate)
        allocation_severe_list.append(allocation_severe)

    # 图像生成
    os.makedirs(output_dir, exist_ok=True)
    output_files = []

    plt.figure(figsize=(10, 6))
    plt.plot(queue_lengths_mild, label="Mild Cases")
    plt.plot(queue_lengths_moderate, label="Moderate Cases")
    plt.plot(queue_lengths_severe, label="Severe Cases")
    plt.xlabel("Days")
    plt.ylabel("Queue Length")
    plt.title("Queue Lengths Over Time")
    plt.legend()
    queue_image_path = os.path.join(output_dir, "queue_lengths.png")
    plt.savefig(queue_image_path)
    plt.close()
    output_files.append(queue_image_path)

    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_cured_mild, label="Cured Mild Cases")
    plt.plot(cumulative_cured_moderate, label="Cured Moderate Cases")
    plt.plot(cumulative_cured_severe, label="Cured Severe Cases")
    plt.xlabel("Days")
    plt.ylabel("Cumulative Cured Cases")
    plt.title("Cumulative Cured Cases Over Time")
    plt.legend()
    cured_image_path = os.path.join(output_dir, "cumulative_cured.png")
    plt.savefig(cured_image_path)
    plt.close()
    output_files.append(cured_image_path)

    # 返回结果
    results = {
        "final_population": S.tolist(),
        "average_queue_length_mild": np.mean(queue_lengths_mild),
        "average_queue_length_moderate": np.mean(queue_lengths_moderate),
        "average_queue_length_severe": np.mean(queue_lengths_severe),
    }
    return results, output_files
