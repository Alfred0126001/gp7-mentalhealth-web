import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_simulation(scenario='peace', total_doctors=1000, sim_time=365, 
                   initial_population=None, allocation_coefficients=None):
    """
    运行模拟程序，返回模拟结果和可视化。

    参数:
        scenario (str): 场景 ('peace' 或 'crisis')。
        total_doctors (int): 总医生人数。
        sim_time (int): 模拟天数。
        initial_population (dict): 初始人口分布字典，默认:
            {
                "healthy": 800000,
                "mild": 20000,
                "moderate": 5000,
                "severe": 3000
            }
        allocation_coefficients (dict): 初始资源分配系数，默认:
            {
                "mild": 0.7,
                "moderate": 0.2,
                "severe": 0.1
            }

    返回:
        dict: 包括模拟结果和统计信息。
    """
    # 默认初始参数
    if initial_population is None:
        initial_population = {"healthy": 800000, "mild": 20000, "moderate": 5000, "severe": 3000}
    if allocation_coefficients is None:
        allocation_coefficients = {"mild": 0.7, "moderate": 0.2, "severe": 0.1}
    
    # 提取初始人口分布
    healthy_population = initial_population["healthy"]
    mild_cases = initial_population["mild"]
    moderate_cases = initial_population["moderate"]
    severe_cases = initial_population["severe"]

    # 初始化转移矩阵
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

    # 初始化状态向量
    S = np.array([healthy_population, mild_cases, moderate_cases, severe_cases], dtype=float)

    # 初始化医生分配和服务速率
    allocation_mild = allocation_coefficients["mild"]
    allocation_moderate = allocation_coefficients["moderate"]
    allocation_severe = allocation_coefficients["severe"]

    doctors_mild = int(total_doctors * allocation_mild)
    doctors_moderate = int(total_doctors * allocation_moderate)
    doctors_severe = total_doctors - doctors_mild - doctors_moderate

    service_rate_mild = 10
    service_rate_moderate = 5
    service_rate_severe = 1

    mu_mild = doctors_mild * service_rate_mild
    mu_moderate = doctors_moderate * service_rate_moderate
    mu_severe = doctors_severe * service_rate_severe

    # 初始化队列
    queue_mild, queue_moderate, queue_severe = 10000, 5000, 1000

    # 记录数据
    results = {
        "queue_lengths_mild": [],
        "queue_lengths_moderate": [],
        "queue_lengths_severe": [],
        "daily_net_new_mild": [],
        "daily_net_new_moderate": [],
        "daily_net_new_severe": [],
        "allocation_mild": [],
        "allocation_moderate": [],
        "allocation_severe": []
    }

    # 模拟循环
    for day in range(sim_time):
        S = np.dot(S, T)
        S = np.maximum(S, 0)

        # 新病例数
        new_mild = S[0] * (T[0, 1] + T[0, 2] + T[0, 3])
        new_moderate = S[1] * (T[1, 2] + T[1, 3])
        new_severe = S[2] * T[2, 3]

        # 排队情况
        queue_mild += max(np.random.poisson(new_mild), 0)
        queue_moderate += max(np.random.poisson(new_moderate), 0)
        queue_severe += max(np.random.poisson(new_severe), 0)

        # 服务患者
        patients_served_mild = min(queue_mild, mu_mild)
        patients_served_moderate = min(queue_moderate, mu_moderate)
        patients_served_severe = min(queue_severe, mu_severe)

        queue_mild -= patients_served_mild
        queue_moderate -= patients_served_moderate
        queue_severe -= patients_served_severe

        # 更新记录
        results["queue_lengths_mild"].append(queue_mild)
        results["queue_lengths_moderate"].append(queue_moderate)
        results["queue_lengths_severe"].append(queue_severe)
        results["allocation_mild"].append(allocation_mild)
        results["allocation_moderate"].append(allocation_moderate)
        results["allocation_severe"].append(allocation_severe)

    return results