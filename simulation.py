# simulation.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import os

def run_simulation():
    # 模拟参数

    # 初始心理健康状态分布
    healthy_population = 800000  # 健康个体数量
    mild_cases = 20000           # 轻度病例数量
    moderate_cases = 5000        # 中度病例数量
    severe_cases = 3000          # 重度病例数量

    # 场景选择
    scenario = 'peace'  # 选择 'peace'（稳定）或 'crisis'（危机）

    # 初始资源分配系数（确保总和为 1.0）
    allocation_mild = 0.7        # 轻度病例的资源分配比例
    allocation_moderate = 0.2    # 中度病例的资源分配比例
    allocation_severe = 0.1      # 重度病例的资源分配比例

    # 心理健康专业人员总数
    total_doctors = 1000  # 医生总数

    # 每个医生的服务率（每天治愈的患者数量）
    service_rate_mild = 10       # 每个医生每天治愈的轻度病例数量
    service_rate_moderate = 5    # 每个医生每天治愈的中度病例数量
    service_rate_severe = 1      # 每个医生每天治愈的重度病例数量

    # 模拟时间（以天为单位）
    SIM_TIME = 365  # 模拟一年

    # 根据场景调整转移矩阵
    if scenario == 'peace':
        T = np.array([
            [0.995, 0.005, 0.00, 0.00],  # 健康个体的转移概率
            [0.1, 0.7, 0.2, 0.00],       # 轻度病例的转移概率
            [0.0, 0.1, 0.7, 0.2],        # 中度病例的转移概率
            [0.00, 0.00, 0.0, 1.00]      # 重度病例的转移概率
        ])
    elif scenario == 'crisis':
        T = np.array([
            [0.99, 0.01, 0.00, 0.00],  # 健康个体的转移概率
            [0.10, 0.75, 0.15, 0.0],   # 轻度病例的转移概率
            [0.0, 0.10, 0.85, 0.05],   # 中度病例的转移概率
            [0.0, 0.0, 0.05, 0.95]     # 重度病例的转移概率
        ])

    # 初始化状态向量 S = [健康, 轻度, 中度, 重度]
    S = np.array([healthy_population, mild_cases, moderate_cases, severe_cases], dtype=float)

    # 计算初始医生分配
    doctors_mild = int(total_doctors * allocation_mild)           # 分配给轻度病例的医生数量
    doctors_moderate = int(total_doctors * allocation_moderate)   # 分配给中度病例的医生数量
    doctors_severe = total_doctors - doctors_mild - doctors_moderate  # 分配给重度病例的医生数量

    # 计算每组的服务率
    mu_mild = doctors_mild * service_rate_mild           # 轻度病例的每日服务能力
    mu_moderate = doctors_moderate * service_rate_moderate   # 中度病例的每日服务能力
    mu_severe = doctors_severe * service_rate_severe       # 重度病例的每日服务能力

    # 初始化资源分配比例列表
    allocation_mild_list = []
    allocation_moderate_list = []
    allocation_severe_list = []

    # 初始化每组的等待队列
    queue_mild = 10000
    queue_moderate = 5000
    queue_severe = 1000

    # 初始化记录队列长度的列表
    queue_lengths_mild = []
    queue_lengths_moderate = []
    queue_lengths_severe = []

    # 初始化跟踪活跃病例数量的列表
    active_mild = [mild_cases]
    active_moderate = [moderate_cases]
    active_severe = [severe_cases]

    # 初始化记录每日净新增病例的列表
    daily_net_new_mild = []
    daily_net_new_moderate = []
    daily_net_new_severe = []

    # 初始化记录累计治愈病例的列表
    cumulative_cured_mild = [0]
    cumulative_cured_moderate = [0]
    cumulative_cured_severe = [0]

    # 模拟循环
    for day in range(SIM_TIME):
        # 更新心理健康状态（马尔可夫链中的一步）
        S = np.dot(S, T)

        # 确保 S 中没有负值
        S = np.maximum(S, 0)

        # 计算总人口
        total_population = S.sum()

        if total_population == 0:
            new_mild = 0
            new_moderate = 0
            new_severe = 0
        else:
            # 计算每个状态的比例
            fraction_mild = S[1] / total_population       # 轻度病例的比例
            fraction_moderate = S[2] / total_population   # 中度病例的比例
            fraction_severe = S[3] / total_population     # 重度病例的比例

            # 计算每组的新病例
            new_mild = S[0] * (T[0,1] + T[0,2] + T[0,3])  # 健康转为轻度/中度/重度
            new_moderate = S[1] * (T[1,2] + T[1,3])       # 轻度转为中度/重度
            new_severe = S[2] * T[2,3]                   # 中度转为重度

        # 确保到达率非负
        lambda_mild = max(new_mild, 0)
        lambda_moderate = max(new_moderate, 0)
        lambda_severe = max(new_severe, 0)

        try:
            num_arrivals_mild = np.random.poisson(lambda_mild)
            num_arrivals_moderate = np.random.poisson(lambda_moderate)
            num_arrivals_severe = np.random.poisson(lambda_severe)
        except ValueError as e:
            num_arrivals_mild = 0
            num_arrivals_moderate = 0
            num_arrivals_severe = 0

        # 更新等待队列
        queue_mild += num_arrivals_mild
        queue_moderate += num_arrivals_moderate
        queue_severe += num_arrivals_severe

        # 检查是否为周末（假设第 0 天是星期一）
        if day % 7 in [5, 6]:  # 5 和 6 对应星期六和星期日
            patients_served_mild = 0
            patients_served_moderate = 0
            patients_served_severe = 0
        else:
            # 根据服务能力计算每组服务的患者
            patients_served_mild = min(queue_mild, mu_mild)
            patients_served_moderate = min(queue_moderate, mu_moderate)
            patients_served_severe = min(queue_severe, mu_severe)

        # 更新队列长度
        queue_mild -= patients_served_mild
        queue_moderate -= patients_served_moderate
        queue_severe -= patients_served_severe

        # 计算每日净新增病例（到达 - 治愈）
        net_new_mild = num_arrivals_mild - patients_served_mild
        net_new_moderate = num_arrivals_moderate - patients_served_moderate
        net_new_severe = num_arrivals_severe - patients_served_severe

        # 记录每日净新增病例
        daily_net_new_mild.append(net_new_mild)
        daily_net_new_moderate.append(net_new_moderate)
        daily_net_new_severe.append(net_new_severe)

        # 更新治愈和恢复计数
        recovered_mild = patients_served_mild
        S[0] += recovered_mild      
        S[1] -= recovered_mild      

        recovered_moderate = patients_served_moderate
        S[0] += recovered_moderate  
        S[2] -= recovered_moderate  

        recovered_severe = patients_served_severe
        S[0] += recovered_severe    
        S[3] -= recovered_severe    

        cumulative_cured_mild.append(cumulative_cured_mild[-1] + recovered_mild)
        cumulative_cured_moderate.append(cumulative_cured_moderate[-1] + recovered_moderate)
        cumulative_cured_severe.append(cumulative_cured_severe[-1] + recovered_severe)

        queue_lengths_mild.append(queue_mild)
        queue_lengths_moderate.append(queue_moderate)
        queue_lengths_severe.append(queue_severe)

        allocation_mild_list.append(allocation_mild)
        allocation_moderate_list.append(allocation_moderate)
        allocation_severe_list.append(allocation_severe)

        active_mild.append(S[1])
        active_moderate.append(S[2])
        active_severe.append(S[3])

        if day % 30 == 0 and day > 0:
            avg_wait_mild = np.mean(queue_lengths_mild[-30:]) / mu_mild if mu_mild > 0 else 0
            avg_wait_moderate = np.mean(queue_lengths_moderate[-30:]) / mu_moderate if mu_moderate > 0 else 0
            avg_wait_severe = np.mean(queue_lengths_severe[-30:]) / mu_severe if mu_severe > 0 else 0

            adjust = False
            delta = 0.05  

            if avg_wait_severe > 7:
                allocation_severe = min(allocation_severe + delta, 1.0)
                allocation_mild = max(allocation_mild - delta / 2, 0)
                allocation_moderate = max(allocation_moderate - delta / 2, 0)
                adjust = True

            if avg_wait_moderate > 14:
                allocation_moderate = min(allocation_moderate + delta, 1.0)
                allocation_mild = max(allocation_mild - delta / 2, 0)
                allocation_severe = max(allocation_severe - delta / 2, 0)
                adjust = True

            if avg_wait_mild > 20:
                allocation_mild = min(allocation_mild + delta, 1.0)
                allocation_moderate = max(allocation_moderate - delta / 2, 0)
                allocation_severe = max(allocation_severe - delta / 2, 0)
                adjust = True

            if adjust:
                total_allocation = allocation_mild + allocation_moderate + allocation_severe
                allocation_mild /= total_allocation
                allocation_moderate /= total_allocation
                allocation_severe /= total_allocation

                doctors_mild = int(total_doctors * allocation_mild)
                doctors_moderate = int(total_doctors * allocation_moderate)
                doctors_severe = total_doctors - doctors_mild - doctors_moderate

                mu_mild = doctors_mild * service_rate_mild
                mu_moderate = doctors_moderate * service_rate_moderate
                mu_severe = doctors_severe * service_rate_severe

    severe_cases_after_one_year = S[3]
    mild_cases_after_one_year = S[1]
    moderate_cases_after_one_year = S[2]

    avg_waiting_time_mild = np.mean(queue_lengths_mild) / mu_mild if mu_mild > 0 else 0
    avg_waiting_time_moderate = np.mean(queue_lengths_moderate) / mu_moderate if mu_moderate > 0 else 0
    avg_waiting_time_severe = np.mean(queue_lengths_severe) / mu_severe if mu_severe > 0 else 0

    avg_queue_length_mild = np.mean(queue_lengths_mild)
    avg_queue_length_moderate = np.mean(queue_lengths_moderate)
    avg_queue_length_severe = np.mean(queue_lengths_severe)

    # 保存图像到 static/images/
    if not os.path.exists('static/images'):
        os.makedirs('static/images')

    plt.figure(figsize=(18, 6))
    plt.plot(range(len(queue_lengths_mild)), queue_lengths_mild, label='轻度病例队列长度')
    plt.plot(range(len(queue_lengths_moderate)), queue_lengths_moderate, label='中度病例队列长度')
    plt.plot(range(len(queue_lengths_severe)), queue_lengths_severe, label='重度病例队列长度')
    plt.xlabel('天数')
    plt.ylabel('队列长度')
    plt.title('队列长度随时间的变化')
    plt.legend()
    plt.tight_layout()
    plt.savefig('static/images/queue_lengths.png')
    plt.close()

    plt.figure(figsize=(18, 6))
    plt.plot(range(len(allocation_mild_list)), allocation_mild_list, label='轻度病例资源分配比例')
    plt.plot(range(len(allocation_moderate_list)), allocation_moderate_list, label='中度病例资源分配比例')
    plt.plot(range(len(allocation_severe_list)), allocation_severe_list, label='重度病例资源分配比例')
    plt.xlabel('天数')
    plt.ylabel('资源分配系数')
    plt.title('资源分配比例随时间的变化')
    plt.legend()
    plt.tight_layout()
    plt.savefig('static/images/resource_allocations.png')
    plt.close()

    plt.figure(figsize=(18, 6))
    plt.plot(range(len(daily_net_new_mild)), daily_net_new_mild, label='轻度病例队列每日净增加')
    plt.plot(range(len(daily_net_new_moderate)), daily_net_new_moderate, label='中度病例队列每日净增加')
    plt.plot(range(len(daily_net_new_severe)), daily_net_new_severe, label='重度病例队列每日净增加')
    plt.xlabel('天数')
    plt.ylabel('患者净增加数量')
    plt.title('各队列每日患者净增加数量')
    plt.legend()
    plt.tight_layout()
    plt.savefig('static/images/daily_net_new_cases.png')
    plt.close()

    plt.figure(figsize=(18, 6))
    plt.plot(range(len(cumulative_cured_mild)), cumulative_cured_mild, label='轻度病例累计治愈人数')
    plt.plot(range(len(cumulative_cured_moderate)), cumulative_cured_moderate, label='中度病例累计治愈人数')
    plt.plot(range(len(cumulative_cured_severe)), cumulative_cured_severe, label='重度病例累计治愈人数')
    plt.xlabel('天数')
    plt.ylabel('累计治愈人数')
    plt.title('各队列累计治愈人数')
    plt.legend()
    plt.tight_layout()
    plt.savefig('static/images/cumulative_cured.png')
    plt.close()

    # 准备输出文本
    outputs = {}
    outputs['mild_cases_after_one_year'] = int(mild_cases_after_one_year)
    outputs['moderate_cases_after_one_year'] = int(moderate_cases_after_one_year)
    outputs['severe_cases_after_one_year'] = int(severe_cases_after_one_year)
    outputs['avg_waiting_time_mild'] = avg_waiting_time_mild
    outputs['avg_waiting_time_moderate'] = avg_waiting_time_moderate
    outputs['avg_waiting_time_severe'] = avg_waiting_time_severe
    outputs['avg_queue_length_mild'] = avg_queue_length_mild
    outputs['avg_queue_length_moderate'] = avg_queue_length_moderate
    outputs['avg_queue_length_severe'] = avg_queue_length_severe

    monthly_allocations = pd.DataFrame({
        '天数': range(len(allocation_mild_list)),
        '轻度病例': allocation_mild_list,
        '中度病例': allocation_moderate_list,
        '重度病例': allocation_severe_list
    })
    monthly_allocations = monthly_allocations[monthly_allocations['天数'] % 30 == 0]
    outputs['monthly_allocations'] = monthly_allocations.to_html(index=False)

    return outputs
