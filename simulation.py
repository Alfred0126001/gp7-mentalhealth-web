# simulation.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def run_simulation(healthy_population, mild_cases, moderate_cases, severe_cases, scenario):
    # Simulation parameters

    # Initial mental health state distribution
    healthy_population = healthy_population    # Number of healthy individuals
    mild_cases = mild_cases                    # Number of mild cases
    moderate_cases = moderate_cases            # Number of moderate cases
    severe_cases = severe_cases                # Number of severe cases

    # Scenario selection
    scenario = scenario  # 'peace' (stable) or 'crisis'

    # Initial resource allocation coefficients (ensure they sum to 1.0)
    allocation_mild = 0.6        # Resource allocation ratio for mild cases
    allocation_moderate = 0.2    # Resource allocation ratio for moderate cases
    allocation_severe = 0.2      # Resource allocation ratio for severe cases

    # Total number of mental health professionals
    total_doctors = 1200  # Total number of doctors

    # Service rate per doctor (number of patients cured per day)
    service_rate_mild = 8       # Number of mild cases cured per doctor per day
    service_rate_moderate = 5    # Number of moderate cases cured per doctor per day
    service_rate_severe = 1      # Number of severe cases cured per doctor per day

    # Simulation time (in days)
    SIM_TIME = 365  # Simulate one year

    # Adjust transition matrix based on scenario
    if scenario == 'peace':
        T = np.array([
            [0.996, 0.004, 0.00, 0.00],  # Transition probabilities for healthy individuals
            [0.10, 0.70, 0.20, 0.00],       # Transition probabilities for mild cases
            [0.0, 0.1, 0.80, 0.10],        # Transition probabilities for moderate cases
            [0.00, 0.00, 0.20, 0.80]      # Transition probabilities for severe cases
        ])
    elif scenario == 'crisis':
        T = np.array([
            [0.995, 0.005, 0.00, 0.00],    # Transition probabilities for healthy individuals
            [0.10, 0.7, 0.2, 0.0],     # Transition probabilities for mild cases
            [0.0, 0.10, 0.7, 0.20],     # Transition probabilities for moderate cases
            [0.0, 0.0, 0.10, 0.90]       # Transition probabilities for severe cases
        ])
    else:
        # Default to 'peace' scenario
        T = np.array([
            [0.996, 0.004, 0.00, 0.00],  # Transition probabilities for healthy individuals
            [0.10, 0.70, 0.20, 0.00],       # Transition probabilities for mild cases
            [0.0, 0.1, 0.80, 0.10],        # Transition probabilities for moderate cases
            [0.00, 0.00, 0.20, 0.80]      # Transition probabilities for severe cases
        ])

    # Initialize state vector S = [Healthy, Mild, Moderate, Severe]
    S = np.array([healthy_population, mild_cases, moderate_cases, severe_cases], dtype=float)

    # Subsequent code remains unchanged; just ensure all required variables are defined within the function

    # Calculate initial doctor allocation
    doctors_mild = int(total_doctors * allocation_mild)               # Number of doctors allocated to mild cases
    doctors_moderate = int(total_doctors * allocation_moderate)       # Number of doctors allocated to moderate cases
    doctors_severe = total_doctors - doctors_mild - doctors_moderate  # Number of doctors allocated to severe cases

    # Calculate service capacity for each group
    mu_mild = doctors_mild * service_rate_mild               # Daily service capacity for mild cases
    mu_moderate = doctors_moderate * service_rate_moderate   # Daily service capacity for moderate cases
    mu_severe = doctors_severe * service_rate_severe         # Daily service capacity for severe cases

    # Initialize resource allocation ratio lists
    allocation_mild_list = []
    allocation_moderate_list = []
    allocation_severe_list = []

    # Initialize waiting queues for each group
    queue_mild = 10000
    queue_moderate = 5000
    queue_severe = 3000

    # Initialize lists to record queue lengths
    queue_lengths_mild = []
    queue_lengths_moderate = []
    queue_lengths_severe = []

    # Initialize lists to track active case numbers
    active_mild = [mild_cases]
    active_moderate = [moderate_cases]
    active_severe = [severe_cases]

    # Initialize lists to record daily net new cases
    daily_net_new_mild = []
    daily_net_new_moderate = []
    daily_net_new_severe = []

    # Initialize lists to record cumulative cured cases
    cumulative_cured_mild = [0]
    cumulative_cured_moderate = [0]
    cumulative_cured_severe = [0]

# Simulation loop
    for day in range(SIM_TIME):
        # Store the current state before updating
        S_old = S.copy()
    
        # Compute the transitions
        transitions = np.zeros_like(T)
        for i in range(4):
            for j in range(4):
                transitions[i, j] = S_old[i] * T[i, j]
    
        # Update the state vector
        S = np.sum(transitions, axis=0)
    
        # Ensure no negative values in S
        S = np.maximum(S, 0)
    
        # Compute net new cases for each group
        net_new_cases = np.sum(transitions, axis=0) - np.sum(transitions, axis=1)
        net_new_mild = net_new_cases[1]
        net_new_moderate = net_new_cases[2]
        net_new_severe = net_new_cases[3]
    
        # Compute arrival rates for queues
        lambda_mild = max(net_new_mild, 0)
        lambda_moderate = max(net_new_moderate, 0)
        lambda_severe = max(net_new_severe, 0)
    
        # Generate the number of new arrivals using Poisson distribution
        try:
            num_arrivals_mild = np.random.poisson(lambda_mild)
            num_arrivals_moderate = np.random.poisson(lambda_moderate)
            num_arrivals_severe = np.random.poisson(lambda_severe)
        except ValueError as e:
            num_arrivals_mild = 0
            num_arrivals_moderate = 0
            num_arrivals_severe = 0

        # Update waiting queues
        queue_mild += num_arrivals_mild
        queue_moderate += num_arrivals_moderate
        queue_severe += num_arrivals_severe

        # Check if it's the weekend (assuming day 0 is Monday)
        if day % 7 in [5, 6]:  # 5 and 6 correspond to Saturday and Sunday
            patients_served_mild = 0
            patients_served_moderate = 0
            patients_served_severe = 0
        else:
            # Calculate the number of patients served based on service capacity
            patients_served_mild = min(queue_mild, mu_mild)
            patients_served_moderate = min(queue_moderate, mu_moderate)
            patients_served_severe = min(queue_severe, mu_severe)

        # Update queue lengths
        queue_mild -= patients_served_mild
        queue_moderate -= patients_served_moderate
        queue_severe -= patients_served_severe


        # Compute daily net new cases (arrivals - treated)
        net_queue_mild = num_arrivals_mild - patients_served_mild
        net_queue_moderate = num_arrivals_moderate - patients_served_moderate
        net_queue_severe = num_arrivals_severe - patients_served_severe
    
    
    
        # Record daily net new cases
        daily_net_new_mild.append(net_queue_mild)
        daily_net_new_moderate.append(net_queue_moderate)
        daily_net_new_severe.append(net_queue_severe)

        # Update recovery counts
        recovered_mild = patients_served_mild
        S[0] += recovered_mild      # Increase healthy individuals
        S[1] -= recovered_mild      # Decrease mild cases

        recovered_moderate = patients_served_moderate
        S[0] += recovered_moderate  # Increase healthy individuals
        S[2] -= recovered_moderate  # Decrease moderate cases

        recovered_severe = patients_served_severe
        S[0] += recovered_severe    # Increase healthy individuals
        S[3] -= recovered_severe    # Decrease severe cases

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

            if avg_wait_severe > 3:
                allocation_severe = min(allocation_severe + delta, 1.0)
                allocation_mild = max(allocation_mild - delta / 2, 0)
                allocation_moderate = max(allocation_moderate - delta / 2, 0)
                adjust = True

            if avg_wait_moderate > 7:
                allocation_moderate = min(allocation_moderate + delta, 1.0)
                allocation_mild = max(allocation_mild - delta / 2, 0)
                allocation_severe = max(allocation_severe - delta / 2, 0)
                adjust = True

            if avg_wait_mild > 14:
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

    # Save images to static/images/
    if not os.path.exists('static/images'):
        os.makedirs('static/images')

    plt.figure(figsize=(18, 6))
    plt.plot(range(len(queue_lengths_mild)), queue_lengths_mild, label='Queue Length for Mild Cases')
    plt.plot(range(len(queue_lengths_moderate)), queue_lengths_moderate, label='Queue Length for Moderate Cases')
    plt.plot(range(len(queue_lengths_severe)), queue_lengths_severe, label='Queue Length for Severe Cases')
    plt.xlabel('Days')
    plt.ylabel('Queue Length')
    plt.title('Queue Length Changes Over Time')
    plt.legend()
    plt.tight_layout()
    plt.savefig('static/images/queue_lengths.png')
    plt.close()

    plt.figure(figsize=(18, 6))
    plt.plot(range(len(allocation_mild_list)), allocation_mild_list, label='Resource Allocation Ratio for Mild Cases')
    plt.plot(range(len(allocation_moderate_list)), allocation_moderate_list, label='Resource Allocation Ratio for Moderate Cases')
    plt.plot(range(len(allocation_severe_list)), allocation_severe_list, label='Resource Allocation Ratio for Severe Cases')
    plt.xlabel('Days')
    plt.ylabel('Resource Allocation Coefficient')
    plt.title('Resource Allocation Ratios Over Time')
    plt.legend()
    plt.tight_layout()
    plt.savefig('static/images/resource_allocations.png')
    plt.close()

    plt.figure(figsize=(18, 6))
    plt.plot(range(len(daily_net_new_mild)), daily_net_new_mild, label='Daily Net Increase for Mild Cases Queue')
    plt.plot(range(len(daily_net_new_moderate)), daily_net_new_moderate, label='Daily Net Increase for Moderate Cases Queue')
    plt.plot(range(len(daily_net_new_severe)), daily_net_new_severe, label='Daily Net Increase for Severe Cases Queue')
    plt.xlabel('Days')
    plt.ylabel('Net Increase in Patients')
    plt.title('Daily Net Increase in Patients for Each Queue')
    plt.legend()
    plt.tight_layout()
    plt.savefig('static/images/daily_net_new_cases.png')
    plt.close()

    plt.figure(figsize=(18, 6))
    plt.plot(range(len(cumulative_cured_mild)), cumulative_cured_mild, label='Cumulative Cured Mild Cases')
    plt.plot(range(len(cumulative_cured_moderate)), cumulative_cured_moderate, label='Cumulative Cured Moderate Cases')
    plt.plot(range(len(cumulative_cured_severe)), cumulative_cured_severe, label='Cumulative Cured Severe Cases')
    plt.xlabel('Days')
    plt.ylabel('Cumulative Cured Cases')
    plt.title('Cumulative Cured Cases for Each Queue')
    plt.legend()
    plt.tight_layout()
    plt.savefig('static/images/cumulative_cured.png')
    plt.close()

    # Prepare output text
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
        'Day': range(len(allocation_mild_list)),
        'Mild Cases': allocation_mild_list,
        'Moderate Cases': allocation_moderate_list,
        'Severe Cases': allocation_severe_list
    })
    monthly_allocations = monthly_allocations[monthly_allocations['Day'] % 30 == 0]
    outputs['monthly_allocations'] = monthly_allocations.to_html(index=False)

    return outputs
