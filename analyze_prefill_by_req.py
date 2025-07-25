import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Load data from the JSON file
with open('benchmarks/result/pre_exp/prefill_result.json', 'r') as f:
    data = json.load(f)

# Group data by input_len
grouped_data = {}
for entry in data:
    input_len = entry['input_len']
    if input_len not in grouped_data:
        grouped_data[input_len] = []
    grouped_data[input_len].append(entry)

# Sort each group by num_req
for input_len, entries in grouped_data.items():
    grouped_data[input_len] = sorted(entries, key=lambda x: x['num_req'])

# Linear function for fitting
def linear_func(x, a, b):
    return a * x + b

# Create a figure
plt.figure(figsize=(12, 8))

# Plot time vs num_req for each input_len
markers = ['o', 's', '^', 'D', 'x']
colors = ['blue', 'green', 'red', 'purple', 'orange']

for i, (input_len, entries) in enumerate(sorted(grouped_data.items())):
    num_reqs = np.array([entry['num_req'] for entry in entries])
    times = np.array([entry['time'] for entry in entries])
    
    # Calculate linear fit
    popt, _ = curve_fit(linear_func, num_reqs, times)
    a, b = popt
    
    marker = markers[i % len(markers)]
    color = colors[i % len(colors)]
    plt.plot(num_reqs, times, marker=marker, color=color, linestyle='-', 
             label=f'input_len={input_len}: {a:.6f}x + {b:.4f}')

# Add labels and title
plt.xlabel('Number of Requests', fontsize=12)
plt.ylabel('Time (seconds)', fontsize=12)
plt.title('Execution Time vs Number of Requests for Different Input Lengths', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)

# Save the plot
plt.savefig('prefill_time_vs_num_req.png', dpi=300, bbox_inches='tight')

# Create a separate plot to show scaling behavior
plt.figure(figsize=(12, 8))

# For each input_len, plot the scaling behavior
for i, (input_len, entries) in enumerate(sorted(grouped_data.items())):
    # Normalize num_req and time
    num_reqs = np.array([entry['num_req'] for entry in entries])
    times = np.array([entry['time'] for entry in entries])
    
    # Compute relative scaling (time_i / time_0) vs (num_req_i / num_req_0)
    relative_num_req = num_reqs / num_reqs[0]
    relative_time = times / times[0]
    
    color = colors[i % len(colors)]
    marker = markers[i % len(markers)]
    plt.plot(relative_num_req, relative_time, marker=marker, color=color, 
             label=f'input_len={input_len}')

# Add reference lines for linear and quadratic scaling
x_ref = np.linspace(1, max([max(np.array([entry['num_req'] for entry in entries])/entries[0]['num_req']) 
                          for entries in grouped_data.values()]), 100)
plt.plot(x_ref, x_ref, 'k--', label='Linear (O(n))')
plt.plot(x_ref, x_ref**2, 'k-.', label='Quadratic (O(n²))')

plt.xlabel('Relative Number of Requests (normalized)', fontsize=12)
plt.ylabel('Relative Time (normalized)', fontsize=12)
plt.title('Scaling Behavior of Execution Time with Number of Requests', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)
# Removed log scale for x-axis
# plt.xscale('log', base=2)
plt.yscale('log', base=2)

plt.savefig('prefill_req_scaling_behavior.png', dpi=300, bbox_inches='tight')
plt.close()

print("Plots saved as prefill_time_vs_num_req.png and prefill_req_scaling_behavior.png") 