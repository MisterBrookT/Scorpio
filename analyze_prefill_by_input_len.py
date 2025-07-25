import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Load data from the JSON file
with open('benchmarks/result/pre_exp/prefill_result.json', 'r') as f:
    data = json.load(f)

# Group data by num_req
grouped_data = {}
for entry in data:
    num_req = entry['num_req']
    if num_req not in grouped_data:
        grouped_data[num_req] = []
    grouped_data[num_req].append(entry)

# Sort each group by input_len
for num_req, entries in grouped_data.items():
    grouped_data[num_req] = sorted(entries, key=lambda x: x['input_len'])

# Linear function for fitting
def linear_func(x, a, b):
    return a * x + b

# Create a figure
plt.figure(figsize=(12, 8))

# Plot time vs input_len for each num_req
markers = ['o', 's', '^', 'D', 'x']
colors = ['blue', 'green', 'red', 'purple', 'orange']

for i, (num_req, entries) in enumerate(sorted(grouped_data.items())):
    input_lens = np.array([entry['input_len'] for entry in entries])
    times = np.array([entry['time'] for entry in entries])
    
    # Calculate linear fit
    popt, _ = curve_fit(linear_func, input_lens, times)
    a, b = popt
    
    marker = markers[i % len(markers)]
    color = colors[i % len(colors)]
    plt.plot(input_lens, times, marker=marker, color=color, linestyle='-', 
             label=f'num_req={num_req}: {a:.6f}x + {b:.4f}')

# Add labels and title
plt.xlabel('Input Length', fontsize=12)
plt.ylabel('Time (seconds)', fontsize=12)
plt.title('Execution Time vs Input Length for Different Number of Requests', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)

# Save the plot
plt.savefig('prefill_time_vs_input_len.png', dpi=300, bbox_inches='tight')

# Create a separate plot to show scaling behavior
plt.figure(figsize=(12, 8))

# For each num_req, plot the scaling behavior
for i, (num_req, entries) in enumerate(sorted(grouped_data.items())):
    # Normalize input_len and time
    input_lens = np.array([entry['input_len'] for entry in entries])
    times = np.array([entry['time'] for entry in entries])
    
    # Compute relative scaling (time_i / time_0) vs (input_len_i / input_len_0)
    relative_input = input_lens / input_lens[0]
    relative_time = times / times[0]
    
    color = colors[i % len(colors)]
    marker = markers[i % len(markers)]
    plt.plot(relative_input, relative_time, marker=marker, color=color, 
             label=f'num_req={num_req}')

# Add reference lines for linear and quadratic scaling
x_ref = np.linspace(1, max([max(np.array([entry['input_len'] for entry in entries])/entries[0]['input_len']) 
                          for entries in grouped_data.values()]), 100)
plt.plot(x_ref, x_ref, 'k--', label='Linear (O(n))')
plt.plot(x_ref, x_ref**2, 'k-.', label='Quadratic (O(n²))')

plt.xlabel('Relative Input Length (normalized)', fontsize=12)
plt.ylabel('Relative Time (normalized)', fontsize=12)
plt.title('Scaling Behavior of Execution Time with Input Length', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)
# Using linear scale for x-axis
plt.yscale('log', base=2)

plt.savefig('prefill_input_scaling_behavior.png', dpi=300, bbox_inches='tight')
plt.close()

print("Plots saved as prefill_time_vs_input_len.png and prefill_input_scaling_behavior.png") 