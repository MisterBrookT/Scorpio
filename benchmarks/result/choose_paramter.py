import pandas as pd

# Read the CSV file
df = pd.read_csv('draw/llama3_8b_lmsys/choose_parameter/choose_parameter.csv')

# Group by policy and sum the good_completion column
policy_sums = df.groupby('policy')['goodput'].sum().reset_index()

# Sort by good_completion in descending order
policy_sums = policy_sums.sort_values('goodput', ascending=False)

# Print the results
print("\nSum of good completions by policy (sorted from highest to lowest):")
print("=" * 50)
for _, row in policy_sums.iterrows():
    print(f"{row['policy']}: {row['goodput']}")
