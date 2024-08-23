import pandas as pd

df = pd.read_csv('fer2013.csv')

# Split the data into training and testing datasets
df_training = df[df['Usage'] == 'Training']
df_testing = df[df['Usage'] == 'PrivateTest']

# Save the split datasets into separate CSV files
df_training.to_csv('training_data.csv', index=False)
df_testing.to_csv('testing_data.csv', index=False)

print("Data split into training_data.csv and testing_data.csv successfully!")
