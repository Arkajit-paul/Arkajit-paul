#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Create a DataFrame from the given data
data = {
    'Years Experience': [1.1, 1.3, 1.5, 2.0, 2.2, 2.9, 3.0, 3.2, 3.2, 3.7, 3.9, 4.0],
    'Salary': [39343, 46205, 37731, 43525, 39891, 56642, 60150, 54445, 64445, 57189, 63218, 55794]
}
df = pd.DataFrame(data)

# Save DataFrame to a CSV file
df.to_csv('salary_experience.csv', index=False)

# Plot scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df['Years Experience'], df['Salary'], color='blue', label='Data Points')

# Perform linear regression
regression = LinearRegression()
regression.fit(df[['Years Experience']], df['Salary'])

# Predict salary for a new candidate with 5 years of experience
new_experience = [[5.0]]
predicted_salary = regression.predict(new_experience)
print("Predicted salary for 5 years of experience:", predicted_salary[0])

# Plot regression line
plt.plot(df['Years Experience'], regression.predict(df[['Years Experience']]), color='red', label='Regression Line')

# Add labels and title
plt.xlabel('Years Experience')
plt.ylabel('Salary')
plt.title('Salary vs Years of Experience')
plt.legend()

# Show plot
plt.grid(True)
plt.show()
