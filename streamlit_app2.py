import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Sample data
data = {
    "Revenue_growth": [-23, -20, -17, -14, -11, -8, -5, -2, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46],
    "Count": [0, 1, 0, 1, 2, 1, 3, 3, 9, 7, 1, 2, 4, 0, 0, 0, 0, 2, 0, 1, 0, 0, 1, 0]
}

# New value pairs data
value_pairs = {
    "Min": 22.2,
    "Max": 22.96,
    "Analyst Prediction": 22.31804262
}

# Create a DataFrame
df = pd.DataFrame(data)

# Create a bar chart using Matplotlib
fig, ax = plt.subplots()
ax.bar(df["Revenue_growth"], df["Count"], label="Original Data")

# Plot the new value pairs as vertical lines
for label, value in value_pairs.items():
    line_color = 'yellow' if label == "Analyst Prediction" else '#a0a0a0'
    line_width = 1 if label == "Analyst Prediction" else 0.5
    ax.axvline(x=value, color=line_color, linestyle='-', linewidth=line_width, label=f"{label}: {value}")

# Customize the chart
ax.set_xlabel("Revenue Growth")
ax.set_ylabel("Count")
ax.set_title("Revenue Growth Distribution")

# Add a legend
ax.legend()

# Display the chart using Streamlit
st.pyplot(fig)
