import streamlit as st
import requests
import json
import pandas as pd
import datetime
import time

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

st.set_page_config(layout="wide")

# @st.cache_data
def definedCombination():
    if 'DCloading' in st.session_state:
      DCloading.warning("Running Simulations")    

    payload = json.dumps({
       "request_data": {
          "inputs": {
            "SECTOR": SectorInput,
            "REGION": RegionInput
          }
       },
        "request_meta": {
          "call_purpose": "FE",
          "source_system": "SPARK",
          "service_category": ""
        }
      })

    # st.json(payload)

    url = "https://excel.uat.jp.coherent.global/clsa/api/v3/folders/Aggregate%20Models/services/Output%20Analysis%20-%20by%20Sector%20&%20Region/execute"
    headers = {
       'Content-Type': 'application/json',
       'x-tenant-name': 'clsa',
       'x-synthetic-key': '3ca9da18-31fa-4a82-a9ba-44130dff5c6a'
    }
    # }

    response = requests.request("POST", url, headers=headers, data=payload, allow_redirects=False)
    if 'DCloading' in st.session_state:
      DCloading.success("API call successful")
    return response



def generate_bar_chart(fig, data_df, config):
    for column in data_df.columns:
        if column != config['x_column']:
            fig.add_trace(go.Bar(
                x=data_df[config['x_column']],
                y=data_df[column],
                name=column,
                text=data_df[column].apply(lambda x: f'{x:.2f}'),  # Format labels to 2 decimals
                textposition='outside',  # Position the labels outside the bars
                insidetextanchor='start',  # Align the labels to the left
                marker=dict(color=config['color'])
            ))

    fig.update_layout(title=config['title'])
    fig.update_xaxes(title_text='Testcase')
    fig.update_yaxes(title_text='Amount ($)')
    fig.update_xaxes(type="category")

def generate_comb_chart(data, value_pairs, title):
    # Create a DataFrame
    df = pd.DataFrame(data)

    # Calculate the bar width based on the data
    bar_width = (df["Historical"].max() - df["Historical"].min()) / len(df)

    # Create a bar chart using Matplotlib with uniform bar widths
    fig, ax = plt.subplots()
    ax.bar(df["Historical"], df["Count"], width=bar_width, align='center', color='#020887', label="Original Data")

    min_value = value_pairs["Min"]
    max_value = value_pairs["Max"]
    ax.fill_betweenx(y=[ax.get_ylim()[0], ax.get_ylim()[1]], x1=min_value, x2=max_value, color='#f4d35e', alpha=0.7, label="Min-Max Range", zorder=0)

    # Plot the new value pairs as vertical lines
    for label, value in value_pairs.items():
        line_color = '#BA1B1D' if label == "Analyst Prediction" else '#f4d35e'
        line_width = 2 if label == "Analyst Prediction" else 1
        ax.axvline(x=value, color=line_color, linestyle='-', linewidth=line_width, label=f"{label}: {value}")

    # Customize the chart
    ax.set_xlabel(title)
    ax.set_ylabel("Count")
    ax.set_title(title + " Distribution")

    # Add a legend
    ax.legend()

    return fig

def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / stddev) ** 2)

def generate_comb_chart_best_fit(data, value_pairs, title):
    # Create a DataFrame
    df = pd.DataFrame(data)

    # Set the font family
    plt.rcParams["font.family"] = ["Arial"]

    # Create a scatter plot using Matplotlib
    fig, ax = plt.subplots()

    # Scatter plot with smaller dots and custom color
    ax.scatter(df["Historical"], df["Count"], color='#a0a0a0', label="Original Data", s=30)

    # Fit the model to the data
    params, covariance = curve_fit(gaussian, df["Historical"], df["Count"], p0=[1, np.mean(df["Historical"]), 1])
    amplitude, mean, stddev = params

    # Create x values for the fitted curve
    x_fit = np.linspace(min(df["Historical"]), max(df["Historical"]), 100)

    # Generate the fitted curve using the fitted parameters
    y_fit = gaussian(x_fit, amplitude, mean, stddev)

    # Plot the fitted curve as a best-fit line with blue color
    ax.plot(x_fit, y_fit, color='blue', linestyle='-', linewidth=2, label="Best-Fit Line")

    # Plot the new value pairs as vertical lines
    for label, value in value_pairs.items():
        line_color = 'red' if label == "Analyst Prediction" else '#f4d35e'
        line_width = 1
        ax.axvline(x=value, color=line_color, linestyle='-', linewidth=line_width, label=f"{label}: {value}")

    # Fill the area below the best-fit line with blue color
    ax.fill_between(x_fit, y_fit, color='blue', alpha=0.2, label="Area Below Best-Fit Line")

    # Plot the min-max range area
    min_value = value_pairs["Min"]
    max_value = value_pairs["Max"]
    ax.fill_betweenx(y=[ax.get_ylim()[0], ax.get_ylim()[1]], x1=min_value, x2=max_value, color='#f4d35e', alpha=0.5, label="Min-Max Range")

    # Customize the chart
    ax.set_xlabel(title)
    ax.set_ylabel("Count")
    ax.set_title(title + " Distribution")

    # Add a legend
    ax.legend()

    return fig

def multiply_and_convert_to_json(input_df):
    # Create a copy of the input DataFrame
    modified_df = input_df.copy()

    for column in modified_df.columns[1:]:  # Start from the second column
      modified_df[column] = pd.to_numeric(modified_df[column], errors='coerce', downcast='integer')

    # Convert the DataFrame to JSON format
    json_data = modified_df.to_json(orient='records')

    return json_data

#Start of UI
image_path = "coherent-logo.png"
st.image(image_path, caption="", width=32)

st.write("## Company Valuations - Region & Sector Analysis")
st.text("‎") 


#initialize data
outputs = {"CompanyResults":[{"COMPANIES":"ALL in Japan","#1":"Sungrow","#2":"","#3":"","#4":"","#5":"","#6":"","#7":"","#8":"","#9":"","#10":""},{"COMPANIES":"Simualtions","#1":256,"#2":"","#3":"","#4":"","#5":"","#6":"","#7":"","#8":"","#9":"","#10":""},{"COMPANIES":"COGS","#1":"NA","#2":"","#3":"","#4":"","#5":"","#6":"","#7":"","#8":"","#9":"","#10":""},{"COMPANIES":"Gross Margin","#1":-0.483069469616304,"#2":"","#3":"","#4":"","#5":"","#6":"","#7":"","#8":"","#9":"","#10":""},{"COMPANIES":"Net Margin","#1":8.70801641373751,"#2":"","#3":"","#4":"","#5":"","#6":"","#7":"","#8":"","#9":"","#10":""},{"COMPANIES":"Profit bf Tax","#1":"NA","#2":"","#3":"","#4":"","#5":"","#6":"","#7":"","#8":"","#9":"","#10":""},{"COMPANIES":"Revenue","#1":"NA","#2":"","#3":"","#4":"","#5":"","#6":"","#7":"","#8":"","#9":"","#10":""},{"COMPANIES":"Revenue Growth","#1":-0.133282547135072,"#2":"","#3":"","#4":"","#5":"","#6":"","#7":"","#8":"","#9":"","#10":""},{"COMPANIES":"Target Multilple","#1":16.0999999999999,"#2":"","#3":"","#4":"","#5":"","#6":"","#7":"","#8":"","#9":"","#10":""},{"COMPANIES":"Target Price","#1":140.25,"#2":"","#3":"","#4":"","#5":"","#6":"","#7":"","#8":"","#9":"","#10":""},{"COMPANIES":"Target Price (Upside)","#1":"NA","#2":"","#3":"","#4":"","#5":"","#6":"","#7":"","#8":"","#9":"","#10":""}],"companySummary_byRegion":[{"#":1,"Region":"China","Company #1":"MGM China","Company #2":"Yum China","Company #3":0,"Company #4":0,"Company #5":0,"Company #6":0,"Company #7":0,"Company #8":0,"Company #9":0,"Company #10":0,"Company #11":0,"Company #12":0,"Company #13":0,"Company #14":0,"Company #15":0,"Company #16":0,"Company #17":0,"Company #18":0,"Company #19":0,"Company #20":0,"Company #21":0,"Company #22":0,"Company #23":0,"Company #24":0,"Company #25":0},{"#":2,"Region":"Korea","Company #1":"PICC","Company #2":0,"Company #3":0,"Company #4":0,"Company #5":0,"Company #6":0,"Company #7":0,"Company #8":0,"Company #9":0,"Company #10":0,"Company #11":0,"Company #12":0,"Company #13":0,"Company #14":0,"Company #15":0,"Company #16":0,"Company #17":0,"Company #18":0,"Company #19":0,"Company #20":0,"Company #21":0,"Company #22":0,"Company #23":0,"Company #24":0,"Company #25":0},{"#":3,"Region":"Japan","Company #1":"Sungrow","Company #2":0,"Company #3":0,"Company #4":0,"Company #5":0,"Company #6":0,"Company #7":0,"Company #8":0,"Company #9":0,"Company #10":0,"Company #11":0,"Company #12":0,"Company #13":0,"Company #14":0,"Company #15":0,"Company #16":0,"Company #17":0,"Company #18":0,"Company #19":0,"Company #20":0,"Company #21":0,"Company #22":0,"Company #23":0,"Company #24":0,"Company #25":0},{"#":4,"Region":"","Company #1":0,"Company #2":0,"Company #3":0,"Company #4":0,"Company #5":0,"Company #6":0,"Company #7":0,"Company #8":0,"Company #9":0,"Company #10":0,"Company #11":0,"Company #12":0,"Company #13":0,"Company #14":0,"Company #15":0,"Company #16":0,"Company #17":0,"Company #18":0,"Company #19":0,"Company #20":0,"Company #21":0,"Company #22":0,"Company #23":0,"Company #24":0,"Company #25":0},{"#":5,"Region":"","Company #1":0,"Company #2":0,"Company #3":0,"Company #4":0,"Company #5":0,"Company #6":0,"Company #7":0,"Company #8":0,"Company #9":0,"Company #10":0,"Company #11":0,"Company #12":0,"Company #13":0,"Company #14":0,"Company #15":0,"Company #16":0,"Company #17":0,"Company #18":0,"Company #19":0,"Company #20":0,"Company #21":0,"Company #22":0,"Company #23":0,"Company #24":0,"Company #25":0},{"#":6,"Region":"","Company #1":0,"Company #2":0,"Company #3":0,"Company #4":0,"Company #5":0,"Company #6":0,"Company #7":0,"Company #8":0,"Company #9":0,"Company #10":0,"Company #11":0,"Company #12":0,"Company #13":0,"Company #14":0,"Company #15":0,"Company #16":0,"Company #17":0,"Company #18":0,"Company #19":0,"Company #20":0,"Company #21":0,"Company #22":0,"Company #23":0,"Company #24":0,"Company #25":0},{"#":7,"Region":"","Company #1":0,"Company #2":0,"Company #3":0,"Company #4":0,"Company #5":0,"Company #6":0,"Company #7":0,"Company #8":0,"Company #9":0,"Company #10":0,"Company #11":0,"Company #12":0,"Company #13":0,"Company #14":0,"Company #15":0,"Company #16":0,"Company #17":0,"Company #18":0,"Company #19":0,"Company #20":0,"Company #21":0,"Company #22":0,"Company #23":0,"Company #24":0,"Company #25":0},{"#":8,"Region":"","Company #1":0,"Company #2":0,"Company #3":0,"Company #4":0,"Company #5":0,"Company #6":0,"Company #7":0,"Company #8":0,"Company #9":0,"Company #10":0,"Company #11":0,"Company #12":0,"Company #13":0,"Company #14":0,"Company #15":0,"Company #16":0,"Company #17":0,"Company #18":0,"Company #19":0,"Company #20":0,"Company #21":0,"Company #22":0,"Company #23":0,"Company #24":0,"Company #25":0},{"#":9,"Region":"","Company #1":0,"Company #2":0,"Company #3":0,"Company #4":0,"Company #5":0,"Company #6":0,"Company #7":0,"Company #8":0,"Company #9":0,"Company #10":0,"Company #11":0,"Company #12":0,"Company #13":0,"Company #14":0,"Company #15":0,"Company #16":0,"Company #17":0,"Company #18":0,"Company #19":0,"Company #20":0,"Company #21":0,"Company #22":0,"Company #23":0,"Company #24":0,"Company #25":0},{"#":10,"Region":"","Company #1":0,"Company #2":0,"Company #3":0,"Company #4":0,"Company #5":0,"Company #6":0,"Company #7":0,"Company #8":0,"Company #9":0,"Company #10":0,"Company #11":0,"Company #12":0,"Company #13":0,"Company #14":0,"Company #15":0,"Company #16":0,"Company #17":0,"Company #18":0,"Company #19":0,"Company #20":0,"Company #21":0,"Company #22":0,"Company #23":0,"Company #24":0,"Company #25":0},{"#":11,"Region":"","Company #1":0,"Company #2":0,"Company #3":0,"Company #4":0,"Company #5":0,"Company #6":0,"Company #7":0,"Company #8":0,"Company #9":0,"Company #10":0,"Company #11":0,"Company #12":0,"Company #13":0,"Company #14":0,"Company #15":0,"Company #16":0,"Company #17":0,"Company #18":0,"Company #19":0,"Company #20":0,"Company #21":0,"Company #22":0,"Company #23":0,"Company #24":0,"Company #25":0},{"#":12,"Region":"","Company #1":0,"Company #2":0,"Company #3":0,"Company #4":0,"Company #5":0,"Company #6":0,"Company #7":0,"Company #8":0,"Company #9":0,"Company #10":0,"Company #11":0,"Company #12":0,"Company #13":0,"Company #14":0,"Company #15":0,"Company #16":0,"Company #17":0,"Company #18":0,"Company #19":0,"Company #20":0,"Company #21":0,"Company #22":0,"Company #23":0,"Company #24":0,"Company #25":0},{"#":13,"Region":"","Company #1":0,"Company #2":0,"Company #3":0,"Company #4":0,"Company #5":0,"Company #6":0,"Company #7":0,"Company #8":0,"Company #9":0,"Company #10":0,"Company #11":0,"Company #12":0,"Company #13":0,"Company #14":0,"Company #15":0,"Company #16":0,"Company #17":0,"Company #18":0,"Company #19":0,"Company #20":0,"Company #21":0,"Company #22":0,"Company #23":0,"Company #24":0,"Company #25":0},{"#":14,"Region":"","Company #1":0,"Company #2":0,"Company #3":0,"Company #4":0,"Company #5":0,"Company #6":0,"Company #7":0,"Company #8":0,"Company #9":0,"Company #10":0,"Company #11":0,"Company #12":0,"Company #13":0,"Company #14":0,"Company #15":0,"Company #16":0,"Company #17":0,"Company #18":0,"Company #19":0,"Company #20":0,"Company #21":0,"Company #22":0,"Company #23":0,"Company #24":0,"Company #25":0},{"#":15,"Region":"","Company #1":0,"Company #2":0,"Company #3":0,"Company #4":0,"Company #5":0,"Company #6":0,"Company #7":0,"Company #8":0,"Company #9":0,"Company #10":0,"Company #11":0,"Company #12":0,"Company #13":0,"Company #14":0,"Company #15":0,"Company #16":0,"Company #17":0,"Company #18":0,"Company #19":0,"Company #20":0,"Company #21":0,"Company #22":0,"Company #23":0,"Company #24":0,"Company #25":0},{"#":16,"Region":"","Company #1":0,"Company #2":0,"Company #3":0,"Company #4":0,"Company #5":0,"Company #6":0,"Company #7":0,"Company #8":0,"Company #9":0,"Company #10":0,"Company #11":0,"Company #12":0,"Company #13":0,"Company #14":0,"Company #15":0,"Company #16":0,"Company #17":0,"Company #18":0,"Company #19":0,"Company #20":0,"Company #21":0,"Company #22":0,"Company #23":0,"Company #24":0,"Company #25":0},{"#":17,"Region":"","Company #1":0,"Company #2":0,"Company #3":0,"Company #4":0,"Company #5":0,"Company #6":0,"Company #7":0,"Company #8":0,"Company #9":0,"Company #10":0,"Company #11":0,"Company #12":0,"Company #13":0,"Company #14":0,"Company #15":0,"Company #16":0,"Company #17":0,"Company #18":0,"Company #19":0,"Company #20":0,"Company #21":0,"Company #22":0,"Company #23":0,"Company #24":0,"Company #25":0},{"#":18,"Region":"","Company #1":0,"Company #2":0,"Company #3":0,"Company #4":0,"Company #5":0,"Company #6":0,"Company #7":0,"Company #8":0,"Company #9":0,"Company #10":0,"Company #11":0,"Company #12":0,"Company #13":0,"Company #14":0,"Company #15":0,"Company #16":0,"Company #17":0,"Company #18":0,"Company #19":0,"Company #20":0,"Company #21":0,"Company #22":0,"Company #23":0,"Company #24":0,"Company #25":0},{"#":19,"Region":"","Company #1":0,"Company #2":0,"Company #3":0,"Company #4":0,"Company #5":0,"Company #6":0,"Company #7":0,"Company #8":0,"Company #9":0,"Company #10":0,"Company #11":0,"Company #12":0,"Company #13":0,"Company #14":0,"Company #15":0,"Company #16":0,"Company #17":0,"Company #18":0,"Company #19":0,"Company #20":0,"Company #21":0,"Company #22":0,"Company #23":0,"Company #24":0,"Company #25":0},{"#":20,"Region":"","Company #1":0,"Company #2":0,"Company #3":0,"Company #4":0,"Company #5":0,"Company #6":0,"Company #7":0,"Company #8":0,"Company #9":0,"Company #10":0,"Company #11":0,"Company #12":0,"Company #13":0,"Company #14":0,"Company #15":0,"Company #16":0,"Company #17":0,"Company #18":0,"Company #19":0,"Company #20":0,"Company #21":0,"Company #22":0,"Company #23":0,"Company #24":0,"Company #25":0}],"companySummary_bySector":[{"#":1,"Sector":"Gaming","Company #1":"MGM China","Company #2":0,"Company #3":0,"Company #4":0,"Company #5":0,"Company #6":0,"Company #7":0,"Company #8":0,"Company #9":0,"Company #10":0,"Company #11":0,"Company #12":0,"Company #13":0,"Company #14":0,"Company #15":0,"Company #16":0,"Company #17":0,"Company #18":0,"Company #19":0,"Company #20":0,"Company #21":0,"Company #22":0,"Company #23":0,"Company #24":0,"Company #25":0},{"#":2,"Sector":"Energy","Company #1":"PICC","Company #2":0,"Company #3":0,"Company #4":0,"Company #5":0,"Company #6":0,"Company #7":0,"Company #8":0,"Company #9":0,"Company #10":0,"Company #11":0,"Company #12":0,"Company #13":0,"Company #14":0,"Company #15":0,"Company #16":0,"Company #17":0,"Company #18":0,"Company #19":0,"Company #20":0,"Company #21":0,"Company #22":0,"Company #23":0,"Company #24":0,"Company #25":0},{"#":3,"Sector":"F&B","Company #1":"Sungrow","Company #2":"Yum China","Company #3":0,"Company #4":0,"Company #5":0,"Company #6":0,"Company #7":0,"Company #8":0,"Company #9":0,"Company #10":0,"Company #11":0,"Company #12":0,"Company #13":0,"Company #14":0,"Company #15":0,"Company #16":0,"Company #17":0,"Company #18":0,"Company #19":0,"Company #20":0,"Company #21":0,"Company #22":0,"Company #23":0,"Company #24":0,"Company #25":0},{"#":4,"Sector":"","Company #1":0,"Company #2":0,"Company #3":0,"Company #4":0,"Company #5":0,"Company #6":0,"Company #7":0,"Company #8":0,"Company #9":0,"Company #10":0,"Company #11":0,"Company #12":0,"Company #13":0,"Company #14":0,"Company #15":0,"Company #16":0,"Company #17":0,"Company #18":0,"Company #19":0,"Company #20":0,"Company #21":0,"Company #22":0,"Company #23":0,"Company #24":0,"Company #25":0},{"#":5,"Sector":"","Company #1":0,"Company #2":0,"Company #3":0,"Company #4":0,"Company #5":0,"Company #6":0,"Company #7":0,"Company #8":0,"Company #9":0,"Company #10":0,"Company #11":0,"Company #12":0,"Company #13":0,"Company #14":0,"Company #15":0,"Company #16":0,"Company #17":0,"Company #18":0,"Company #19":0,"Company #20":0,"Company #21":0,"Company #22":0,"Company #23":0,"Company #24":0,"Company #25":0},{"#":6,"Sector":"","Company #1":0,"Company #2":0,"Company #3":0,"Company #4":0,"Company #5":0,"Company #6":0,"Company #7":0,"Company #8":0,"Company #9":0,"Company #10":0,"Company #11":0,"Company #12":0,"Company #13":0,"Company #14":0,"Company #15":0,"Company #16":0,"Company #17":0,"Company #18":0,"Company #19":0,"Company #20":0,"Company #21":0,"Company #22":0,"Company #23":0,"Company #24":0,"Company #25":0},{"#":7,"Sector":"","Company #1":0,"Company #2":0,"Company #3":0,"Company #4":0,"Company #5":0,"Company #6":0,"Company #7":0,"Company #8":0,"Company #9":0,"Company #10":0,"Company #11":0,"Company #12":0,"Company #13":0,"Company #14":0,"Company #15":0,"Company #16":0,"Company #17":0,"Company #18":0,"Company #19":0,"Company #20":0,"Company #21":0,"Company #22":0,"Company #23":0,"Company #24":0,"Company #25":0},{"#":8,"Sector":"","Company #1":0,"Company #2":0,"Company #3":0,"Company #4":0,"Company #5":0,"Company #6":0,"Company #7":0,"Company #8":0,"Company #9":0,"Company #10":0,"Company #11":0,"Company #12":0,"Company #13":0,"Company #14":0,"Company #15":0,"Company #16":0,"Company #17":0,"Company #18":0,"Company #19":0,"Company #20":0,"Company #21":0,"Company #22":0,"Company #23":0,"Company #24":0,"Company #25":0},{"#":9,"Sector":"","Company #1":0,"Company #2":0,"Company #3":0,"Company #4":0,"Company #5":0,"Company #6":0,"Company #7":0,"Company #8":0,"Company #9":0,"Company #10":0,"Company #11":0,"Company #12":0,"Company #13":0,"Company #14":0,"Company #15":0,"Company #16":0,"Company #17":0,"Company #18":0,"Company #19":0,"Company #20":0,"Company #21":0,"Company #22":0,"Company #23":0,"Company #24":0,"Company #25":0},{"#":10,"Sector":"","Company #1":0,"Company #2":0,"Company #3":0,"Company #4":0,"Company #5":0,"Company #6":0,"Company #7":0,"Company #8":0,"Company #9":0,"Company #10":0,"Company #11":0,"Company #12":0,"Company #13":0,"Company #14":0,"Company #15":0,"Company #16":0,"Company #17":0,"Company #18":0,"Company #19":0,"Company #20":0,"Company #21":0,"Company #22":0,"Company #23":0,"Company #24":0,"Company #25":0},{"#":11,"Sector":"","Company #1":0,"Company #2":0,"Company #3":0,"Company #4":0,"Company #5":0,"Company #6":0,"Company #7":0,"Company #8":0,"Company #9":0,"Company #10":0,"Company #11":0,"Company #12":0,"Company #13":0,"Company #14":0,"Company #15":0,"Company #16":0,"Company #17":0,"Company #18":0,"Company #19":0,"Company #20":0,"Company #21":0,"Company #22":0,"Company #23":0,"Company #24":0,"Company #25":0},{"#":12,"Sector":"","Company #1":0,"Company #2":0,"Company #3":0,"Company #4":0,"Company #5":0,"Company #6":0,"Company #7":0,"Company #8":0,"Company #9":0,"Company #10":0,"Company #11":0,"Company #12":0,"Company #13":0,"Company #14":0,"Company #15":0,"Company #16":0,"Company #17":0,"Company #18":0,"Company #19":0,"Company #20":0,"Company #21":0,"Company #22":0,"Company #23":0,"Company #24":0,"Company #25":0},{"#":13,"Sector":"","Company #1":0,"Company #2":0,"Company #3":0,"Company #4":0,"Company #5":0,"Company #6":0,"Company #7":0,"Company #8":0,"Company #9":0,"Company #10":0,"Company #11":0,"Company #12":0,"Company #13":0,"Company #14":0,"Company #15":0,"Company #16":0,"Company #17":0,"Company #18":0,"Company #19":0,"Company #20":0,"Company #21":0,"Company #22":0,"Company #23":0,"Company #24":0,"Company #25":0},{"#":14,"Sector":"","Company #1":0,"Company #2":0,"Company #3":0,"Company #4":0,"Company #5":0,"Company #6":0,"Company #7":0,"Company #8":0,"Company #9":0,"Company #10":0,"Company #11":0,"Company #12":0,"Company #13":0,"Company #14":0,"Company #15":0,"Company #16":0,"Company #17":0,"Company #18":0,"Company #19":0,"Company #20":0,"Company #21":0,"Company #22":0,"Company #23":0,"Company #24":0,"Company #25":0},{"#":15,"Sector":"","Company #1":0,"Company #2":0,"Company #3":0,"Company #4":0,"Company #5":0,"Company #6":0,"Company #7":0,"Company #8":0,"Company #9":0,"Company #10":0,"Company #11":0,"Company #12":0,"Company #13":0,"Company #14":0,"Company #15":0,"Company #16":0,"Company #17":0,"Company #18":0,"Company #19":0,"Company #20":0,"Company #21":0,"Company #22":0,"Company #23":0,"Company #24":0,"Company #25":0},{"#":16,"Sector":"","Company #1":0,"Company #2":0,"Company #3":0,"Company #4":0,"Company #5":0,"Company #6":0,"Company #7":0,"Company #8":0,"Company #9":0,"Company #10":0,"Company #11":0,"Company #12":0,"Company #13":0,"Company #14":0,"Company #15":0,"Company #16":0,"Company #17":0,"Company #18":0,"Company #19":0,"Company #20":0,"Company #21":0,"Company #22":0,"Company #23":0,"Company #24":0,"Company #25":0},{"#":17,"Sector":"","Company #1":0,"Company #2":0,"Company #3":0,"Company #4":0,"Company #5":0,"Company #6":0,"Company #7":0,"Company #8":0,"Company #9":0,"Company #10":0,"Company #11":0,"Company #12":0,"Company #13":0,"Company #14":0,"Company #15":0,"Company #16":0,"Company #17":0,"Company #18":0,"Company #19":0,"Company #20":0,"Company #21":0,"Company #22":0,"Company #23":0,"Company #24":0,"Company #25":0},{"#":18,"Sector":"","Company #1":0,"Company #2":0,"Company #3":0,"Company #4":0,"Company #5":0,"Company #6":0,"Company #7":0,"Company #8":0,"Company #9":0,"Company #10":0,"Company #11":0,"Company #12":0,"Company #13":0,"Company #14":0,"Company #15":0,"Company #16":0,"Company #17":0,"Company #18":0,"Company #19":0,"Company #20":0,"Company #21":0,"Company #22":0,"Company #23":0,"Company #24":0,"Company #25":0},{"#":19,"Sector":"","Company #1":0,"Company #2":0,"Company #3":0,"Company #4":0,"Company #5":0,"Company #6":0,"Company #7":0,"Company #8":0,"Company #9":0,"Company #10":0,"Company #11":0,"Company #12":0,"Company #13":0,"Company #14":0,"Company #15":0,"Company #16":0,"Company #17":0,"Company #18":0,"Company #19":0,"Company #20":0,"Company #21":0,"Company #22":0,"Company #23":0,"Company #24":0,"Company #25":0},{"#":20,"Sector":"","Company #1":0,"Company #2":0,"Company #3":0,"Company #4":0,"Company #5":0,"Company #6":0,"Company #7":0,"Company #8":0,"Company #9":0,"Company #10":0,"Company #11":0,"Company #12":0,"Company #13":0,"Company #14":0,"Company #15":0,"Company #16":0,"Company #17":0,"Company #18":0,"Company #19":0,"Company #20":0,"Company #21":0,"Company #22":0,"Company #23":0,"Company #24":0,"Company #25":0}],"noOfCompanies":1}
DCerrors = []

with st.expander("Spark Model", expanded=True):
  st.markdown('[https://spark.uat.jp.coherent.global/clsa/products/Original%20Models/Output%20Analysis%20-%20by%20Sector%20&%20Region/apiTester/test](https://spark.uat.jp.coherent.global/clsa/products/Original%20Models/Output%20Analysis%20-%20by%20Sector%20&%20Region/apiTester/test)')

st.write("Select Parameters")
with st.form("DC Form"):
  
  Go = True
  ERRORBOX = st.empty()
  DCLoading = st.empty()

  col01, col02 = st.columns([1,1])
  with col01:
    SectorOptions = ["ALL", "Gaming", "Energy", "Healthcare", "F&B", "Healthcare", "Banking", "Technology", "Technology"]
    SectorInput = st.selectbox("Sector", SectorOptions)

  with col02:
    # make an array with string all, china korea japan
    RegionOptions = ["ALL", "China", "Korea", "Australia", "China", "Australia", "Malaysia", "Singapore", "Taiwan"]
    RegionInput = st.selectbox("Region", RegionOptions)

  DCbutton_clicked = st.form_submit_button("Calculate")
  if DCbutton_clicked:   
    DCalldata = definedCombination()
    outputs = DCalldata.json()['response_data']['outputs']
    # st.json(Spark_outputs)
    DCerrors = DCalldata.json()['response_data']['errors']

  # Add the style tag to change button color to blue
  st.markdown("""
  <style>
      .stButton button { /* Adjust the class name according to your button's class */
          background-color: blue !important;
          color: white !important;
      }
  </style>
  """, unsafe_allow_html=True)


st.write("Results")
with st.expander("", expanded=True):
  NumCompanies_Metric_placeholder = st.empty()
  st.write('Filtered Companies:')
  SummaryOfCompanies_Df_placeholder = st.empty()
  st.write('Companies by Region:')
  SummaryByRegion_Df_placeholder = st.empty()
  st.write('Companies by Sector:')
  SummaryBySector_Df_placeholder = st.empty()

  NumCompanies_Metric_placeholder.metric("Number of Companies", outputs['noOfCompanies'])
  
  SummaryByRegion_Df = pd.DataFrame(outputs['companySummary_byRegion'])
  SummaryByRegion_Df_placeholder.dataframe(SummaryByRegion_Df, use_container_width=True)
  
  SummaryBySector_Df = pd.DataFrame(outputs['companySummary_bySector'])
  SummaryBySector_Df_placeholder.dataframe(SummaryBySector_Df, use_container_width=True)
  
  SummaryOfCompanies_Df = pd.DataFrame(outputs['CompanyResults'])
  SummaryOfCompanies_Df_placeholder.dataframe(SummaryOfCompanies_Df, use_container_width=True)
