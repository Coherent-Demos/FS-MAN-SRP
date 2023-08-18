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

@st.cache_data
def definedCombination(inputdata):
    if 'DCloading' in st.session_state:
      DCloading.warning("Running Simulations")
    url = "https://excel.uat.us.coherent.global/coherent/api/v3/folders/Spark FE Demos/services/Xcall Yum China - Defined Comb - output template4/Execute"

    payload = json.dumps({
       "request_data": {
          "inputs": {
            "Base_Inputs": inputdata
          }
       },
        "request_meta": {
            "compiler_type": "Neuron",
        }
    })
    headers = {
       'Content-Type': 'application/json',
       'x-tenant-name': 'coherent',
       'SecretKey': '2277565c-9fad-4bf4-ad2b-1efe5748dd11'
    }

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

    # Set the font family
    plt.rcParams["font.family"] = ["Arial"]

    # Create a bar chart using Matplotlib
    fig, ax = plt.subplots()
    ax.bar(df["Historical"], df["Count"], color='#020887', label="Original Data")

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
      modified_df[column] = pd.to_numeric(modified_df[column], errors='coerce', downcast='integer') / 100

    # Convert the DataFrame to JSON format
    json_data = modified_df.to_json(orient='records')

    return json_data

#Start of UI
image_path = "coherent-logo.png"
st.image(image_path, caption="", width=32)

st.write("## Pricing Simulation")
st.text("‎") 


#initialize data
json_data = [
  {
    "Inputs": "KFC - Cost of Sales (%)",
    "Historical 1 sd": 0.0621,
    "2023 Point assumption": 0.31,
    "2023 Min": 0.2479,
    "2023 Max": 0.3721,
    "2024 Point assumption": 0.31,
    "2024 Min": 0.2479,
    "2024 Max": 0.3721
  },
  {
    "Inputs": "KFC - SSSG (%)",
    "Historical 1 sd": 0.0142,
    "2023 Point assumption": 0.05,
    "2023 Min": 0.0358,
    "2023 Max": 0.0642,
    "2024 Point assumption": 0.01,
    "2024 Min": -0.0042,
    "2024 Max": 0.0242
  },
  {
    "Inputs": "Pizza Hut - Cost of Sales (%)",
    "Historical 1 sd": 0.0604,
    "2023 Point assumption": 0.31,
    "2023 Min": 0.2496,
    "2023 Max": 0.3704,
    "2024 Point assumption": 0.31,
    "2024 Min": 0.2496,
    "2024 Max": 0.3704
  },
  {
    "Inputs": "Pizza Hut - SSSG (%)",
    "Historical 1 sd": 0.0177,
    "2023 Point assumption": 0.07,
    "2023 Min": 0.0523,
    "2023 Max": 0.0877,
    "2024 Point assumption": 0.02,
    "2024 Min": 0.0023,
    "2024 Max": 0.0377
  }
]

DCoutputs = {"Simualtions":6561,"rg_htable":[{"Historical":-23,"Count":0},{"Historical":-20,"Count":1},{"Historical":-17,"Count":0},{"Historical":-14,"Count":1},{"Historical":-11,"Count":2},{"Historical":-8,"Count":1},{"Historical":-5,"Count":3},{"Historical":-2,"Count":3},{"Historical":1,"Count":9},{"Historical":4,"Count":7},{"Historical":7,"Count":1},{"Historical":10,"Count":2},{"Historical":13,"Count":4},{"Historical":16,"Count":0},{"Historical":19,"Count":0},{"Historical":22,"Count":0},{"Historical":25,"Count":0},{"Historical":28,"Count":2},{"Historical":31,"Count":0},{"Historical":34,"Count":1},{"Historical":37,"Count":0},{"Historical":40,"Count":0},{"Historical":43,"Count":1},{"Historical":46,"Count":0}],"npm_htable":[{"Historical":-5,"Count":0},{"Historical":-4,"Count":0},{"Historical":-3,"Count":0},{"Historical":-2,"Count":1},{"Historical":-1,"Count":1},{"Historical":0,"Count":0},{"Historical":1,"Count":1},{"Historical":2,"Count":0},{"Historical":3,"Count":4},{"Historical":4,"Count":6},{"Historical":5,"Count":1},{"Historical":6,"Count":4},{"Historical":7,"Count":2},{"Historical":8,"Count":5},{"Historical":9,"Count":7},{"Historical":10,"Count":3},{"Historical":11,"Count":1},{"Historical":12,"Count":1},{"Historical":13,"Count":0},{"Historical":14,"Count":0},{"Historical":15,"Count":0},{"Historical":16,"Count":0},{"Historical":17,"Count":0},{"Historical":18,"Count":1},{"Historical":19,"Count":0},{"Historical":20,"Count":0}],"Avg_Target_Price":65.9358329522939,"Avg_Profit_before_Tax":1234.48363123162,"Avg_Revenue_growth":22.3180426153235,"Avg_Revenue":11704.6134978605,"minmaxtable":[{"Metric":"Analys Prediction","Revenue Growth":22.3180426153223,"Net Profit Margin":7.31448737395571,"Gross Margin":70.1100673500095},{"Metric":"Min (Simulation)","Revenue Growth":20.4222890641395,"Net Profit Margin":8.74377992124965,"Gross Margin":64.1633419557201},{"Metric":"Max (Simulation)","Revenue Growth":24.2137961665051,"Net Profit Margin":8.79453121893399,"Gross Margin":76.0498405626466}],"gm_htable":[{"Historical":65,"Count":1},{"Historical":66,"Count":1},{"Historical":67,"Count":1},{"Historical":68,"Count":6},{"Historical":69,"Count":1},{"Historical":70,"Count":12},{"Historical":71,"Count":17},{"Historical":72,"Count":11},{"Historical":73,"Count":6},{"Historical":74,"Count":2},{"Historical":75,"Count":0}],"Avg_COGS":-3498.5010914521,"Avg_Net_profit_margin":8.76949725186518,"Avg_Gross_Margin":70.110182486921}
DCerrors = []

st.write("Edit Min and Max Values")
with st.form("DC Form"):
  Go = True
  ERRORBOX = st.empty()
  DCLoading = st.empty()
  def highlight_col(x):
    r = 'background-color: #fafafa; color: #a0a0a0'
    df1 = pd.DataFrame('', index=x.index, columns=x.columns)
    df1.iloc[:, 1:] = r
    return df1  
  
  # Create a DataFrame from JSON data
  df = pd.DataFrame(json_data)
  numeric_columns = df.select_dtypes(include=[float, int]).columns
  df[numeric_columns] = df[numeric_columns] * 100
  df[numeric_columns] = df[numeric_columns].applymap('{:.1f}'.format)

  inputTable = st.data_editor(
    df.style.apply(highlight_col, axis=None),
    use_container_width=True,
    column_config={
      "Inputs": st.column_config.Column(disabled=True),
      "Historical 1 sd": st.column_config.Column("Historical (1SD)", disabled=True),
      "2023 Point assumption": st.column_config.Column("2023 Point assumption %", disabled=True),
      "2024 Point assumption": st.column_config.Column("2024 Point assumption %", disabled=True),
      "2023 Min": st.column_config.Column("2023 Min (%)"),
      "2023 Max": st.column_config.Column("2023 Max (%)"),
      "2024 Min": st.column_config.Column("2024 Min (%)"),
      "2024 Max": st.column_config.Column("2024 Max (%)")

    }
  )

  DCbutton_clicked = st.form_submit_button("Generate Output")
  if DCbutton_clicked:   
    for index, row in inputTable.iterrows():
        if row["2023 Min"] > row["2023 Max"] or row["2024 Min"] > row["2024 Max"]:
          APIGO = False
          break
        else:
          APIGO = True
    if True:      
    # if APIGO==True:
      apiInput = multiply_and_convert_to_json(inputTable)
      apiInput_dict = json.loads(apiInput)
      DCalldata = definedCombination(apiInput_dict)
      DCoutputs = DCalldata.json()['response_data']['outputs']
      DCerrors = DCalldata.json()['response_data']['errors']
    else:
      ERRORBOX.error("Min must be greater than Max")

  # Add the style tag to change button color to blue
  st.markdown("""
  <style>
      .stButton button { /* Adjust the class name according to your button's class */
          background-color: blue !important;
          color: white !important;
      }
  </style>
  """, unsafe_allow_html=True)


st.write("Simulation Results")
with st.expander("", expanded=True):
  # apiInput = multiply_and_convert_to_json(inputTable)
  # apiInput_dict = json.loads(apiInput)
  # DCalldata = definedCombination(apiInput_dict)
  # DCoutputs = DCalldata.json()['response_data']['outputs']

  col11, col12, col13 = st.columns([1,1,1])
  with col11:
    RG_CHART_placeholder = st.empty()
  with col12:
    GM_CHART_placeholder = st.empty()
  with col13:
    NPM_CHART_placeholder = st.empty()

  # st.markdown('***')
  # col11, col12, col13 = st.columns([1,1,1])
  # with col11:
  #   RG_CHART_bf_placeholder = st.empty()
  # with col12:
  #   GM_CHART_bf_placeholder = st.empty()
  # with col13:
  #   NPM_CHART_bf_placeholder = st.empty()

  st.markdown('***')
  col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1])
  with col1:
    DCNumberOfSimulations_placeholder = st.empty()
  with col2:
    DCAvgCost_placeholder = st.empty()
  with col3:  
    DCAvgProfit_placeholder = st.empty()
  with col4:  
    DCAvgRevenue_placeholder = st.empty()
  with col5:  
    DCAvgTargetPrice_placeholder = st.empty()

  # col31, col32 = st.columns([1,1])
  # with col31:
  #   st.json(apiInput)
  # with col32: 
  #   st.json(DCoutputs) 

  #generate line chart of results
  if not DCerrors:
    data_rg = pd.DataFrame(DCoutputs["rg_htable"])
    value_pairs_rg = {
      "Min": format(DCoutputs["minmaxtable"][1]["Revenue Growth"], ".1f"),
      "Max": format(DCoutputs["minmaxtable"][2]["Revenue Growth"], ".1f"),
      "Analyst Prediction": format(DCoutputs["minmaxtable"][0]["Revenue Growth"], ".1f")
    }
    chart_fig = generate_comb_chart(data_rg, value_pairs_rg, "Revenue Growth")
    RG_CHART_placeholder.pyplot(chart_fig)
    chart_fig = generate_comb_chart_best_fit(data_rg, value_pairs_rg, "Revenue Growth")
    # RG_CHART_bf_placeholder.pyplot(chart_fig)

    data_gm = pd.DataFrame(DCoutputs["gm_htable"])
    value_pairs_gm = {
      "Min": format(DCoutputs["minmaxtable"][1]["Gross Margin"], ".1f"),
      "Max": format(DCoutputs["minmaxtable"][2]["Gross Margin"], ".1f"),
      "Analyst Prediction": format(DCoutputs["minmaxtable"][0]["Gross Margin"], ".1f")
    }
    chart_fig = generate_comb_chart(data_gm, value_pairs_gm, "Gross Margin")
    GM_CHART_placeholder.pyplot(chart_fig)
    chart_fig = generate_comb_chart_best_fit(data_gm, value_pairs_gm, "Gross Margin")
    # GM_CHART_bf_placeholder.pyplot(chart_fig)

    data_npm = pd.DataFrame(DCoutputs["npm_htable"])
    value_pairs_npm = {
      "Min": format(DCoutputs["minmaxtable"][1]["Net Profit Margin"], ".2f"),
      "Max": format(DCoutputs["minmaxtable"][2]["Net Profit Margin"], ".2f"),
      "Analyst Prediction": format(DCoutputs["minmaxtable"][0]["Net Profit Margin"], ".2f")
    }
    chart_fig = generate_comb_chart(data_npm, value_pairs_npm, "Net Profit Margin")
    NPM_CHART_placeholder.pyplot(chart_fig)
    chart_fig = generate_comb_chart_best_fit(data_npm, value_pairs_npm, "Net Profit Margin")
    # NPM_CHART_bf_placeholder.pyplot(chart_fig)

    DCNumberOfSimulations = "{:,.0f}".format(DCoutputs["Simualtions"])
    DCNumberOfSimulations_placeholder.metric(label='Number of Simulations', value=DCNumberOfSimulations)

    DCAvgCost = "{:,.2f}".format(DCoutputs["Avg_COGS"])
    DCAvgCost_placeholder.metric(label='Avg Cost of Goods ($)', value=DCAvgCost)

    DCAvgProfit = "{:,.2f}".format(DCoutputs["Avg_Profit_before_Tax"])
    DCAvgProfit_placeholder.metric(label='Avg Profit b.Tax ($)', value=DCAvgProfit)

    DCAvgRevenue = "{:,.2f}".format(DCoutputs["Avg_Revenue"])
    DCAvgRevenue_placeholder.metric(label='Avg Revenue ($)', value=DCAvgRevenue)

    DCAvgTargetPrice = "{:,.2f}".format(DCoutputs["Avg_Target_Price"])
    DCAvgTargetPrice_placeholder.metric(label='Avg Target Price ($)', value=DCAvgTargetPrice)

    initState = False
  else:
    error_messages = [error["message"] for error in DCerrors]
    if error_messages:
        ERRORBOX.error("\n ".join(error_messages))
