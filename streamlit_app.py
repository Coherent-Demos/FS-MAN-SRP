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
    url = "https://excel.uat.jp.coherent.global/clsa/api/v3/folders/Trial/services/Xcall Yum China - Defined Comb - output template3A/Execute"

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
       'x-tenant-name': 'clsa',
       'x-synthetic-key': '3ca9da18-31fa-4a82-a9ba-44130dff5c6a'
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

DCoutputs = {"TM_htable":[{"Historical":16,"Count":11},{"Historical":18,"Count":86},{"Historical":20,"Count":138},{"Historical":22,"Count":349},{"Historical":24,"Count":342},{"Historical":26,"Count":151},{"Historical":28,"Count":270},{"Historical":30,"Count":213},{"Historical":32,"Count":91},{"Historical":34,"Count":13},{"Historical":36,"Count":26},{"Historical":38,"Count":11},{"Historical":40,"Count":11},{"Historical":42,"Count":8},{"Historical":44,"Count":0}],"Simualtions":6561,"rg_htable":[{"Historical":-23,"Count":0},{"Historical":-20,"Count":1},{"Historical":-17,"Count":0},{"Historical":-14,"Count":1},{"Historical":-11,"Count":2},{"Historical":-8,"Count":1},{"Historical":-5,"Count":3},{"Historical":-2,"Count":3},{"Historical":1,"Count":9},{"Historical":4,"Count":7},{"Historical":7,"Count":1},{"Historical":10,"Count":2},{"Historical":13,"Count":4},{"Historical":16,"Count":0},{"Historical":19,"Count":0},{"Historical":22,"Count":0},{"Historical":25,"Count":0},{"Historical":28,"Count":2},{"Historical":31,"Count":0},{"Historical":34,"Count":1},{"Historical":37,"Count":0},{"Historical":40,"Count":0},{"Historical":43,"Count":1},{"Historical":46,"Count":0}],"npm_htable":[{"Historical":-5,"Count":0},{"Historical":-4,"Count":0},{"Historical":-3,"Count":0},{"Historical":-2,"Count":1},{"Historical":-1,"Count":1},{"Historical":0,"Count":0},{"Historical":1,"Count":1},{"Historical":2,"Count":0},{"Historical":3,"Count":4},{"Historical":4,"Count":6},{"Historical":5,"Count":1},{"Historical":6,"Count":4},{"Historical":7,"Count":2},{"Historical":8,"Count":5},{"Historical":9,"Count":7},{"Historical":10,"Count":3},{"Historical":11,"Count":1},{"Historical":12,"Count":1},{"Historical":13,"Count":0},{"Historical":14,"Count":0},{"Historical":15,"Count":0},{"Historical":16,"Count":0},{"Historical":17,"Count":0},{"Historical":18,"Count":1},{"Historical":19,"Count":0},{"Historical":20,"Count":0}],"Avg_Profit_before_Tax":1234.48363123162,"Avg_Revenue_growth":22.3180426153235,"Avg_Revenue":11704.6134978605,"Targetprice_upside":0.246622656894583,"minmaxtable":[{"Metric":"Analys Prediction","Revenue Growth":22.3180426153223,"Net Profit Margin":7.31448737395571,"Gross Margin":70.1100673500095,"Target multiple":28,"Target Price upside":24.4343891402715},{"Metric":"Min (Simulation)","Revenue Growth":20.4222890641395,"Net Profit Margin":3.18964054144646,"Gross Margin":64.1633419557201,"Target multiple":28,"Target Price upside":-0.41553544494721},{"Metric":"Max (Simulation)","Revenue Growth":24.2137961665051,"Net Profit Margin":11.4404619705489,"Gross Margin":76.0498405626466,"Target multiple":28,"Target Price upside":0.941930618401207}],"gm_htable":[{"Historical":65,"Count":1},{"Historical":66,"Count":1},{"Historical":67,"Count":1},{"Historical":68,"Count":6},{"Historical":69,"Count":1},{"Historical":70,"Count":12},{"Historical":71,"Count":17},{"Historical":72,"Count":11},{"Historical":73,"Count":6},{"Historical":74,"Count":2},{"Historical":75,"Count":0}],"Avg_COGS":-3498.5010914521,"TPU_htable":[{"Historical":-44,"Count":46},{"Historical":-32,"Count":135},{"Historical":-20,"Count":172},{"Historical":-8,"Count":168},{"Historical":4,"Count":241},{"Historical":16,"Count":293},{"Historical":28,"Count":188},{"Historical":40,"Count":119},{"Historical":52,"Count":81},{"Historical":64,"Count":20},{"Historical":76,"Count":6},{"Historical":88,"Count":0}],"Avg_Net_profit_margin":7.31451188955846,"Avg_Target_Multiple":28,"Avg_Gross_Margin":70.110182486921}
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

  col21, col22, col23 = st.columns([1,1,1])
  with col21:
    DCRG_Avg_placeholder = st.empty()
  with col22:
    DCGM_Avg_placeholder = st.empty()
  with col23:  
    DCNPM_Avg_placeholder = st.empty()
  st.markdown('***')


  col14, col15, col16 = st.columns([1,1,1])
  with col14:
    TM_CHART_placeholder = st.empty()
  with col15:
    TPU_CHART_placeholder = st.empty()
  with col16:
    ADD_CHART_placeholder = st.empty()

  # col31, col32 = st.columns([1,1])
  # with col31:
  #   st.json(apiInput)
  # with col32: 
  #   st.json(DCoutputs) 

  col24, col25, col26 = st.columns([1,1,1])
  with col24:
    DCTM_Avg_placeholder = st.empty()
  with col25:
    DCTPU_Avg_placeholder = st.empty()
  with col26:  
    DCADD_placeholder = st.empty()

  st.markdown('***')
  col31, col32, col33, col34 = st.columns([1,1,1,1])
  with col31:
    DCNumberOfSimulations_placeholder = st.empty()
  with col32:
    DCAvgCost_placeholder = st.empty()
  with col33:  
    DCAvgProfit_placeholder = st.empty()
  with col34:  
    DCAvgRevenue_placeholder = st.empty()

  #generate line chart of results
  if not DCerrors:
    data_rg = pd.DataFrame(DCoutputs["rg_htable"])
    value_pairs_rg = {
      "Min": round(DCoutputs["minmaxtable"][1]["Revenue Growth"], 1),
      "Max": round(DCoutputs["minmaxtable"][2]["Revenue Growth"], 1),
      "Analyst Prediction": round(DCoutputs["minmaxtable"][0]["Revenue Growth"], 1)
    }
    chart_fig = generate_comb_chart(data_rg, value_pairs_rg, "Revenue Growth")
    RG_CHART_placeholder.pyplot(chart_fig)

    data_gm = pd.DataFrame(DCoutputs["gm_htable"])
    value_pairs_gm = {
      "Min": round(DCoutputs["minmaxtable"][1]["Gross Margin"], 1),
      "Max": round(DCoutputs["minmaxtable"][2]["Gross Margin"], 1),
      "Analyst Prediction": round(DCoutputs["minmaxtable"][0]["Gross Margin"], 1)
    }
    chart_fig = generate_comb_chart(data_gm, value_pairs_gm, "Gross Margin")
    GM_CHART_placeholder.pyplot(chart_fig)

    data_npm = pd.DataFrame(DCoutputs["npm_htable"])
    value_pairs_npm = {
      "Min": round(DCoutputs["minmaxtable"][1]["Net Profit Margin"], 2),
      "Max": round(DCoutputs["minmaxtable"][2]["Net Profit Margin"], 2),
      "Analyst Prediction": round(DCoutputs["minmaxtable"][0]["Net Profit Margin"], 2)
    }
    chart_fig = generate_comb_chart(data_npm, value_pairs_npm, "Net Profit Margin")
    NPM_CHART_placeholder.pyplot(chart_fig)

    data_tm = pd.DataFrame(DCoutputs["TM_htable"])
    value_pairs_tm = {
      "Min": round(DCoutputs["minmaxtable"][1]["Target multiple"], 2),
      "Max": round(DCoutputs["minmaxtable"][2]["Target multiple"], 2),
      "Analyst Prediction": round(DCoutputs["minmaxtable"][0]["Target multiple"], 2)
    }
    chart_fig = generate_comb_chart(data_tm, value_pairs_tm, "Target Multiple")
    TM_CHART_placeholder.pyplot(chart_fig)

    data_tpu = pd.DataFrame(DCoutputs["TPU_htable"])
    value_pairs_tpu = {
      "Min": round(DCoutputs["minmaxtable"][1]["Target Price upside"], 2),
      "Max": round(DCoutputs["minmaxtable"][2]["Target Price upside"], 2),
      "Analyst Prediction": round(DCoutputs["minmaxtable"][0]["Target Price upside"], 2)
    }
    chart_fig = generate_comb_chart(data_tpu, value_pairs_tpu, "Target Price Upside")
    TPU_CHART_placeholder.pyplot(chart_fig)

    # Chart value averages 
    DCRG_Avg = "{:,.2f}".format(DCoutputs["Avg_Revenue_growth"])
    DCRG_Avg_placeholder.info("**Avg Revenue Growth**: " + DCRG_Avg + " *(6561 Simulations)*")
    DCGM_Avg = "{:,.2f}".format(DCoutputs["Avg_Gross_Margin"])
    DCGM_Avg_placeholder.info("**Avg Gross Margin**: " + DCGM_Avg + " *(6561 Simulations)*")
    DCNPM_Avg = "{:,.2f}".format(DCoutputs["Avg_Net_profit_margin"])
    DCNPM_Avg_placeholder.info("**Avg Net Profit Margin**: " + DCNPM_Avg + " *(6561 Simulations)*")

    DCTM_Avg = "{:,.2f}".format(DCoutputs["Avg_Target_Multiple"])
    DCTM_Avg_placeholder.info("**Avg Target Multiple**: " + DCTM_Avg + " *(6561 Simulations)*")
    DCTPU_Avg = "{:,.2f}".format(DCoutputs["Targetprice_upside"])
    DCTPU_Avg_placeholder.info("**Avg TPU**: " + DCTPU_Avg + " *(6561 Simulations)*")

    DCNumberOfSimulations = "{:,.0f}".format(DCoutputs["Simualtions"])
    DCNumberOfSimulations_placeholder.metric(label='Number of Simulations', value=DCNumberOfSimulations)
    DCAvgCost = "{:,.2f}".format(DCoutputs["Avg_COGS"])
    DCAvgCost_placeholder.metric(label='Avg Cost of Goods ($)', value=DCAvgCost)
    DCAvgProfit = "{:,.2f}".format(DCoutputs["Avg_Profit_before_Tax"])
    DCAvgProfit_placeholder.metric(label='Avg Profit b.Tax ($)', value=DCAvgProfit)
    DCAvgRevenue = "{:,.2f}".format(DCoutputs["Avg_Revenue"])
    DCAvgRevenue_placeholder.metric(label='Avg Revenue ($)', value=DCAvgRevenue)

    initState = False
  else:
    error_messages = [error["message"] for error in DCerrors]
    if error_messages:
        ERRORBOX.error("\n ".join(error_messages))
