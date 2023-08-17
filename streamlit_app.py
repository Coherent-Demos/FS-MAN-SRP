import streamlit as st
import requests
import json
import pandas as pd
import datetime
import time

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt

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
    plt.rcParams["font.family"] = ["Source Sans Pro", "Arial"]

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
    "2023 Min": -0.0042,
    "2023 Max": 0.0242,
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
    "2023 Min": 0.0023,
    "2023 Max": 0.0377,
    "2024 Point assumption": 0.02,
    "2024 Min": 0.0023,
    "2024 Max": 0.0377
  }
]

st.write("Edit Min and Max Values")
with st.form("DC Form"):
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
      apiInput = multiply_and_convert_to_json(inputTable)
      apiInput_dict = json.loads(apiInput)
      DCalldata = definedCombination(apiInput_dict)
      DCoutputs = DCalldata.json()['response_data']['outputs']

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
  apiInput = multiply_and_convert_to_json(inputTable)
  apiInput_dict = json.loads(apiInput)
  DCalldata = definedCombination(apiInput_dict)
  DCoutputs = DCalldata.json()['response_data']['outputs']

  col11, col12, col13 = st.columns([1,1,1])
  with col11:
    RG_CHART_placeholder = st.empty()
  with col12:
    GM_CHART_placeholder = st.empty()
  with col13:
    NPM_CHART_placeholder = st.empty()

  st.markdown('***')
  col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1])
  with col1:
    DCNumberOfSimulations = "{:,.0f}".format(DCoutputs["Simualtions"])
    st.metric(label='Number of Simulations', value=DCNumberOfSimulations)
  with col2:
    DCAvgCost = "{:,.2f}".format(DCoutputs["Avg_COGS"])
    st.metric(label='Avg Cost of Goods ($)', value=DCAvgCost)
  with col3:  
    DCAvgProfit = "{:,.2f}".format(DCoutputs["Avg_Profit_before_Tax"])
    st.metric(label='Avg Profit b.Tax ($)', value=DCAvgProfit)
  with col4:  
    DCAvgRevenue = "{:,.2f}".format(DCoutputs["Avg_Revenue"])
    st.metric(label='Avg Revenue ($)', value=DCAvgRevenue)
  with col5:  
    DCAvgTargetPrice = "{:,.2f}".format(DCoutputs["Avg_Target_Price"])
    st.metric(label='Avg Target Price ($)', value=DCAvgTargetPrice)

  # col31, col32 = st.columns([1,1])
  # with col31:
  #   st.json(apiInput)
  # with col32: 
  #   st.json(DCoutputs) 

  #generate line chart of results
  data_rg = pd.DataFrame(DCoutputs["rg_htable"])
  value_pairs_rg = {
    "Min": DCoutputs["minmaxtable"][1]["Revenue Growth"],
    "Max": DCoutputs["minmaxtable"][2]["Revenue Growth"],
    "Analyst Prediction": DCoutputs["minmaxtable"][0]["Revenue Growth"]
  }
  chart_fig = generate_comb_chart(data_rg, value_pairs_rg, "Revenue Growth")
  RG_CHART_placeholder.pyplot(chart_fig)

  data_gm = pd.DataFrame(DCoutputs["gm_htable"])
  value_pairs_gm = {
    "Min": DCoutputs["minmaxtable"][1]["Gross Margin"],
    "Max": DCoutputs["minmaxtable"][2]["Gross Margin"],
    "Analyst Prediction": DCoutputs["minmaxtable"][0]["Gross Margin"]
  }
  chart_fig = generate_comb_chart(data_gm, value_pairs_gm, "Gross Margin")
  GM_CHART_placeholder.pyplot(chart_fig)

  data_npm = pd.DataFrame(DCoutputs["npm_htable"])
  value_pairs_npm = {
    "Min": DCoutputs["minmaxtable"][1]["Net Profit Margin"],
    "Max": DCoutputs["minmaxtable"][2]["Net Profit Margin"],
    "Analyst Prediction": DCoutputs["minmaxtable"][0]["Net Profit Margin"]
  }
  chart_fig = generate_comb_chart(data_npm, value_pairs_npm, "Net Profit Margin")
  NPM_CHART_placeholder.pyplot(chart_fig)

