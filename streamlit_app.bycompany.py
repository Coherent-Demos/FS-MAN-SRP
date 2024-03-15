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
def definedCombination(inputdata):
    if 'DCloading' in st.session_state:
      DCloading.warning("Running Simulations")    

    payload = json.dumps({
       "request_data": {
          "inputs": {
            "Base_Inputs": inputdata,
            "CompanyName": selected_option, 
            "Range_inSD": int(RangeinSD)
          }
       },
        "request_meta": {
          "call_purpose": "Spark - API Tester",
          "source_system": "SPARK",
          "service_category": ""
        }
      })

    # st.json(payload)

    url = "https://excel.uat.jp.coherent.global/clsa/api/v3/folders/Original%20Models/services/Aggregate%20Output%20-%20v2/execute"
    headers = {
       'Content-Type': 'application/json',
       'x-tenant-name': 'clsa',
       'x-synthetic-key': '3ca9da18-31fa-4a82-a9ba-44130dff5c6a'
    }

    # url = "https://excel.uat.jp.coherent.global/clsa/api/v3/folders/Trial/services/Xcall Yum China - Defined Comb - output template3B/Execute"
    # headers = {
    #    'Content-Type': 'application/json',
    #    'x-tenant-name': 'clsa',
    #    'x-synthetic-key': '3ca9da18-31fa-4a82-a9ba-44130dff5c6a'
    # }

    # url = "https://excel.uat.us.coherent.global/coherent/api/v3/folders/Spark FE Demos/services/Xcall Yum China - Defined Comb - output template3B/Execute"
    # headers = {
    #    'Content-Type': 'application/json',
    #    'x-tenant-name': 'coherent',
    #    'SecretKey': '2277565c-9fad-4bf4-ad2b-1efe5748dd11'
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
image_path = "coherent-clsa-logo.png"
st.image(image_path, caption="")

st.write("## Pricing Simulation")
st.text("‎") 


#initialize data
json_data = [
  {
    "INPUTS": "INPUT 1",
    "Historical 1 SD": 1.5033853094776,
    "CURR - BASE": 0.24,
    "CURR - MIN": -1.2633853094776,
    "CURR - MAX": 1.7433853094776,
    "NEXT - BASE": 0.12,
    "NEXT - MIN": -1.3833853094776,
    "NEXT - MAX": 1.6233853094776
  },
  {
    "INPUTS": "INPUT 2",
    "Historical 1 SD": 3.15565794909837,
    "CURR - BASE": 0.51,
    "CURR - MIN": -2.64565794909837,
    "CURR - MAX": 3.66565794909837,
    "NEXT - BASE": 0.23,
    "NEXT - MIN": -2.92565794909837,
    "NEXT - MAX": 3.38565794909837
  },
  {
    "INPUTS": "INPUT 3",
    "Historical 1 SD": 4.07886347775884,
    "CURR - BASE": 1.15,
    "CURR - MIN": -2.92886347775884,
    "CURR - MAX": 5.22886347775884,
    "NEXT - BASE": 0.1,
    "NEXT - MIN": -3.97886347775884,
    "NEXT - MAX": 4.17886347775884
  },
  {
    "INPUTS": "INPUT 4",
    "Historical 1 SD": 8.21102123851926,
    "CURR - BASE": 0.7,
    "CURR - MIN": -7.51102123851926,
    "CURR - MAX": 8.91102123851926,
    "NEXT - BASE": 0.25,
    "NEXT - MIN": -7.96102123851926,
    "NEXT - MAX": 8.46102123851926
  }
]

# Spark_outputs = {"TM_htable":[{"Historical":16,"Count":11},{"Historical":18,"Count":86},{"Historical":20,"Count":138},{"Historical":22,"Count":349},{"Historical":24,"Count":342},{"Historical":26,"Count":151},{"Historical":28,"Count":270},{"Historical":30,"Count":213},{"Historical":32,"Count":91},{"Historical":34,"Count":13},{"Historical":36,"Count":26},{"Historical":38,"Count":11},{"Historical":40,"Count":11},{"Historical":42,"Count":8},{"Historical":44,"Count":0}],"Simualtions":6561,"rg_htable":[{"Historical":-23,"Count":0},{"Historical":-20,"Count":1},{"Historical":-17,"Count":0},{"Historical":-14,"Count":1},{"Historical":-11,"Count":2},{"Historical":-8,"Count":1},{"Historical":-5,"Count":3},{"Historical":-2,"Count":3},{"Historical":1,"Count":9},{"Historical":4,"Count":7},{"Historical":7,"Count":1},{"Historical":10,"Count":2},{"Historical":13,"Count":4},{"Historical":16,"Count":0},{"Historical":19,"Count":0},{"Historical":22,"Count":0},{"Historical":25,"Count":0},{"Historical":28,"Count":2},{"Historical":31,"Count":0},{"Historical":34,"Count":1},{"Historical":37,"Count":0},{"Historical":40,"Count":0},{"Historical":43,"Count":1},{"Historical":46,"Count":0}],"npm_htable":[{"Historical":-5,"Count":0},{"Historical":-4,"Count":0},{"Historical":-3,"Count":0},{"Historical":-2,"Count":1},{"Historical":-1,"Count":1},{"Historical":0,"Count":0},{"Historical":1,"Count":1},{"Historical":2,"Count":0},{"Historical":3,"Count":4},{"Historical":4,"Count":6},{"Historical":5,"Count":1},{"Historical":6,"Count":4},{"Historical":7,"Count":2},{"Historical":8,"Count":5},{"Historical":9,"Count":7},{"Historical":10,"Count":3},{"Historical":11,"Count":1},{"Historical":12,"Count":1},{"Historical":13,"Count":0},{"Historical":14,"Count":0},{"Historical":15,"Count":0},{"Historical":16,"Count":0},{"Historical":17,"Count":0},{"Historical":18,"Count":1},{"Historical":19,"Count":0},{"Historical":20,"Count":0}],"Avg_Profit_before_Tax":1234.48363123162,"Avg_Revenue_growth":22.3180426153235,"Avg_Revenue":11704.6134978605,"Targetprice_upside":0.247315194704849,"minmaxtable":[{"Metric":"Analys Prediction","Revenue Growth":22.3180426153223,"Net Profit Margin":7.31448737395571,"Gross Margin":70.1100673500095,"Target multiple":28,"Target Price upside":24.4343891402715},{"Metric":"Min (Simulation)","Revenue Growth":20.4222890641395,"Net Profit Margin":3.18964054144646,"Gross Margin":64.1633419557201,"Target multiple":28,"Target Price upside":-41.553544494721},{"Metric":"Max (Simulation)","Revenue Growth":24.2137961665051,"Net Profit Margin":11.4404619705489,"Gross Margin":76.0498405626466,"Target multiple":28,"Target Price upside":96.078431372549}],"gm_htable":[{"Historical":65,"Count":1},{"Historical":66,"Count":1},{"Historical":67,"Count":1},{"Historical":68,"Count":6},{"Historical":69,"Count":1},{"Historical":70,"Count":12},{"Historical":71,"Count":17},{"Historical":72,"Count":11},{"Historical":73,"Count":6},{"Historical":74,"Count":2},{"Historical":75,"Count":0}],"Avg_COGS":-3498.5010914521,"TPU_htable":[{"Historical":-44,"Count":46},{"Historical":-32,"Count":135},{"Historical":-20,"Count":172},{"Historical":-8,"Count":168},{"Historical":4,"Count":241},{"Historical":16,"Count":293},{"Historical":28,"Count":188},{"Historical":40,"Count":119},{"Historical":52,"Count":81},{"Historical":64,"Count":20},{"Historical":76,"Count":6},{"Historical":88,"Count":0}],"Avg_Net_profit_margin":7.31451188955846,"Avg_Target_Multiple":28,"Avg_Gross_Margin":70.110182486921}
Spark_outputs = {"gm_htable":[{"Historical":0,"Count":0},{"Historical":0,"Count":0},{"Historical":0,"Count":0},{"Historical":0,"Count":0},{"Historical":0,"Count":0},{"Historical":0,"Count":0},{"Historical":0,"Count":0},{"Historical":0,"Count":0},{"Historical":0,"Count":0},{"Historical":0,"Count":0},{"Historical":0,"Count":0}],"listOfOutputs":[{"DS OUTPUTS":"COGS"},{"DS OUTPUTS":"Gross_Margin"},{"DS OUTPUTS":"Net_profit_margin"},{"DS OUTPUTS":"profitbeforetax"},{"DS OUTPUTS":"revenue"},{"DS OUTPUTS":"Revenue_growth"},{"DS OUTPUTS":"target_multiple"},{"DS OUTPUTS":"targetprice"},{"DS OUTPUTS":"targetprice_upside"}],"minmaxtable":[{"Metric":"Analys Prediction","Revenue Growth":350.540556197299,"Net Profit Margin":12.0240055813407,"Gross Margin":0,"Target multiple":16.8226055718116,"Target Price upside":91.2929594181175},{"Metric":"Min (Simulation)","Revenue Growth":-168785.246569999,"Net Profit Margin":-195750.126172926,"Gross Margin":-195750.126172926,"Target multiple":-1816.8112330525,"Target Price upside":-168785.246569999},{"Metric":"Max (Simulation)","Revenue Growth":149444.938250861,"Net Profit Margin":149444.938250861,"Gross Margin":149444.938250861,"Target multiple":1275.86504996688,"Target Price upside":149444.938250861}],"npm_htable":[{"Historical":-110,"Count":0},{"Historical":-104,"Count":2},{"Historical":-98,"Count":1},{"Historical":-92,"Count":0},{"Historical":-86,"Count":0},{"Historical":-80,"Count":0},{"Historical":-74,"Count":0},{"Historical":-68,"Count":0},{"Historical":-62,"Count":0},{"Historical":-56,"Count":0},{"Historical":-50,"Count":0},{"Historical":-44,"Count":1},{"Historical":-38,"Count":0},{"Historical":-32,"Count":2},{"Historical":-26,"Count":0},{"Historical":-20,"Count":0},{"Historical":-14,"Count":0},{"Historical":-8,"Count":2},{"Historical":-2,"Count":1},{"Historical":4,"Count":5},{"Historical":10,"Count":8},{"Historical":16,"Count":13},{"Historical":22,"Count":5},{"Historical":28,"Count":0},{"Historical":34,"Count":0},{"Historical":40,"Count":0}],"rg_htable":[{"Historical":-79,"Count":1},{"Historical":-58,"Count":1},{"Historical":-37,"Count":3},{"Historical":-16,"Count":11},{"Historical":5,"Count":17},{"Historical":26,"Count":1},{"Historical":47,"Count":2},{"Historical":68,"Count":2},{"Historical":89,"Count":0},{"Historical":110,"Count":0},{"Historical":131,"Count":0},{"Historical":152,"Count":0},{"Historical":173,"Count":0},{"Historical":194,"Count":0},{"Historical":215,"Count":0},{"Historical":236,"Count":0},{"Historical":257,"Count":0},{"Historical":278,"Count":0},{"Historical":299,"Count":0},{"Historical":320,"Count":0},{"Historical":341,"Count":1},{"Historical":362,"Count":0},{"Historical":383,"Count":0},{"Historical":404,"Count":0}],"Simualtions":256,"TM_htable":[{"Historical":8,"Count":997},{"Historical":62,"Count":17},{"Historical":116,"Count":18},{"Historical":170,"Count":0},{"Historical":224,"Count":0},{"Historical":278,"Count":0},{"Historical":332,"Count":0},{"Historical":386,"Count":0},{"Historical":440,"Count":2},{"Historical":494,"Count":1},{"Historical":548,"Count":2},{"Historical":602,"Count":0},{"Historical":656,"Count":0},{"Historical":710,"Count":0},{"Historical":764,"Count":0}],"TPU_htable":[{"Historical":-58,"Count":275},{"Historical":-37,"Count":181},{"Historical":-16,"Count":106},{"Historical":5,"Count":132},{"Historical":26,"Count":286},{"Historical":47,"Count":230},{"Historical":68,"Count":118},{"Historical":89,"Count":78},{"Historical":110,"Count":48},{"Historical":131,"Count":14},{"Historical":152,"Count":1},{"Historical":173,"Count":0}],"COGS":-7498.04230986393,"Gross Margin":-26.5211597935216,"Net Margin":-63.3687284735114,"Profit bf Tax":-345.873183328916,"Revenue":17711.224543644,"Revenue Growth":85.0896075205776,"Target Multilple":27.9991591091745,"Target Price":55.859375,"Target Price (Upside)":0.0531556372549198}
DCerrors = []

with st.expander("Spark Model", expanded=True):
  # st.markdown('[https://spark.uat.jp.coherent.global/clsa/products/Original%20Models/Aggregate%20Output%20-%20v2/apiTester/test](https://spark.uat.jp.coherent.global/clsa/products/Original%20Models/Aggregate%20Output%20-%20v2/apiTester/test)')
  st.markdown('[https://spark.uat.jp.coherent.global/clsa/products/Original%20Models/Output%20Analysis%20-%20by%20Company/apiTester/test](https://spark.uat.jp.coherent.global/clsa/products/Original%20Models/Output%20Analysis%20-%20by%20Company/apiTester/test)')

st.write("Select Parameters")
with st.form("DC Form"):
  
  Go = True
  ERRORBOX = st.empty()
  DCLoading = st.empty()

  col01, col02 = st.columns([1,1])
  with col01:
    options = ["Yum China", "MGM China", "PICC", "Sungrow"]
    selected_option = st.selectbox("Select a Company", options)

  with col02:
    # create input text box below (not text area)    
    options = ["1", "2", "3", "4"]
    RangeinSD = st.selectbox("Select SD", options)

  st.markdown('***')

  def highlight_col(x):
    r = 'background-color: #fafafa; color: #a0a0a0'
    df1 = pd.DataFrame('', index=x.index, columns=x.columns)
    df1.iloc[:, 1:] = r
    return df1  
  
  # Create a DataFrame from JSON data
  df = pd.DataFrame(json_data)
  numeric_columns = df.select_dtypes(include=[float, int]).columns
  # df[numeric_columns] = df[numeric_columns]
  # df[numeric_columns] = df[numeric_columns].applymap('{:.1f}'.format)

  inputTable = st.data_editor(
    df.style.apply(highlight_col, axis=None),
    use_container_width=True,
    column_config={
      "INPUTS": st.column_config.Column(disabled=True),
      "Historical 1 SD": st.column_config.Column("Historical 1 SD", disabled=True),
      "CURR - BASE": st.column_config.Column("CURR - BASE", disabled=True),
      "NEXT - BASE": st.column_config.Column("NEXT - BASE", disabled=True),
      "CURR - MIN": st.column_config.Column("CURR - MIN"),
      "CURR -  MAX": st.column_config.Column("CURR -  MAX"),
      "NEXT - MIN": st.column_config.Column("NEXT - MIN"),
      "NEXT - MAX": st.column_config.Column("NEXT - MAX")
    }
  )

  DCbutton_clicked = st.form_submit_button("Generate Output")
  if DCbutton_clicked:   
    for index, row in inputTable.iterrows():
        if row["CURR - MIN"] > row["CURR - MAX"] or row["NEXT - MIN"] > row["NEXT - MAX"]:
          APIGO = False
          break
        else:
          APIGO = True
    if True:      
    # if APIGO==True:
      apiInput = multiply_and_convert_to_json(inputTable)
      apiInput_dict = json.loads(apiInput)
      DCalldata = definedCombination(apiInput_dict)
      Spark_outputs = DCalldata.json()['response_data']['outputs']
      # st.json(Spark_outputs)
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
  # Spark_outputs = DCalldata.json()['response_data']['outputs']

  col11, col12, col13 = st.columns([1,1,1])
  with col11:
    RG_CHART_placeholder = st.empty()
  with col12:
    # GM_CHART_placeholder = st.empty()
    TPU_CHART_placeholder = st.empty()
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
    # DCGM_Avg_placeholder = st.empty()
    DCTPU_Avg_placeholder = st.empty()
  with col23:  
    DCNPM_Avg_placeholder = st.empty()
  st.markdown('***')


  col14, col15, col16 = st.columns([1,1,1])
  with col14:
    TM_CHART_placeholder = st.empty()
  with col15:
    # TPU_CHART_placeholder = st.empty()
    ADD_CHART2_placeholder = st.empty()
  with col16:
    ADD_CHART_placeholder = st.empty()

  # col31, col32 = st.columns([1,1])
  # with col31:
  #   st.json(apiInput)
  # with col32: 
  #   st.json(Spark_outputs) 

  col24, col25, col26 = st.columns([1,1,1])
  with col24:
    DCTM_Avg_placeholder = st.empty()
  with col25:
    # DCTPU_Avg_placeholder = st.empty()
    DCADD2_placeholder = st.empty()
  with col26:  
    DCADD_placeholder = st.empty()

  st.markdown('***')
  col32, col33, col34 = st.columns([1,1,1])
  with col32:
    DCAvgCost_placeholder = st.empty()
  with col33:  
    DCAvgProfit_placeholder = st.empty()
  with col34:  
    DCAvgRevenue_placeholder = st.empty()

  #generate line chart of results
  if not DCerrors:
    data_rg = pd.DataFrame(Spark_outputs["rg_htable"])
    value_pairs_rg = {
      "Min": round(Spark_outputs["minmaxtable"][1]["Revenue Growth"], 1),
      "Max": round(Spark_outputs["minmaxtable"][2]["Revenue Growth"], 1),
      # "Analyst Prediction": round(Spark_outputs["minmaxtable"][0]["Revenue Growth"], 1)
      "Analyst Prediction": Spark_outputs["minmaxtable"][0]["Revenue Growth"]
    }
    chart_fig = generate_comb_chart(data_rg, value_pairs_rg, "Revenue Growth")
    RG_CHART_placeholder.pyplot(chart_fig)

    # data_gm = pd.DataFrame(Spark_outputs["gm_htable"])
    # value_pairs_gm = {
    #   "Min": round(Spark_outputs["minmaxtable"][1]["Gross Margin"], 1),
    #   "Max": round(Spark_outputs["minmaxtable"][2]["Gross Margin"], 1),
    #   # "Analyst Prediction": round(Spark_outputs["minmaxtable"][0]["Gross Margin"], 1)
    #   "Analyst Prediction": Spark_outputs["minmaxtable"][0]["Gross Margin"]
    # }
    # chart_fig = generate_comb_chart(data_gm, value_pairs_gm, "Gross Margin")
    # GM_CHART_placeholder.pyplot(chart_fig)

    data_npm = pd.DataFrame(Spark_outputs["npm_htable"])
    value_pairs_npm = {
      "Min": round(Spark_outputs["minmaxtable"][1]["Net Profit Margin"], 2),
      "Max": round(Spark_outputs["minmaxtable"][2]["Net Profit Margin"], 2),
      # "Analyst Prediction": round(Spark_outputs["minmaxtable"][0]["Net Profit Margin"], 2)
      "Analyst Prediction": Spark_outputs["minmaxtable"][0]["Net Profit Margin"]
    }
    chart_fig = generate_comb_chart(data_npm, value_pairs_npm, "Net Profit Margin")
    NPM_CHART_placeholder.pyplot(chart_fig)

    data_tm = pd.DataFrame(Spark_outputs["TM_htable"])
    value_pairs_tm = {
      "Min": round(Spark_outputs["minmaxtable"][1]["Target multiple"], 2),
      "Max": round(Spark_outputs["minmaxtable"][2]["Target multiple"], 2),
      # "Analyst Prediction": round(Spark_outputs["minmaxtable"][0]["Target multiple"], 2)
      "Analyst Prediction": Spark_outputs["minmaxtable"][0]["Target multiple"]
    }
    chart_fig = generate_comb_chart(data_tm, value_pairs_tm, "Target Multiple")
    TM_CHART_placeholder.pyplot(chart_fig)

    data_tpu = pd.DataFrame(Spark_outputs["TPU_htable"])
    value_pairs_tpu = {
      "Min": round(Spark_outputs["minmaxtable"][1]["Target Price upside"], 2),
      "Max": round(Spark_outputs["minmaxtable"][2]["Target Price upside"], 2),
      # "Analyst Prediction": round(Spark_outputs["minmaxtable"][0]["Target Price upside"], 2)
      "Analyst Prediction": Spark_outputs["minmaxtable"][0]["Target Price upside"]
    }
    chart_fig = generate_comb_chart(data_tpu, value_pairs_tpu, "Target Price Upside")
    TPU_CHART_placeholder.pyplot(chart_fig)

    # Function to format values safely
    def safe_format(value):
        try:
            # If the value is zero or can be converted to a float, format it.
            return "{:,.2f}".format(float(value))
        except (TypeError, ValueError):
            # If there is an error during formatting, return a default string
            return "N/A"

    # Chart value averages 
    DCRG_Avg = safe_format(Spark_outputs.get("Revenue Growth", "N/A"))
    DCRG_Avg_placeholder.info(f"**Avg Revenue Growth**: {DCRG_Avg}")

    DCNPM_Avg = safe_format(Spark_outputs.get("Net Margin", "N/A"))
    DCNPM_Avg_placeholder.info(f"**Avg Net Profit Margin**: {DCNPM_Avg}")

    DCTM_Avg = safe_format(Spark_outputs.get("Target Multiple", "N/A"))  # Fixed typo in "Multiple"
    DCTM_Avg_placeholder.info(f"**Avg Target Multiple**: {DCTM_Avg}")

    DCTPU_Avg = safe_format(Spark_outputs.get("Target Price", "N/A"))
    DCTPU_Avg_placeholder.info(f"**Avg TPU**: {DCTPU_Avg}")

    DCAvgCost = safe_format(Spark_outputs.get("COGS", "N/A"))
    DCAvgCost_placeholder.metric(label='Avg Cost of Goods ($)', value=DCAvgCost)

    DCAvgProfit = safe_format(Spark_outputs.get("Profit bf Tax", "N/A"))
    DCAvgProfit_placeholder.metric(label='Avg Profit b.Tax ($)', value=DCAvgProfit)

    DCAvgRevenue = safe_format(Spark_outputs.get("Revenue", "N/A"))
    DCAvgRevenue_placeholder.metric(label='Avg Revenue ($)', value=DCAvgRevenue)

    initState = False
  else:
    error_messages = [error["message"] for error in DCerrors]
    if error_messages:
        ERRORBOX.error("\n ".join(error_messages))
