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
            "CompanyName": selectedCompany
          }
       },
        "request_meta": {
          "call_purpose": "CLSA FE",
          "source_system": "SPARK",
          "service_category": ""
        }
      })

    # st.json(payload)

    url = "https://excel.uat.jp.coherent.global/clsa/api/v3/folders/Aggregate%20Models/services/Output%20Analysis%20-%20by%20Company/execute"
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

@st.cache_data
def discoveryAPI(selectedCompany):  

    payload = json.dumps({
       "request_data": {
          "inputs": {
            "Company": selectedCompany
          }
       },
        "request_meta": {
          "call_purpose": "CLSA FE",
          "source_system": "SPARK",
          "service_category": ""
        }
      })

    url = "https://excel.uat.jp.coherent.global/clsa/api/v3/folders/Aggregate%20Models/services/Analysis%20-%20CLSA%20Spark%20Services/execute"
    headers = {
       'Content-Type': 'application/json',
       'x-tenant-name': 'clsa',
       'x-synthetic-key': '3ca9da18-31fa-4a82-a9ba-44130dff5c6a'
    }

    response = requests.request("POST", url, headers=headers, data=payload, allow_redirects=False)
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
    # Check for None values in value_pairs
    if any(value is None for value in value_pairs.values()):
        # Create an empty chart with a message
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No Data Available', fontsize=12, ha='center')
        ax.set_xlabel(title)
        ax.set_ylabel("Count")
        ax.set_title(title + " Distribution")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        return fig
    
    # Continue with the original function if no None values are found
    df = pd.DataFrame(data)
    bar_width = (df["Historical"].max() - df["Historical"].min()) / len(df)
    fig, ax = plt.subplots()
    ax.bar(df["Historical"], df["Count"], width=bar_width, align='center', color='#020887', label="Original Data")

    min_value = value_pairs["Min"]
    max_value = value_pairs["Max"]
    ax.fill_betweenx(y=[ax.get_ylim()[0], ax.get_ylim()[1]], x1=min_value, x2=max_value, color='#f4d35e', alpha=0.7, label="Min-Max Range", zorder=0)

    for label, value in value_pairs.items():
        line_color = '#BA1B1D' if label == "Analyst Prediction" else '#f4d35e'
        line_width = 2 if label == "Analyst Prediction" else 1
        ax.axvline(x=value, color=line_color, linestyle='-', linewidth=line_width, label=f"{label}: {value}")

    ax.set_xlabel(title)
    ax.set_ylabel("Count")
    ax.set_title(title + " Distribution")
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
      # trim " %" from the end of the string
      modified_df[column] = modified_df[column].str.rstrip(" %")
      modified_df[column] = pd.to_numeric(modified_df[column], errors='coerce', downcast='integer') / 100

    # Convert the DataFrame to JSON format
    json_data = modified_df.to_json(orient='records')

    return json_data

#Start of UI
image_path = "coherent-clsa-logo.png"
st.image(image_path, caption="")

st.write("## Investment Analytics - By Company")
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

#initialize sample output
Spark_outputs = {"gm_htable":[{"Historical":0,"Count":0},{"Historical":0,"Count":0},{"Historical":0,"Count":0},{"Historical":0,"Count":0},{"Historical":0,"Count":0},{"Historical":0,"Count":0},{"Historical":0,"Count":0},{"Historical":0,"Count":0},{"Historical":0,"Count":0},{"Historical":0,"Count":0},{"Historical":0,"Count":0}],"listOfOutputs":[{"DS OUTPUTS":"net_profit_margin"},{"DS OUTPUTS":"revenue_growth"},{"DS OUTPUTS":"target_multiple"},{"DS OUTPUTS":"target_price"}],"minmaxtable":[{"Metric":"Analys Prediction","Revenue Growth":350.540556197299,"Net Profit Margin":12.0240055813407,"Gross Margin":"","Target multiple":16.8226055718116,"Target Price upside":91.2929594181175},{"Metric":"Min (Simulation)","Revenue Growth":-3.3931025420385,"Net Profit Margin":-838.675349051121,"Gross Margin":-838.675349051121,"Target multiple":-838.675349051121,"Target Price upside":-3.3931025420385},{"Metric":"Max (Simulation)","Revenue Growth":5.1129808598092,"Net Profit Margin":1089.80887713455,"Gross Margin":1089.80887713455,"Target multiple":127.252380327068,"Target Price upside":5.1129808598092}],"npm_htable":[{"Historical":-110,"Count":0},{"Historical":-104,"Count":2},{"Historical":-98,"Count":1},{"Historical":-92,"Count":0},{"Historical":-86,"Count":0},{"Historical":-80,"Count":0},{"Historical":-74,"Count":0},{"Historical":-68,"Count":0},{"Historical":-62,"Count":0},{"Historical":-56,"Count":0},{"Historical":-50,"Count":0},{"Historical":-44,"Count":1},{"Historical":-38,"Count":0},{"Historical":-32,"Count":2},{"Historical":-26,"Count":0},{"Historical":-20,"Count":0},{"Historical":-14,"Count":0},{"Historical":-8,"Count":2},{"Historical":-2,"Count":1},{"Historical":4,"Count":5},{"Historical":10,"Count":8},{"Historical":16,"Count":13},{"Historical":22,"Count":5},{"Historical":28,"Count":0},{"Historical":34,"Count":0},{"Historical":40,"Count":0}],"rg_htable":[{"Historical":-79,"Count":1},{"Historical":-58,"Count":1},{"Historical":-37,"Count":3},{"Historical":-16,"Count":11},{"Historical":5,"Count":17},{"Historical":26,"Count":1},{"Historical":47,"Count":2},{"Historical":68,"Count":2},{"Historical":89,"Count":0},{"Historical":110,"Count":0},{"Historical":131,"Count":0},{"Historical":152,"Count":0},{"Historical":173,"Count":0},{"Historical":194,"Count":0},{"Historical":215,"Count":0},{"Historical":236,"Count":0},{"Historical":257,"Count":0},{"Historical":278,"Count":0},{"Historical":299,"Count":0},{"Historical":320,"Count":0},{"Historical":341,"Count":1},{"Historical":362,"Count":0},{"Historical":383,"Count":0},{"Historical":404,"Count":0}],"Simualtions":256,"TM_htable":[{"Historical":8,"Count":997},{"Historical":62,"Count":17},{"Historical":116,"Count":18},{"Historical":170,"Count":0},{"Historical":224,"Count":0},{"Historical":278,"Count":0},{"Historical":332,"Count":0},{"Historical":386,"Count":0},{"Historical":440,"Count":2},{"Historical":494,"Count":1},{"Historical":548,"Count":2},{"Historical":602,"Count":0},{"Historical":656,"Count":0},{"Historical":710,"Count":0},{"Historical":764,"Count":0}],"TPU_htable":[{"Historical":-58,"Count":275},{"Historical":-37,"Count":181},{"Historical":-16,"Count":106},{"Historical":5,"Count":132},{"Historical":26,"Count":286},{"Historical":47,"Count":230},{"Historical":68,"Count":118},{"Historical":89,"Count":78},{"Historical":110,"Count":48},{"Historical":131,"Count":14},{"Historical":152,"Count":1},{"Historical":173,"Count":0}],"Gross Margin":0,"Net Margin":3.77899625339069,"Revenue Growth":0.179666874434118,"Target Multilple":9.79730203448272,"Target Price":0.859706677248583,"Target Price (Upside)":0}
discoveryData = {"listOfCompanies":[{"List of Companies":"MGM China"},{"List of Companies":"PICC"},{"List of Companies":"Sungrow"},{"List of Companies":"Yum China"},{"List of Companies":"CSL"},{"List of Companies":"CIMB"},{"List of Companies":"Grab"},{"List of Companies":"MediaTek"}],"listOfRegions":[{"List of Regions":"China"},{"List of Regions":"Korea"},{"List of Regions":"Australia"},{"List of Regions":"Malaysia"},{"List of Regions":"Singapore"},{"List of Regions":"Taiwan"}],"listOfSectors":[{"List of Sectors":"Gaming"},{"List of Sectors":"F&B"},{"List of Sectors":"Healthcare"},{"List of Sectors":"Banks"},{"List of Sectors":"Transport"},{"List of Sectors":"Tech"}],"Model_Inputs":[{"Model Inputs":"Mgmcotai Massdropgrowth","CURR":0.7,"NEXT":0.25},{"Model Inputs":"Mgmcotai Vipturnovergrowth","CURR":1.15,"NEXT":0.1},{"Model Inputs":"Mgmmacau Massdropgrowth","CURR":0.51,"NEXT":0.23},{"Model Inputs":"Mgmmacau Vipturnovergrowth","CURR":0.24,"NEXT":0.12}],"Model_Mapping":[{"INPUTS":"FQ0_MGMcotai_Massdropgrowth","OUTPUTS":"net_profit_margin"},{"INPUTS":"FQ0_MGMcotai_VIPturnovergrowth","OUTPUTS":"revenue_growth"},{"INPUTS":"FQ0_MGMmacau_Massdropgrowth","OUTPUTS":"target_multiple"},{"INPUTS":"FQ0_MGMmacau_VIPturnovergrowth","OUTPUTS":"target_price"},{"INPUTS":"FQ1_MGMcotai_Massdropgrowth","OUTPUTS":"historicaldata"},{"INPUTS":"FQ1_MGMcotai_VIPturnovergrowth","OUTPUTS":""},{"INPUTS":"FQ1_MGMmacau_Massdropgrowth","OUTPUTS":""},{"INPUTS":"FQ1_MGMmacau_VIPturnovergrowth","OUTPUTS":""}],"Model_Outputs":[{"Model Outputs":"Net Profit Margin"},{"Model Outputs":"Revenue Growth"},{"Model Outputs":"Target Multiple"},{"Model Outputs":"Target Price"},{"Model Outputs":"Historicaldata"}],"Region":"China","Sector":"Gaming","Input Frequency":"FQ","Spark Service":"Company models/MGM China Model_20240315","Logo URL":"https://www.google.com/search?sca_esv=aab80bd44fbfc9fb&sca_upv=1&rlz=1C1GCEA_enHK973HK973&sxsrf=ACQVn08ECl4eOVZxTmn_ulGXAKgx9BvaHg:1711350191071&q=mgm+china&tbm=isch&source=lnms&prmd=nivmsbtz&sa=X&ved=2ahUKEwjyl-ah7I6FAxWg0jQHHVSBBJ0Q0pQJegQIFhAB&biw=2560&bih=1225&dpr=0.75#imgrc=SxDwMHDDJOruCM","No of Companies":8,"No of Regions":6,"No of Sectors":6}
DCerrors = []
companyOptions = ['MGM China', 'PICC', 'Sungrow', 'Yum China', 'CSL', 'CIMB', 'Grab']
update_data = discoveryData.get("Model_Inputs")

with st.expander("Spark Model", expanded=True):
  st.markdown('[https://spark.uat.jp.coherent.global/clsa/products/Aggregate%20Models/Output%20Analysis%20-%20by%20Company/apiTester/test](https://spark.uat.jp.coherent.global/clsa/products/Aggregate%20Models/Output%20Analysis%20-%20by%20Company/apiTester/test)')

st.write("  ")

list_of_companies_data = discoveryData.get("listOfCompanies")
if list_of_companies_data:
  companyOptions = [item["List of Companies"] for item in list_of_companies_data]
colcompany01, colcompany02, colcompany03 = st.columns([1, 1, 1])
with colcompany01:  
  selectedCompany = st.selectbox("Company", companyOptions)
with colcompany02:
  st.write("  ")
with colcompany03:
  st.write("  ")
  

with st.form("DC Form"):
  Go = True
  ERRORBOX = st.empty()
  DCLoading = st.empty()

  st.write("Enter Min & Max Inputs")
  inputTableContainer = st.container()
  DCbutton_clickedContainer = st.container()

st.write("  ")
st.write("  ")
st.write("  ")

ResultsContainer = st.container()

# Call discoveryAPI
response = discoveryAPI(selectedCompany)

# Parse the JSON response
discoveryData = response.json()['response_data']['outputs']
update_data = discoveryData.get("Model_Inputs")

with inputTableContainer:
  for i in range(len(json_data)):
    json_data[i]["INPUTS"] = update_data[i]["Model Inputs"]  # Update INPUTS with Model Inputs
    json_data[i]["CURR - BASE"] = update_data[i]["CURR"]  # Update CURR - BASE with CURR
    json_data[i]["NEXT - BASE"] = update_data[i]["NEXT"]  # Update NEXT - BASE with NEXT

  def highlight_col(x):
    r = 'background-color: #fafafa; color: #909090'
    df1 = pd.DataFrame('', index=x.index, columns=x.columns)
    df1.loc[:, :] = r
    return df1  

  # Create a DataFrame from JSON data
  df = pd.DataFrame(json_data)
  all_numeric_columns = df.select_dtypes(include=[float, int]).columns

  columns_to_exclude = []
  numeric_columns = [col for col in all_numeric_columns if col not in columns_to_exclude]

  df[numeric_columns] = df[numeric_columns] * 100
  df[all_numeric_columns] = df[all_numeric_columns].applymap('{:.2f}'.format) + " %"

  inputTable = st.data_editor(
    df.style.apply(highlight_col, axis=None),
    use_container_width=True,
    hide_index=True,
    column_config={
      "INPUTS": st.column_config.Column("INPUTS", disabled=True),
      "Historical 1 SD": st.column_config.Column("Historical 1 SD", disabled=True),
      "CURR - BASE": st.column_config.Column("CURR - BASE", disabled=True),
      "NEXT - BASE": st.column_config.Column("NEXT - BASE", disabled=True),
      "CURR - MIN": st.column_config.Column("CURR - MIN"),
      "CURR -  MAX": st.column_config.Column("CURR -  MAX"),
      "NEXT - MIN": st.column_config.Column("NEXT - MIN"),
      "NEXT - MAX": st.column_config.Column("NEXT - MAX")
    }
  )

with DCbutton_clickedContainer:
  DCbutton_clicked = st.form_submit_button("Generate Output")
  if DCbutton_clicked: 
    apiInput = multiply_and_convert_to_json(inputTable)
    apiInput_dict = json.loads(apiInput)
    DCalldata = definedCombination(apiInput_dict)
    processingTime = DCalldata.json()['response_meta']['process_time']
    Spark_outputs = DCalldata.json()['response_data']['outputs']
    DCerrors = DCalldata.json()['response_data']['errors']

    with ResultsContainer:
      st.write(selectedCompany + " Info")
      with st.expander("", expanded=True):
        col30, col31, col32, col33, col34 = st.columns([0.1,1,1,1,1])
        with col30:
          st.write("  ")
        with col31:
          LOGO_placeholder = st.empty()
        with col32:  
          COMPANY_placeholder = st.empty()
        with col33:
          REGION_placeholder = st.empty()
        with col34:
          SECTOR_placeholder = st.empty()
        st.write("  ")

      
      colresults01, colresults02 = st.columns([1, 1])
      with colresults01:
        st.write(selectedCompany + " Results")
      with colresults02: 
        st.markdown(f"<p style='text-align: right;'>Processing Time: {processingTime} ms</p>", unsafe_allow_html=True)
      
      with st.expander("", expanded=True):

        col01, col02, col03, col04, col05, col06 = st.columns([1,1,1,1,1,1])
        with col01:
          GM_METRIC_placeholder = st.empty()
        with col02:
          NPM_METRIC_placeholder = st.empty()
        with col03:
          RG_METRIC_placeholder = st.empty()
        with col04:
          TM_METRIC_placeholder = st.empty()
        with col05:
          TP_METRIC_placeholder = st.empty()
        with col06:
          TPU_METRIC_placeholder = st.empty()

        st.markdown('***')

        col11, col12, col13 = st.columns([1,1,1])
        with col11:
          RG_CHART_placeholder = st.empty()
        with col12:
          NPM_CHART_placeholder = st.empty()
        with col13:
          TM_CHART_placeholder = st.empty()

        col14, col15, col16 = st.columns([1,1,1])
        with col14:
          TPU_CHART_placeholder = st.empty()
        with col15:
          # GM_CHART_placeholder = st.empty()
          ADD_CHART2_placeholder = st.empty()
        with col16:
          ADD_CHART_placeholder = st.empty()

      LOGO_placeholder.image(discoveryData.get("Logo URL"), caption="", width=200)
      COMPANY_placeholder.metric("Company", selectedCompany)
      REGION_placeholder.metric("Region", discoveryData.get("Region"))
      SECTOR_placeholder.metric("Sector", discoveryData.get("Sector"))

      # Chart value averages 
      RG_AVG = round(Spark_outputs["Revenue Growth"] * 100, 2)
      RG_METRIC_placeholder.metric(label='Revenue Growth (%)', value=f"{RG_AVG}%" if RG_AVG != 0 else "N/A") 

      GM_AVG = round(Spark_outputs["Gross Margin"] * 100, 1)
      GM_METRIC_placeholder.metric(label='Gross Margin (%)', value=f"{GM_AVG}%" if GM_AVG != 0 else "N/A")

      NPM_AVG = round(Spark_outputs["Net Margin"] * 100, 2)
      NPM_METRIC_placeholder.metric(label='Net Margin (%)', value=f"{NPM_AVG}%" if NPM_AVG != 0 else "N/A")

      TM_AVG = round(Spark_outputs["Target Multilple"], 2)
      TM_METRIC_placeholder.metric(label='Target Multiple (x)', value=str(TM_AVG) if TM_AVG != 0 else "N/A")

      TP_AVG = round(Spark_outputs["Target Price"], 2)
      TP_METRIC_placeholder.metric(label='Target Price ($)', value=str(TP_AVG) if TP_AVG != 0 else "N/A")

      TPU_AVG = round(Spark_outputs["Target Price (Upside)"], 2)
      TPU_METRIC_placeholder.metric(label='Target Price Upside ($)', value=str(TPU_AVG) if TPU_AVG != 0 else "N/A")
      
      #generate line chart of results
      if not DCerrors:
        data_rg = pd.DataFrame(Spark_outputs["rg_htable"])
        value_pairs_rg = {
          "Min": round(Spark_outputs["minmaxtable"][1]["Revenue Growth"], 1),
          "Max": round(Spark_outputs["minmaxtable"][2]["Revenue Growth"], 1),
          "Analyst Prediction": Spark_outputs["minmaxtable"][0]["Revenue Growth"]
        }
        chart_fig = generate_comb_chart(data_rg, value_pairs_rg, "Revenue Growth")
        RG_CHART_placeholder.pyplot(chart_fig)

        data_npm = pd.DataFrame(Spark_outputs["npm_htable"])
        value_pairs_npm = {
          "Min": round(Spark_outputs["minmaxtable"][1]["Net Profit Margin"], 2),
          "Max": round(Spark_outputs["minmaxtable"][2]["Net Profit Margin"], 2),
          "Analyst Prediction": Spark_outputs["minmaxtable"][0]["Net Profit Margin"]
        }
        chart_fig = generate_comb_chart(data_npm, value_pairs_npm, "Net Profit Margin")
        NPM_CHART_placeholder.pyplot(chart_fig)

        data_tm = pd.DataFrame(Spark_outputs["TM_htable"])
        value_pairs_tm = {
          "Min": round(Spark_outputs["minmaxtable"][1]["Target multiple"], 2),
          "Max": round(Spark_outputs["minmaxtable"][2]["Target multiple"], 2),
          "Analyst Prediction": Spark_outputs["minmaxtable"][0]["Target multiple"]
        }
        chart_fig = generate_comb_chart(data_tm, value_pairs_tm, "Target Multiple")
        TM_CHART_placeholder.pyplot(chart_fig)

        data_tpu = pd.DataFrame(Spark_outputs["TPU_htable"])
        value_pairs_tpu = {
          "Min": round(Spark_outputs["minmaxtable"][1]["Target Price upside"], 2),
          "Max": round(Spark_outputs["minmaxtable"][2]["Target Price upside"], 2),
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

        initState = False
      else:
        error_messages = [error["message"] for error in DCerrors]
        if error_messages:
            ERRORBOX.error("\n ".join(error_messages))
  
# Add the style tag to change button color to blue
st.markdown("""
<style>
    .stButton button { /* Adjust the class name according to your button's class */
        background-color: blue !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)