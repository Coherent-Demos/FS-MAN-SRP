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
            "SECTOR": SectorInput.split(" ⠀ ")[1],
            "REGION": RegionInput.split(" ⠀ ")[1]
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

@st.cache_data
def discoveryAPI():  

    payload = json.dumps({
       "request_data": {
          "inputs": {}
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
outputs = {"CompanyResults":[{"CompanyName":"MGM China","Region":"China","Sector":"Gaming","Simulations":256,"Gross Margin":0,"Net Margin":-0.181755533943624,"Revenue Growth":0.19509258296829,"Target Multilple":9.29849549401816,"Target Price":0.91295226285384,"Target Price (Upside)":0},{"CompanyName":"PICC","Region":"Korea","Sector":"F&B","Simulations":256,"Gross Margin":0.0268488180328546,"Net Margin":0.067738030217524,"Revenue Growth":0.083660118252909,"Target Multilple":1.1,"Target Price":12.2975599356256,"Target Price (Upside)":0},{"CompanyName":"Sungrow","Region":"Australia","Sector":"Healthcare","Simulations":256,"Gross Margin":0.702185551501273,"Net Margin":6.93832733065174,"Revenue Growth":0.260258180483239,"Target Multilple":16.0999999999999,"Target Price":111.6875,"Target Price (Upside)":0},{"CompanyName":"Yum China","Region":"China","Sector":"F&B","Simulations":256,"Gross Margin":63.7823130139502,"Net Margin":33.5897409027668,"Revenue Growth":22.3180426153228,"Target Multilple":0,"Target Price":0,"Target Price (Upside)":0.126950179110116},{"CompanyName":"CSL","Region":"Australia","Sector":"Healthcare","Simulations":256,"Gross Margin":0,"Net Margin":-0.0341486679423975,"Revenue Growth":0.218057636557498,"Target Multilple":20.2916895550052,"Target Price":0.968889597682799,"Target Price (Upside)":0},{"CompanyName":"CIMB","Region":"Malaysia","Sector":"Banks","Simulations":256,"Gross Margin":0.0209244090164282,"Net Margin":0.0679660630439283,"Revenue Growth":0.083660118252909,"Target Multilple":1.1,"Target Price":12.3035930377106,"Target Price (Upside)":0},{"CompanyName":"Grab","Region":"Singapore","Sector":"Transport","Simulations":256,"Gross Margin":0.591072183635433,"Net Margin":7.48244144706627,"Revenue Growth":0.422713744051762,"Target Multilple":16.0999999999999,"Target Price":120.4375,"Target Price (Upside)":0},{"CompanyName":"MediaTek","Region":"Taiwan","Sector":"Tech","Simulations":256,"Gross Margin":85.2715823577931,"Net Margin":50.8860013643354,"Revenue Growth":22.8733767223018,"Target Multilple":0,"Target Price":0,"Target Price (Upside)":-0.106953478506799}],"noOfCompanies_filtered":8}
DCerrors = []
discoveryData = {"listOfCompanies":[{"List of Companies":"MGM China"},{"List of Companies":"PICC"},{"List of Companies":"Sungrow"},{"List of Companies":"Yum China"},{"List of Companies":"CSL"},{"List of Companies":"CIMB"},{"List of Companies":"Grab"},{"List of Companies":"MediaTek"}],"listOfRegions":[{"List of Regions":"China"},{"List of Regions":"Korea"},{"List of Regions":"Australia"},{"List of Regions":"Malaysia"},{"List of Regions":"Singapore"},{"List of Regions":"Taiwan"}],"listOfSectors":[{"List of Sectors":"Gaming"},{"List of Sectors":"F&B"},{"List of Sectors":"Healthcare"},{"List of Sectors":"Banks"},{"List of Sectors":"Transport"},{"List of Sectors":"Tech"}],"Model_Inputs":[{"Model Inputs":"Mgmcotai Massdropgrowth","CURR":0.7,"NEXT":0.25},{"Model Inputs":"Mgmcotai Vipturnovergrowth","CURR":1.15,"NEXT":0.1},{"Model Inputs":"Mgmmacau Massdropgrowth","CURR":0.51,"NEXT":0.23},{"Model Inputs":"Mgmmacau Vipturnovergrowth","CURR":0.24,"NEXT":0.12}],"Model_Mapping":[{"INPUTS":"FQ0_MGMcotai_Massdropgrowth","OUTPUTS":"net_profit_margin"},{"INPUTS":"FQ0_MGMcotai_VIPturnovergrowth","OUTPUTS":"revenue_growth"},{"INPUTS":"FQ0_MGMmacau_Massdropgrowth","OUTPUTS":"target_multiple"},{"INPUTS":"FQ0_MGMmacau_VIPturnovergrowth","OUTPUTS":"target_price"},{"INPUTS":"FQ1_MGMcotai_Massdropgrowth","OUTPUTS":"historicaldata"},{"INPUTS":"FQ1_MGMcotai_VIPturnovergrowth","OUTPUTS":""},{"INPUTS":"FQ1_MGMmacau_Massdropgrowth","OUTPUTS":""},{"INPUTS":"FQ1_MGMmacau_VIPturnovergrowth","OUTPUTS":""}],"Model_Outputs":[{"Model Outputs":"Net Profit Margin"},{"Model Outputs":"Revenue Growth"},{"Model Outputs":"Target Multiple"},{"Model Outputs":"Target Price"},{"Model Outputs":"Historicaldata"}],"Region":"China","Sector":"Gaming","Input Frequency":"FQ","Spark Service":"Company models/MGM China Model_20240315","Logo URL":"https://www.google.com/search?sca_esv=aab80bd44fbfc9fb&sca_upv=1&rlz=1C1GCEA_enHK973HK973&sxsrf=ACQVn08ECl4eOVZxTmn_ulGXAKgx9BvaHg:1711350191071&q=mgm+china&tbm=isch&source=lnms&prmd=nivmsbtz&sa=X&ved=2ahUKEwjyl-ah7I6FAxWg0jQHHVSBBJ0Q0pQJegQIFhAB&biw=2560&bih=1225&dpr=0.75#imgrc=SxDwMHDDJOruCM","No of Companies":8,"No of Regions":6,"No of Sectors":6}

with st.expander("Spark Model", expanded=True):
  st.markdown('[https://spark.uat.jp.coherent.global/clsa/products/Aggregate%20Models/Output%20Analysis%20-%20by%20Sector%20&%20Region/apiTester/test](https://spark.uat.jp.coherent.global/clsa/products/Aggregate%20Models/Output%20Analysis%20-%20by%20Sector%20&%20Region/apiTester/test)')

#Call discoveryAPI
response = discoveryAPI()

# Parse the JSON response
discoveryData = response.json()['response_data']['outputs']
update_data = discoveryData.get("Model_Inputs")
   
with st.form("DC Form"):
  
  Go = True
  ERRORBOX = st.empty()
  DCLoading = st.empty()

  col01, col02, col03, col04 = st.columns([1, 1, 1, 1])
  with col01:
    list_of_sectors_data = discoveryData.get("listOfSectors")
    if list_of_sectors_data:
      SectorOptions = ["ALL"] + [f"{item.get('Icon', '')} ⠀ {item['List of Sectors']}" for item in list_of_sectors_data]
      SectorInput = st.selectbox("Sector", SectorOptions)

  with col02:
    list_of_regions_data = discoveryData.get("listOfRegions")
    if list_of_regions_data:
      RegionOptions = ["ALL"] + [f"{item.get('Icon', '')} ⠀ {item['List of Regions']}" for item in list_of_regions_data]
      RegionInput = st.selectbox("Region", RegionOptions)

  with col03:
    st.text("‎")

  with col04:
    st.text("‎")

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
  SummaryOfCompanies_Df_placeholder = st.empty()

  NumCompanies_Metric_placeholder.metric("Number of Companies Filtered", outputs['noOfCompanies_filtered'])
  
  SummaryOfCompanies_Df = pd.DataFrame(outputs['CompanyResults'])
  
  # Create the mapping
  company_logos = {company["List of Companies"]: company["Logo"] for company in discoveryData["listOfCompanies"]}

  # Add the "Logo" column based on "List of Companies"
  SummaryOfCompanies_Df['Logo'] = SummaryOfCompanies_Df['CompanyName'].map(company_logos)

  # Make "Logo" the first column
  SummaryOfCompanies_Df.insert(0, 'Logo', SummaryOfCompanies_Df.pop('Logo'))

  SummaryOfCompanies_Df_placeholder.data_editor(
    SummaryOfCompanies_Df,
    use_container_width=True,
    hide_index=True,
    column_config={
      "Logo": st.column_config.ImageColumn("", width=200, help="Streamlit app preview screenshots")
    }
  )
