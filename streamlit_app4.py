import streamlit as st
import requests
import json
import pandas as pd
import datetime
import time

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

st.set_page_config(layout="wide")

timeelapsed = 0

DCoutputs = {"Avg_COGS":0,"Avg_Profit_before_Tax":0,"Avg_Revenue":10,"Avg_Target_Price":0,"Simualtions":0}

def definedCombination(inputdata):
    if 'DCloading' in st.session_state:
        DCloading.warning("Running Simulations")
    
    url = "https://excel.uat.us.coherent.global/coherent/api/v3/folders/CLSA/services/Xcall Solution - Yum China - Defined Comb - v1/Execute"

    payload = json.dumps({
      "request_data": {
      "inputs": {
        "Base_Inputs": [
          {
          "#": "Base",
          "KFC_Cost_of_Sales": inputdata["DCKFCCost"],
          "KFC_Same_store_sales_growth": inputdata["DCKFCSSSG"],
          "Pizza_hut_Cost_of_Sales": inputdata["DCPHCost"],
          "Pizza_hut_Same_store_sales_growth": inputdata["DCPHSSSG"]
          }
        ],
        "Max_Deviation": inputdata["MAXDEV"],
        "Min_Deviation": inputdata["MINDEV"],
        "Step": inputdata["STEP"]
        }
      },
      "request_meta": {
      "compiler_type": "Neuron",
      "version_id": "a1a9b821-3d97-40cf-a159-555e3d7f3ce1"
      }
    })
    headers = {
       'Content-Type': 'application/json',
       'x-tenant-name': 'coherent',
       'SecretKey': '2277565c-9fad-4bf4-ad2b-1efe5748dd11'
    }

    start_time = time.time()  # Record the start time
    response = requests.request("POST", url, headers=headers, data=payload, allow_redirects=False)
    
    end_time = time.time()  # Record the end time
    elapsed_time_ms = round((end_time - start_time) * 1000, 2)  # Calculate elapsed time in milliseconds

    if 'DCloading' in st.session_state:
        DCloading.success(f"API call successful (Elapsed Time: {elapsed_time_ms} ms)")

    return response, elapsed_time_ms

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

#Start of UI
image_path = "coherent-logo.png"
st.image(image_path, caption="", width=32)

st.write("## Spark Model Test Machine - Pricing Simulation")

col21, col22, col23 = st.columns([12, 2, 32])
col21, col22, col23 = st.columns([12, 2, 32])

with col21:
  st.text("‎") 
  st.write("Simulation Controls")
  with st.form("DC Form"):
    with st.expander("**BASE**", expanded=True):
      col211, col212 = st.columns([1,1])
      with col211:
        DCKFCCost = st.number_input("KFC - Cost of Sales (%)", key="DCKFCCost", value=31.00)
        DCKFCSSSG = st.number_input("KFC - SSSG (%)", key="DCKFCSSSG", value=5.00)
      with col212:
        DCPHCost = st.number_input("Pizza Hut - Cost of Sales (%)", key="DCPHCost", value=31.00)
        DCPHSSSG = st.number_input("Pizza Hut - SSSG (%)", key="DCPHSSSG", value=7.00)
    with st.expander("**RANGE**", expanded=True):
      MAXDEV = st.number_input("Max Deviation (%)", key="MAXDEV", value=2.00)
      MINDEV = st.number_input("Min Deviation (%)", key="MINDEV", value=-2.00)
      STEP = st.number_input("Step (%)", key="STEP", value=1.00)   
    with st.expander("**Generate Output**", expanded=True):
      st.markdown(
            """
            <style>
            .stButton>button {
                background-color: blue;
                color: white;
            }
            .stButton>button:hover {
                color: white;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

      DCbutton_clicked = st.form_submit_button("Submit", use_container_width=True)
      if DCbutton_clicked:
        #API call 
        inputData2 = {
          "DCKFCCost": DCKFCCost / 100,
          "DCKFCSSSG": DCKFCSSSG / 100,
          "DCPHCost": DCPHCost / 100,
          "DCPHSSSG": DCPHSSSG / 100,
          "MAXDEV": MAXDEV / 100,
          "MINDEV": MINDEV / 100,
          "STEP": STEP / 100
        }
        DCalldata, timeelapsed = definedCombination(inputData2)
        DCoutputs = DCalldata.json()['response_data']['outputs']
        # df_DCsimresults = pd.DataFrame(DCoutputs["Sim_Results"])

        # add success alert after api call

with col22:
  st.text("‎") 

with col23:
  st.text("‎") 
  st.write("Simulation Results")

  with st.expander("**Illustration**", expanded=True):

    st.markdown('***')
    col321, col322, col323, col324, col325, col326 = st.columns([1,1,1,1,1,1])
    with col321:
      DCNumberOfSimulations_placeholder = st.empty()
    with col322:
      DCAvgCost_placeholder = st.empty()
    with col323:  
      DCAvgProfit_placeholder = st.empty()
    with col324:  
      DCAvgRevenue_placeholder = st.empty()
    with col325: 
      DCAvgTargetPrice_placeholder = st.empty() 
    with col326: 
      timeelapsed_placeholder = st.empty() 
    st.markdown('***')

timeelapsed_placeholder.metric(label='Time Elapsed (ms)', value=timeelapsed)
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