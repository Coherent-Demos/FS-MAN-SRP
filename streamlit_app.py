import streamlit as st
import requests
import json
import pandas as pd
import datetime

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

st.set_page_config(layout="wide")

@st.cache_data
def normalDistribution(inputData):

    url = "https://excel.uat.us.coherent.global/coherent/api/v3/folders/CLSA/services/Xcall Solution - Yum China - Normal Distn/Execute"

    payload = json.dumps({
       "request_data": {
          "inputs": {
            "Simualtions": inputData["NDNumberOfSimulations"],
            "Stat_Inputs": [
              {
                "Statistical Inputs": "Historical Mean",
                "KFC_Cost_of_Sales": inputData["NDKFCCostMean"],
                "KFC_Same_store_sales_growth": inputData["NDKFCGrowthMean"],
                "Pizza_hut_Cost_of_Sales": inputData["NDPHCostMean"],
                "Pizza_hut_Same_store_sales_growth": inputData["NDPHGrowthMean"]
              },
              {
                "Statistical Inputs": "Historical Stainputndard Deviation",
                "KFC_Cost_of_Sales": inputData["NDKFCCostDev"],
                "KFC_Same_store_sales_growth": inputData["NDKFCGrowthDev"],
                "Pizza_hut_Cost_of_Sales": inputData["NDPHCostDev"],
                "Pizza_hut_Same_store_sales_growth": inputData["NDPHGrowthDev"]
              }
            ]
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
    return response

@st.cache_data
def definedCombination(inputdata):

    url = "https://excel.uat.us.coherent.global/coherent/api/v3/folders/Spark FE Demos/services/loan-origination/Execute"

    payload = json.dumps({
       "request_data": {
          "inputs": {
            "Channel": inputdata['Channel'],
            "Dependants": inputdata['Dependants'],
            "DOB": inputdata['DOB'],
            "DurationOfLoan": inputdata['DurationOfLoan'],
            "Education": inputdata['Education'],
            "ExistingCustomer": inputdata['ExistingCustomer'],
            "Gender": inputdata['Gender'],
            "Home": inputdata['Home'],
            "Income": inputdata['Income'],
            "Living_Area": inputdata['Living_Area'],
            "LoanAmount": inputdata['LoanAmount'],
            "LoanStart": inputdata['LoanStart'],
            "NationalID": inputdata['NationalID'],
            "Nationality": inputdata['Nationality'],
            "Occupation": inputdata['Occupation']
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
    return response

def generate_line_chart(fig, data_df, config):
    for column in data_df.columns:
        if column != config['x_column']:
            fig.add_trace(go.Scatter(
                x=data_df[config['x_column']],
                y=data_df[column],
                mode='lines',
                name=column
            ))
    fig.update_layout(title=config['title'])

#Start of UI
image_path = "coherent-logo.png"
st.image(image_path, caption="", width=32)

st.write("## Pricing Simulation")

tab1, tab2 = st.tabs(['Normal Distribution', 'Defined Combination'])

with tab1:
  col21, col22, col23 = st.columns([12, 2, 32])

  with col21:
    st.text("‎") 
    st.write("Simulation Controls")
    with st.expander("**General**", expanded=True):
      NDNumberOfSimulations = st.number_input("Number of Simulations", key="NDNumberOfSimulations", value=10, max_value=10)
    with st.expander("**KFC - Cost of Sales**", expanded=True):
      col211, col212 = st.columns([1,1])
      with col211:
        NDKFCCostMean = st.number_input("Historical Mean (%)", key="NDKFCCostMean", value=31.65)
      with col212:
        NDKFCCostDev = st.number_input("Historical Deviation (%)", key="NDKFCCostDev", value=1.42)
    with st.expander("**KFC - Cost of Sales**", expanded=True):
      col211, col212 = st.columns([1,1])
      with col211:
        NDKFCGrowthMean = st.number_input("Historical Mean (%)", key="NDKFCGrowthMean", value=-2.8)
      with col212:
        NDKFCGrowthDev = st.number_input("Historical Deviation (%)", key="NDKFCGrowthDev", value=6.21)
    with st.expander("**Pizza Hut - Cost of Sales**", expanded=True):
      col211, col212 = st.columns([1,1])
      with col211:
        NDPHCostMean = st.number_input("Historical Mean (%)", key="NDPHCostMean", value=29.43)
      with col212:
        NDPHCostDev = st.number_input("Historical Deviation (%)", key="NDPHCostDev", value=1.77)
    with st.expander("**Pizza Hut - Cost of Sales**", expanded=True):
      col211, col212 = st.columns([1,1])
      with col211:
        NDPHGrowthMean = st.number_input("Historical Mean (%)", key="NDPHGrowthMean", value=-3.78)
      with col212:
        NDPHGrowthDev = st.number_input("Historical Deviation (%)", key="NDPHGrowthDev", value=6.04)

  with col22:
    st.text("‎") 

  with col23:
    st.text("‎") 
    #API call 
    inputData = {
      "NDNumberOfSimulations": NDNumberOfSimulations,
      "NDKFCCostMean": NDKFCCostMean,
      "NDKFCGrowthMean": NDKFCGrowthMean,
      "NDPHCostMean": NDPHCostMean,
      "NDPHGrowthMean": NDPHGrowthMean,
      "NDKFCCostDev": NDKFCCostDev,
      "NDKFCGrowthDev": NDKFCGrowthDev,
      "NDPHCostDev": NDPHCostDev,
      "NDPHGrowthDev": NDPHGrowthDev
    }

    alldata = normalDistribution(inputData)
    outputs = alldata.json()['response_data']['outputs']
    st.write("Simulation Results")

    with st.expander("**Illustration**", expanded=True):
      st.markdown('***')
      col1, col2, col3, col4 = st.columns([1,1,1,1])
      with col1:
        formatted_monthlypayment = "{:,.0f}".format(outputs["Avg_Target_Price"])
        st.metric(label='Avg Cost of Goods ($)', value=formatted_monthlypayment)
      with col2:  
        formatted_totalpayment = "{:,.0f}".format(outputs["Avg_Profit_before_Tax"])
        st.metric(label='Avg Profit b.Tax ($)', value=formatted_totalpayment)
      with col3:  
        formatted_interest = "{:,.0f}".format(outputs["Avg_Revenue"])
        st.metric(label='Avg Revenue ($)', value=formatted_interest)
      with col4:  
        formatted_totalinterest = "{:,.0f}".format(outputs["Avg_COGS"])
        st.metric(label='Avg Target Price ($)', value=formatted_totalinterest)
      st.markdown('***')

      #generate line chart of results
      df_simresults = pd.DataFrame(outputs["Sim_Results"])
      fig_simresults = go.Figure()
      config_simresults = {
          'x_column': 'Testcase',
          'title': '      Testcase'
      }
      generate_line_chart(fig_simresults, df_simresults, config_simresults)
      st.plotly_chart(fig_simresults, use_container_width=True)

      st.markdown('***')
      st.dataframe(df_simresults, use_container_width=True)

with tab2:
  col21, col22, col23 = st.columns([12, 2, 32])
  col21, col22, col23 = st.columns([12, 2, 32])

  with col21:
    st.text("‎") 
    st.write("Simulation Controls")
    with st.expander("**Base**", expanded=True):
      col211, col212 = st.columns([1,1])
      with col211:
        DCKFCCost = st.number_input("KFC - Cost of Sales (%)", key="DCKFCCost", value=31.65)
        DCKFCGrowth = st.number_input("KFC SSSG (%)", key="DCKFCGrowth", value=31.65)
      with col212:
        DCPHCost = st.number_input("Pizza Hut - Cost of Sales (%)", key="DCPHCost", value=1.42)
        DCPHgrowth = st.number_input("KFC SSSG (%)", key="DCPHgrowth", value=31.65)
    with st.expander("**Deviation**", expanded=True):
      col211, col212 = st.columns([1,1])
      with col211:
        DCDevMin = st.number_input("Min (%)", key="DCDevMin", value=-2.8)
      with col212:
        DCDevMax = st.number_input("Max (%)", key="DCDevMax", value=6.21)

  with col22:
    st.text("‎") 

  with col23:
    st.text("‎") 
    #API call 
    # alldata = definedCombination(inputData)
    # outputs = alldata.json()['response_data']['outputs']
    st.write("Simulation Results")

    with st.expander("**Illustration**", expanded=True):
      st.markdown('***')
      col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1])
      with col1:
        formatted_monthlypayment = "{:,.0f}".format(1234)
        st.metric(label='Number of Simulations', value=formatted_monthlypayment)
      with col2:
        formatted_monthlypayment = "{:,.0f}".format(1234)
        st.metric(label='Avg Cost of Goods ($)', value=formatted_monthlypayment)
      with col3:  
        formatted_totalpayment = "{:,.0f}".format(1234)
        st.metric(label='Avg Profit b.Tax ($)', value=formatted_totalpayment)
      with col4:  
        formatted_interest = "{:,.0f}".format(1234)
        st.metric(label='Avg Revenue ($)', value=formatted_interest)
      with col5:  
        formatted_totalinterest = "{:,.0f}".format(1234)
        st.metric(label='Avg Target Price ($)', value=formatted_totalinterest)
      st.markdown('***')

      # #generate line chart of results
      # df_simresults = pd.DataFrame(outputs["Sim_Results"])
      # fig_simresults = go.Figure()
      # config_simresults = {
      #     'x_column': 'Testcase',
      #     'title': '      Testcase'
      # }
      # generate_line_chart(fig_simresults, df_simresults, config_simresults)
      st.plotly_chart(fig_simresults, use_container_width=True)

      # st.markdown('***')
      st.dataframe(df_simresults, use_container_width=True)

