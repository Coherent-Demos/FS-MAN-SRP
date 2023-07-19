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

    url = "https://excel.uat.us.coherent.global/coherent/api/v3/folders/CLSA/services/Xcall Solution - Yum China - Defined Comb/Execute"

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
            "Max_Deviation": inputdata["DCMaxDev"],
            "Min_Deviation": inputdata["DCMinDev"],
            "Step": 0.005
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
                mode='lines+markers',
                name=column,
                line=dict(color=config['color']),
                marker=dict(size=10,)
            ))

    fig.update_layout(title=config['title'], height=360)
    fig.update_xaxes(title_text='Testcase')
    fig.update_yaxes(title_text='Amount ($)')
    fig.update_xaxes(type="category")

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
        NDKFCCostMean = st.number_input("Historical Mean (%)", key="NDKFCCostMean", value=31.65, step=0.5)
      with col212:
        NDKFCCostDev = st.number_input("Historical Deviation (%)", key="NDKFCCostDev", value=1.42, step=0.5)
    with st.expander("**KFC - Same Store Sales Growth**", expanded=True):
      col211, col212 = st.columns([1,1])
      with col211:
        NDKFCGrowthMean = st.number_input("Historical Mean (%)", key="NDKFCGrowthMean", value=-2.8, step=0.5)
      with col212:
        NDKFCGrowthDev = st.number_input("Historical Deviation (%)", key="NDKFCGrowthDev", value=6.21, step=0.5)
    with st.expander("**Pizza Hut - Cost of Sales**", expanded=True):
      col211, col212 = st.columns([1,1])
      with col211:
        NDPHCostMean = st.number_input("Historical Mean (%)", key="NDPHCostMean", value=29.43, step=0.5)
      with col212:
        NDPHCostDev = st.number_input("Historical Deviation (%)", key="NDPHCostDev", value=1.77, step=0.5)
    with st.expander("**Pizza Hut - Same Store Sales Growth**", expanded=True):
      col211, col212 = st.columns([1,1])
      with col211:
        NDPHGrowthMean = st.number_input("Historical Mean (%)", key="NDPHGrowthMean", value=-3.78, step=0.5)
      with col212:
        NDPHGrowthDev = st.number_input("Historical Deviation (%)", key="NDPHGrowthDev", value=6.04, step=0.5)
    with st.expander("**Generate Output**", expanded=True):
      if st.button("Submit", use_container_width=True):
        #API call 
        inputData = {
          "NDNumberOfSimulations": NDNumberOfSimulations,
          "NDKFCCostMean": NDKFCCostMean / 100,
          "NDKFCGrowthMean": NDKFCGrowthMean / 100,
          "NDPHCostMean": NDPHCostMean / 100,
          "NDPHGrowthMean": NDPHGrowthMean / 100,
          "NDKFCCostDev": NDKFCCostDev / 100,
          "NDKFCGrowthDev": NDKFCGrowthDev / 100,
          "NDPHCostDev": NDPHCostDev / 100,
          "NDPHGrowthDev": NDPHGrowthDev / 100
        }

        st.write(inputData)
        alldata = normalDistribution(inputData)
        outputs = alldata.json()['response_data']['outputs']
        st.write("Simulation Results")
      

  with col22:
    st.text("‎") 

  with col23:
    st.text("‎") 
    #API call 
    inputData = {
      "NDNumberOfSimulations": NDNumberOfSimulations,
      "NDKFCCostMean": NDKFCCostMean / 100,
      "NDKFCGrowthMean": NDKFCGrowthMean / 100,
      "NDPHCostMean": NDPHCostMean / 100,
      "NDPHGrowthMean": NDPHGrowthMean / 100,
      "NDKFCCostDev": NDKFCCostDev / 100,
      "NDKFCGrowthDev": NDKFCGrowthDev / 100,
      "NDPHCostDev": NDPHCostDev / 100,
      "NDPHGrowthDev": NDPHGrowthDev / 100
    }

    alldata = normalDistribution(inputData)
    outputs = alldata.json()['response_data']['outputs']
    df_simresults = pd.DataFrame(outputs["Sim_Results"])

    st.write("Simulation Results")

    with st.expander("**Illustration**", expanded=True):
      st.markdown('***')
      col1, col2, col3, col4 = st.columns([1,1,1,1])
      with col1:
        formatted_monthlypayment = "{:,.0f}".format(outputs["Avg_Target_Price"])
        st.metric(label='Avg Target Price ($)', value=formatted_monthlypayment)
      with col2:  
        formatted_totalpayment = "{:,.0f}".format(outputs["Avg_Profit_before_Tax"])
        st.metric(label='Avg Profit b.Tax ($)', value=formatted_totalpayment)
      with col3:  
        formatted_interest = "{:,.0f}".format(outputs["Avg_Revenue"])
        st.metric(label='Avg Revenue ($)', value=formatted_interest)
      with col4:  
        formatted_totalinterest = "{:,.0f}".format(outputs["Avg_COGS"])
        st.metric(label='Avg Cost of Goods ($)', value=formatted_totalinterest)
      st.markdown('***')

      #generate line chart of results
      df_NDCOGS = df_simresults[[df_simresults.columns[0], df_simresults.columns[1]]]
      fig_NDCOGS = go.Figure()
      config_NDCOGS = {
          'x_column': 'Testcase',
          'title': '      COGS',
          'color': 'red'
      }
      generate_line_chart(fig_NDCOGS, df_NDCOGS, config_NDCOGS)
      st.plotly_chart(fig_NDCOGS, use_container_width=True)

      #generate line chart of results
      df_NDProfit = df_simresults[[df_simresults.columns[0], df_simresults.columns[2]]]
      fig_NDProfit = go.Figure()
      config_NDProfit = {
          'x_column': 'Testcase',
          'title': '      Profit Before Tax',
          'color': 'blue'
      }
      generate_line_chart(fig_NDProfit, df_NDProfit, config_NDProfit)
      st.plotly_chart(fig_NDProfit, use_container_width=True)
      #generate line chart of results
      df_NDRevenue = df_simresults[[df_simresults.columns[0], df_simresults.columns[3]]]
      fig_NDRevenue = go.Figure()
      config_NDRevenue = {
          'x_column': 'Testcase',
          'title': '      Revenue',
          'color': 'purple'
      }
      generate_line_chart(fig_NDRevenue, df_NDRevenue, config_NDRevenue)
      st.plotly_chart(fig_NDRevenue, use_container_width=True)
      
      #generate line chart of results
      df_NDPrice = df_simresults[[df_simresults.columns[0], df_simresults.columns[4]]]
      fig_NDPrice = go.Figure()
      config_NDPrice = {
          'x_column': 'Testcase',
          'title': '      Target Price',
          'color': 'orange'
      }
      generate_line_chart(fig_NDPrice, df_NDPrice, config_NDPrice)
      st.plotly_chart(fig_NDPrice, use_container_width=True)

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
        DCKFCSSSG = st.number_input("KFC SSSG (%)", key="DCKFCSSSG", value=31.65)
      with col212:
        DCPHCost = st.number_input("Pizza Hut - Cost of Sales (%)", key="DCPHCost", value=1.42)
        DCPHSSSG = st.number_input("KFC SSSG (%)", key="DCPHSSSG", value=31.65)
    with st.expander("**Deviation**", expanded=True):
      DCMinDev, DCMaxDev = st.slider("Select a Minimum and Maximum (%)", min_value=0.0, max_value=100.0, value=(1.0, 10.0), step=0.5)

  with col22:
    st.text("‎") 

  with col23:
    st.text("‎") 
    # #API call 
    # inputData2 = {
    #   "DCKFCCost": DCKFCCost / 100, 
    #   "DCKFCSSSG": DCKFCSSSG / 100, 
    #   "DCPHCost": DCPHCost / 100, 
    #   "DCPHSSSG": DCPHSSSG / 100, 
    #   "DCMaxDev": DCMaxDev / 100, 
    #   "DCMinDev": DCMinDev / 100
    # }

    # alldata = definedCombination(inputData2)
    # outputs = alldata.json()['response_data']['outputs']
    # st.write("Simulation Results")

    # with st.expander("**Illustration**", expanded=True):
    #   st.markdown('***')
    #   col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1])
    #   with col1:
    #     DCNumberOfSimulations = "{:,.0f}".format(1234)
    #     st.metric(label='Number of Simulations', value=DCNumberOfSimulations)
    #   with col2:
    #     DCAvgCost = "{:,.0f}".format(1234)
    #     st.metric(label='Avg Cost of Goods ($)', value=DCAvgCost)
    #   with col3:  
    #     DCAvgProfit = "{:,.0f}".format(1234)
    #     st.metric(label='Avg Profit b.Tax ($)', value=DCAvgProfit)
    #   with col4:  
    #     DCAvgRevenue = "{:,.0f}".format(1234)
    #     st.metric(label='Avg Revenue ($)', value=DCAvgRevenue)
    #   with col5:  
    #     DCAvgTargetPrice = "{:,.0f}".format(1234)
    #     st.metric(label='Avg Target Price ($)', value=DCAvgTargetPrice)
    #   st.markdown('***')

    #   # #generate line chart of results
    #   # df_simresults = pd.DataFrame(outputs["Sim_Results"])
    #   # fig_simresults = go.Figure()
    #   # config_simresults = {
    #   #     'x_column': 'Testcase',
    #   #     'title': '      Testcase'
    #   # }
    #   # generate_line_chart(fig_simresults, df_simresults, config_simresults)
    #   st.plotly_chart(fig_simresults, use_container_width=True)

    #   # st.markdown('***')
    #   st.dataframe(df_simresults, use_container_width=True)
