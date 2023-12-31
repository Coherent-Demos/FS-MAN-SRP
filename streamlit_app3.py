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

    url = "https://excel.sit.coherent.global/coherent/api/v3/folders/CLSA/services/Xcall Solution - Yum China - Normal Distn - output/Execute"

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
       'x-synthetic-key': '14e31d2a-bb4e-4cb5-bf54-e07d2fdae4fa'
    }


    response = requests.request("POST", url, headers=headers, data=payload, allow_redirects=False)
    return response

@st.cache_data
def definedCombination(inputdata):
    if 'DCloading' in st.session_state:
      DCloading.warning("Running Simulations")
    url = "https://excel.uat.us.coherent.global/coherent/api/v3/folders/CLSA/services/Xcall Yum China - Defined Comb - output template/Execute"

    payload = json.dumps({
       "request_data": {
          "inputs": {
            "Base_Inputs": [
              {
                "Inputs": "Y2023",
                "KFC cost of sales (%)": inputdata["DCKFCCost2023"],
                "KFC Same store sales growth %": inputdata["DCKFCSSSG2023"],
                "Pizza hut cost of sales (%)": inputdata["DCPHCost2023"],
                "Pizza hut Same store sales growth %": inputdata["DCPHSSSG2023"]
              },
              {
                "Inputs": "Y2024",
                "KFC cost of sales (%)": inputdata["DCKFCCost2024"],
                "KFC Same store sales growth %": inputdata["DCKFCSSSG2024"],
                "Pizza hut cost of sales (%)": inputdata["DCPHCost2024"],
                "Pizza hut Same store sales growth %": inputdata["DCPHSSSG2024"]
              },
              {
                "Inputs": "1 SD",
                "KFC cost of sales (%)": inputdata["DCKFCCostDev"],
                "KFC Same store sales growth %": inputdata["DCKFCSSSGDev"],
                "Pizza hut cost of sales (%)": inputdata["DCPHCostDev"],
                "Pizza hut Same store sales growth %": inputdata["DCPHSSSGDev"]
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

#Start of UI
image_path = "coherent-logo.png"
st.image(image_path, caption="", width=32)

st.write("## Pricing Simulation")

tab1, tab2 = st.tabs(['Defined Combination', 'Normal Distribution'])

with tab1:
  col21, col22, col23 = st.columns([12, 2, 32])
  col21, col22, col23 = st.columns([12, 2, 32])

  with col21:
    st.text("‎") 
    st.write("Simulation Controls")
    with st.form("DC Form"):
      with st.expander("**Y2023**", expanded=True):
        col211, col212 = st.columns([1,1])
        with col211:
          DCKFCCost2023 = st.number_input("KFC - Cost of Sales (%)", key="DCKFCCost2023", value=31.00)
          DCKFCSSSG2023 = st.number_input("KFC - SSSG (%)", key="DCKFCSSSG2023", value=1.00)
        with col212:
          DCPHCost2023 = st.number_input("Pizza Hut - Cost of Sales (%)", key="DCPHCost2023", value=31.00)
          DCPHSSSG2023 = st.number_input("Pizza Hut - SSSG (%)", key="DCPHSSSG2023", value=2.00)
      with st.expander("**Y2024**", expanded=True):
        col211, col212 = st.columns([1,1])
        with col211:
          DCKFCCost2024 = st.number_input("KFC - Cost of Sales (%)", key="DCKFCCost2024", value=31.00)
          DCKFCSSSG2024 = st.number_input("KFC - SSSG (%)", key="DCKFCSSSG2024", value=1.00)
        with col212:
          DCPHCost2024 = st.number_input("Pizza Hut - Cost of Sales (%)", key="DCPHCost2024", value=31.00)
          DCPHSSSG2024 = st.number_input("Pizza Hut - SSSG (%)", key="DCPHSSSG2024", value=2.00)      
      with st.expander("**1 SD**", expanded=True):
        col211, col212 = st.columns([1,1])
        with col211:
          DCKFCCostDev = st.number_input("KFC - Cost of Sales (%)", key="DCKFCCostDev", value=6.2146779660914)
          DCKFCSSSGDev = st.number_input("KFC - SSSG (%)", key="DCKFCSSSGDev", value=1.4185217180759)
        with col212:
          DCPHCostDev = st.number_input("Pizza Hut - Cost of Sales (%)", key="DCPHCostDev", value=6.0369234254249)
          DCPHSSSGDev = st.number_input("Pizza Hut - SSSG (%)", key="DCPHSSSGDev", value=1.7665147022123) 
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
            "DCKFCCost2023": DCKFCCost2023 / 100,
            "DCKFCSSSG2023": DCKFCSSSG2023 / 100,
            "DCPHCost2023": DCPHCost2023 / 100,
            "DCPHSSSG2023": DCPHSSSG2023 / 100,
            "DCKFCCost2024": DCKFCCost2024 / 100,
            "DCKFCSSSG2024": DCKFCSSSG2024 / 100,
            "DCPHCost2024": DCPHCost2024 / 100,
            "DCPHSSSG2024": DCPHSSSG2024 / 100,
            "DCKFCCostDev": DCKFCCostDev / 100,
            "DCKFCSSSGDev": DCKFCSSSGDev / 100,
            "DCPHCostDev": DCPHCostDev / 100,
            "DCPHSSSGDev": DCPHSSSGDev / 100
          }
          DCalldata = definedCombination(inputData2)
          DCoutputs = DCalldata.json()['response_data']['outputs']
          df_DCsimresults = pd.DataFrame(DCoutputs["Sim_Results"])

          # add success alert after api call

  with col22:
    st.text("‎") 

  with col23:
    st.text("‎") 
    st.write("Simulation Results")
    #API call 
    inputData2 = {
      "DCKFCCost2023": DCKFCCost2023 / 100,
      "DCKFCSSSG2023": DCKFCSSSG2023 / 100,
      "DCPHCost2023": DCPHCost2023 / 100,
      "DCPHSSSG2023": DCPHSSSG2023 / 100,
      "DCKFCCost2024": DCKFCCost2024 / 100,
      "DCKFCSSSG2024": DCKFCSSSG2024 / 100,
      "DCPHCost2024": DCPHCost2024 / 100,
      "DCPHSSSG2024": DCPHSSSG2024 / 100,
      "DCKFCCostDev": DCKFCCostDev / 100,
      "DCKFCSSSGDev": DCKFCSSSGDev / 100,
      "DCPHCostDev": DCPHCostDev / 100,
      "DCPHSSSGDev": DCPHSSSGDev / 100
    }
    DCalldata = definedCombination(inputData2)
    DCoutputs = DCalldata.json()['response_data']['outputs']
    df_DCsimresults = pd.DataFrame(DCoutputs["Sim_Results"])

    with st.expander("**Illustration**", expanded=True):
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
      st.markdown('***')

      #generate line chart of results
      df_DCCOGS = df_DCsimresults[[df_DCsimresults.columns[0], df_DCsimresults.columns[1]]]
      fig_DCCOGS = go.Figure()
      config_DCCOGS = {
          'x_column': 'Testcase',
          'title': '      COGS',
          'color': 'purple'
      }
      generate_bar_chart(fig_DCCOGS, df_DCCOGS, config_DCCOGS)
      st.plotly_chart(fig_DCCOGS, use_container_width=True)

      #generate line chart of results
      df_DCProfit = df_DCsimresults[[df_DCsimresults.columns[0], df_DCsimresults.columns[2]]]
      fig_DCProfit = go.Figure()
      config_DCProfit = {
          'x_column': 'Testcase',
          'title': '      Profit Before Tax',
          'color': 'green'
      }
      generate_bar_chart(fig_DCProfit, df_DCProfit, config_DCProfit)
      st.plotly_chart(fig_DCProfit, use_container_width=True)
      #generate line chart of results
      df_DCRevenue = df_DCsimresults[[df_DCsimresults.columns[0], df_DCsimresults.columns[3]]]
      fig_DCRevenue = go.Figure()
      config_DCRevenue = {
          'x_column': 'Testcase',
          'title': '      Revenue',
          'color': 'blue'
      }
      generate_bar_chart(fig_DCRevenue, df_DCRevenue, config_DCRevenue)
      st.plotly_chart(fig_DCRevenue, use_container_width=True)
      
      #generate line chart of results
      df_DCPrice = df_DCsimresults[[df_DCsimresults.columns[0], df_DCsimresults.columns[4]]]
      fig_DCPrice = go.Figure()
      config_DCPrice = {
          'x_column': 'Testcase',
          'title': '      Target Price',
          'color': 'orange'
      }
      generate_bar_chart(fig_DCPrice, df_DCPrice, config_DCPrice)
      st.plotly_chart(fig_DCPrice, use_container_width=True)

      st.markdown('***')
      st.dataframe(df_DCsimresults, use_container_width=True)

with tab2:
  col21, col22, col23 = st.columns([12, 2, 32])

  with col21:
    st.text("‎") 
    st.write("Simulation Controls")
    with st.form("ND Form"):
      with st.expander("**General**", expanded=True):
        NDNumberOfSimulations = st.number_input("Number of Simulations", key="NDNumberOfSimulations", value=10, max_value=10000)
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

        button_clicked = st.form_submit_button("Submit", use_container_width=True)
        if button_clicked:
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
          NDalldata = normalDistribution(inputData)
          NDoutputs = NDalldata.json()['response_data']['outputs']
          df_NDsimresults = pd.DataFrame(NDoutputs["Sim_Results"])
      

  with col22:
    st.text("‎") 

  with col23:
    st.text("‎") 
    st.write("Simulation Results")
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
    NDalldata = normalDistribution(inputData)
    NDoutputs = NDalldata.json()['response_data']['outputs']
    df_NDsimresults = pd.DataFrame(NDoutputs["Sim_Results"])

    with st.expander("**Illustration**", expanded=True):
      st.markdown('***')
      col1, col2, col3, col4 = st.columns([1,1,1,1])
      with col1:
        formatted_monthlypayment = "{:,.0f}".format(NDoutputs["Avg_Target_Price"])
        st.metric(label='Avg Target Price ($)', value=formatted_monthlypayment)
      with col2:  
        formatted_totalpayment = "{:,.0f}".format(NDoutputs["Avg_Profit_before_Tax"])
        st.metric(label='Avg Profit b.Tax ($)', value=formatted_totalpayment)
      with col3:  
        formatted_interest = "{:,.0f}".format(NDoutputs["Avg_Revenue"])
        st.metric(label='Avg Revenue ($)', value=formatted_interest)
      with col4:  
        formatted_totalinterest = "{:,.0f}".format(NDoutputs["Avg_COGS"])
        st.metric(label='Avg Cost of Goods ($)', value=formatted_totalinterest)
      st.markdown('***')

      #generate line chart of results
      df_NDCOGS = df_NDsimresults[[df_NDsimresults.columns[0], df_NDsimresults.columns[1]]]
      fig_NDCOGS = go.Figure()
      config_NDCOGS = {
          'x_column': 'Testcase',
          'title': '      COGS',
          'color': 'purple'
      }
      generate_bar_chart(fig_NDCOGS, df_NDCOGS, config_NDCOGS)
      st.plotly_chart(fig_NDCOGS, use_container_width=True)

      #generate line chart of results
      df_NDProfit = df_NDsimresults[[df_NDsimresults.columns[0], df_NDsimresults.columns[2]]]
      fig_NDProfit = go.Figure()
      config_NDProfit = {
          'x_column': 'Testcase',
          'title': '      Profit Before Tax',
          'color': 'green'
      }
      generate_bar_chart(fig_NDProfit, df_NDProfit, config_NDProfit)
      st.plotly_chart(fig_NDProfit, use_container_width=True)
      #generate line chart of results
      df_NDRevenue = df_NDsimresults[[df_NDsimresults.columns[0], df_NDsimresults.columns[3]]]
      fig_NDRevenue = go.Figure()
      config_NDRevenue = {
          'x_column': 'Testcase',
          'title': '      Revenue',
          'color': 'blue'
      }
      generate_bar_chart(fig_NDRevenue, df_NDRevenue, config_NDRevenue)
      st.plotly_chart(fig_NDRevenue, use_container_width=True)
      
      #generate line chart of results
      df_NDPrice = df_NDsimresults[[df_NDsimresults.columns[0], df_NDsimresults.columns[4]]]
      fig_NDPrice = go.Figure()
      config_NDPrice = {
          'x_column': 'Testcase',
          'title': '      Target Price',
          'color': 'orange'
      }
      generate_bar_chart(fig_NDPrice, df_NDPrice, config_NDPrice)
      st.plotly_chart(fig_NDPrice, use_container_width=True)

      st.markdown('***')
      st.dataframe(df_NDsimresults, use_container_width=True)
