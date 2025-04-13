import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

st.title('DS4420 Final Project: Chat(GPT) are we cooked?')
st.subheader('by Isabella Fisch and Caroline Han')
tab1, tab2, tab3 = st.tabs(["About", "Stack Overflow Data", "Time Series"])

df = pd.read_csv('TotalQuestions.csv')
df['Month'] = pd.to_datetime(df['Month'])
df['All'] = df.sum(axis=1, numeric_only=True)
df = df.dropna()

with tab1:
    st.header("About Our Project")
    st.write("""
    Open AI’s trailblazing large language model (LLM), ChatGPT, was first 
    released in November 2022, and has been credited for jump-starting 
    the AI boom. The widespread use of generative AI has affected sectors
    including education, customer service, healthcare, content creation, and 
    more. One strength that ChatGPT and similar AI models have been 
    credited with are their ability to debug code, increasing the 
    efficiency of software engineers, data scientists, and similar professions.
    Because of the wide-ranging effects of ChatGPT and other generative AI 
    models, we wanted to investigate the ways in which they have potentially 
    altered how people interact with pre-existing technologies. Using both time series 
    models and item-item collaborative filtering, we analyzed datasets related to the usage of 
    other technologies to discover how they may have been affected by the rise in LLMs. 
    This website displays some of our investigation and findings from the time series models. 
    """)

with tab2:
    st.header("Number of Stack Overflow Questions Over Time by Programming Language")
    st.write("""The plot below shows the number of questions asked on the Stack Overflow 
    website every month. Use the dropdown menu to view only questions asked in a specific 
    programming language.""")

    languages = df.columns[1:].tolist()
    language = st.selectbox('Select a Language', languages, index=languages.index('Python'))

    graph1 = df[['Month', language]]
    graph1.set_index('Month', inplace=True)
    graph1 = graph1.dropna()
    fig, ax = plt.subplots()
    ax.plot(graph1[language], label=f'{language} Questions')
    ax.axvline(x=pd.to_datetime('2022-11-01'), color='r', label='ChatGPT Release')
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Questions')
    ax.set_title(f'{language} Questions over Stack Overflow History')
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

with tab3:
    st.header("ARIMA Model Predicting Total Number of Stack Overflow Questions")
    st.write("""The graph below depicts the true number of Stack Overflow Questions 
    and the number predicted by an ARIMA time series model. Use the dropdown menu to select a 
    lag (order) for the model to use.""")

    graph2 = df[['Month', 'All']]
    graph2.set_index('Month', inplace=True)
    graph2 = graph2.dropna()

    lags = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    lag = st.selectbox('Select a Lag', lags, index=lags.index(11))

    for i in range(1, 20):
        graph2[f'Lag {i}'] = graph2['All'].shift(i)
    graph2 = graph2.dropna()

    train_size = int(.8 * len(graph2))
    train_data = graph2[:train_size]
    test_data = graph2[train_size:]

    y_train = np.array(train_data['All']).reshape(-1, 1)
    y_test = np.array(test_data['All']).reshape(-1, 1)

    arm_model = ARIMA(y_train, order=(lag, 0, lag))
    arm_results = arm_model.fit()
    arm_pred = arm_results.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)
    pred_data = pd.DataFrame({'Month': test_data.index, 'pred Total Questions': arm_pred.flatten()})
    pred_data.set_index('Month', inplace=True)

    fig, ax = plt.subplots()
    ax.plot(test_data['All'], label='Actual Total Questions')
    ax.plot(pred_data['pred Total Questions'], label=f'ARM({lag}, 0, {lag}) Predicted Total Questions', linestyle='--')
    ax.set_xlabel('Date')
    ax.set_ylabel('Total Questions over Stack Overflow History')
    ax.set_title(f'ARM({lag}, 0, {lag}) Model Predictions')
    ax.legend()
    ax.grid(True)
    fig.autofmt_xdate()

    st.pyplot(fig)
