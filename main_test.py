import pandas as pd
import matplotlib.pyplot as plt
import openai
from langchain_openai import OpenAI
# from langchain.llms import AzureOpenAI
from langchain_experimental.agents import create_csv_agent
import os
from dotenv import load_dotenv
from langchain_experimental.agents import create_pandas_dataframe_agent

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
# OPENAI_DEPLOYMENT_NAME = "deployment name"
# OPENAI_MODEL_NAME = "model_name" 

csv_agent = create_csv_agent(
    OpenAI(temperature=0.5),
    "Company_Dataset.csv",
    verbose=False,
    handle_parsing_errors=True
)

def create_pd_agent(filename: str):

    # llm = AzureOpenAI(openai_api_key=OPENAI_API_KEY, 
    #                   deployment_name=OPENAI_DEPLOYMENT_NAME, 
    #                   model_name=OPENAI_MODEL_NAME)
    df = pd.read_csv(filename)
    return create_pandas_dataframe_agent(OpenAI(temperature=0.5), df, verbose=False, handle_parsing_errors=True)

def query_pd_agent(agent, query):
    prompt = (
        """
        You must need to use Plotly library if required to create a any interactive chart.

        If the query requires creating a chart, please save the chart as "./chart_html/chart.html" and "Here is the chart:" when reply as follows:
        {"chart": "Here is the chart:"}

        If the query requires creating a table, reply as follows:
        {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}
        
        If the query is just not asking for a chart, but requires a response, reply as follows:
        {"answer": "answer"}
        Example:
        {"answer": "The product with the highest sales is 'Lego'."}
        
        Lets think step by step.

        Here is the query: 
        """
        + query
    )

    response = agent.run(prompt)
    return response.__str__()
