from langchain_openai import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import openai
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType
import os
import pandas as pd
from dotenv import load_dotenv


load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

df = pd.read_csv("Company_Dataset.csv")
agent = create_pandas_dataframe_agent(OpenAI(temperature=0.5), df, verbose=False)

query = "Give me the Gender distribution based on the dataset on a bar graph"
prompt = (
        """
        Let's decode the way to respond to the queries. The responses depend on the type of information requested in the query. 
        ONLY EXECUTE THIS PART IF the word 'table', 'bar' or 'line' is mentioned

        ***1. If the query requires a table, format your answer like this:
           {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

        2. For a bar chart, respond like this:
           {"bar": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

        3. If a line chart is more appropriate, your reply should look like this:
           {"line": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}***

        ONLY EXECUTE THIS PART IF NO CHART TYPE IS MENTIONED
        4. For a plain question that doesn't need a chart or table, your response should be:
           {"answer": "Your answer goes here"}

        For example:
           {"answer": "The Product with the highest Orders is '15143Exfo'"}

        5. If the answer is not known or available, respond with:
           {"answer": "I do not know."}

        Return all output as a string. Remember to encase all strings in the "columns" list and data list in double quotes. 
        For example: {"columns": ["Products", "Orders"], "data": [["51993Masc", 191], ["49631Foun", 152]]}

        Now, let's tackle the query step by step. Here's the query for you to work on: 
        """
        + query
    )

    # Run the prompt through the agent and capture the response.
response = agent.run(prompt)

    # Return the response converted to a string.
print(str(response))

