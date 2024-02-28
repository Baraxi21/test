import os
import json
from langchain_experimental.agents import create_csv_agent
import openai
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, ChatMessage


# --------------------------------------------------------------
# Load OpenAI API Token From the .env File
# --------------------------------------------------------------

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

csv_agent = create_csv_agent(
    ChatOpenAI(temperature=0.5, model="gpt-3.5-turbo-0613"),
    "Company_Dataset.csv",
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True
)


user_prompt = "What is the Department Distribution?"
    # Simulate querying the CSV agent; replace with actual CSV agent call
analysis_result = csv_agent.invoke(user_prompt)
print(analysis_result['output'])

# function_descriptions = [
#     {
#         "name": "get_visualization",
#         "description": "Get parameters required to create a visualisation",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "chart": {
#                     "type": "string",
#                     "description": "The chart-type to create visualisation eg: bar, pie, scatter plot etc.",
#                 },
#                 "x": {
#                     "type": "string",
#                     "limit": "int",
#                     "description": "Column Values that need to be present in x-axis",
#                 },
#                 "y": {
#                     "type": "string",
#                     "limit": "int",
#                     "description": "Column Values that need to be present in y-axis",
#                 }
#             },
#             "required": ["chart-type", "x", "y"],
#         },
#     }
# ]


# def get_visualization(x,y,chart):
#     """Get flight information between two locations."""

#     # Example output returned from an API or database
#     vis_info = {
#         "chart-type": chart,
#         "x": x,
#         "y": y,
#     }

#     return json.dumps(vis_info)

# analysis = f"Based on this {analysis_result['output']}, You need to answer to the user prompt accordingly"

completion = openai.chat.completions.create(
        model="gpt-3.5-turbo-0613",
        temperature=0.5,
        messages=[{"role": "user", "content": analysis_result['output']}],
        functions=[
        {
        "name": "get_visualization",
        "description": "Get parameters required to create a visualisation",
        "parameters": {
            "type": "object",
            "properties": {
                "chart": {
                    "type": "string",
                    "description": "The chart-type to create visualisation eg: bar, pie, scatter plot etc.",
                },
                "x": {
                    "type": "string",
                    "description": "The Column Names that need to be present in x-axis should be in a list of strings showing all column names",
                },
                "y": {
                    "type": "integer",
                    "description": "Column values that need to be present in y-axis corresponding to the Column Names in x-axis should be in a list of integers",
                }
            },
            "required": ["chart-type", "x", "y"],
        },
    }
        ],
        function_call='auto'
    ) # specify the function call

print(completion.choices[0].message.function_call.arguments)