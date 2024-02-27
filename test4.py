from langchain_experimental.agents import create_csv_agent
import openai
from langchain_openai import OpenAI
from langchain.agents.agent_types import AgentType
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# Load the CSV file into the CSV Agent
csv_agent = create_csv_agent(
    OpenAI(temperature=0.5, model="gpt-3.5-turbo"),
    "Company_Dataset.csv",
    verbose=True,
)

def generate_visualization(csv_agent, user_query):  
    
    json_string = """
    {
    "data": [{
        "type": "chart-type",
        "x": "[ColumnValue1], [ColumnValue2], .....",
        "y": "[ColumnValue1], [ColumnValue2], ....."
    }],
    "layout": {
        "title": "TitleName"
      }
    }
    """
    ques = f"""With the help of the data from the {csv_agent} answer accordingly with respect to the {user_query}, Give in a JSON response format like: {json_string}

    with respect to the user query mentioned and make user the response is generated accordingly like the following
    """

    viz = openai.chat.completions.create(
            model="gpt-3.5-turbo", 
            messages=[{"role": "system", "content": ques}],
            temperature=0.5,  
            stop=None
        )

    vis =  viz.choices[0].message.content.strip()
    return vis

if __name__ == '__main__':
    while True:
        user_query = input("You: ")
        if user_query.lower() == "exit":
            break
        
        visualization_json = generate_visualization(csv_agent, user_query)
        print(visualization_json)
