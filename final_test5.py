import os
import json
from datetime import datetime
from main_test import create_pd_agent, query_pd_agent
import openai
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# This function remains unchanged, handling the saving of responses to files.
def save_response_to_file(response_dict):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    if "answer" in response_dict:
        answer_filename = f"answer_{timestamp}.txt"
        with open(answer_filename, "w") as file:
            file.write(response_dict["answer"])
        print(f"Answer saved at {answer_filename}")
        response_dict["answer_path"] = answer_filename  # Include the file path in the response dictionary

    if "chart" in response_dict:
        chart_filename = f"chart_{timestamp}.html"
        # Assuming the chart is already generated and saved as "./chart_html/chart.html"
        os.rename("./chart_html/chart.html", f"./chart_html/{chart_filename}")
        print(f"Chart saved at ./chart_html/{chart_filename}")
        response_dict["chart_path"] = f"./chart_html/{chart_filename}"  # Include the file path in the response dictionary

    if "table" in response_dict:
        table_filename = f"table_data_{timestamp}.csv"
        data = response_dict["table"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        df.to_csv(table_filename, index=False)
        print(f"Table data saved as {table_filename}")
        response_dict["table_path"] = table_filename  # Include the file path in the response dictionary

# This new function will handle both conversational and analytical queries.
def process_combined_query(file_path, user_query):
    # Define keywords that trigger analytical processing.
    analytical_keywords = ['chart', 'graph', 'table', 'distribution', 'average', 'median', 'mode', 'maximum', 'minimum']
    # Check if the query should be handled analytically.
    if any(keyword in user_query.lower() for keyword in analytical_keywords):
        # Process the query analytically using the pandas dataframe agent.
        agent = create_pd_agent(file_path)
        response_str = query_pd_agent(agent, user_query)
        response_dict = json.loads(response_str)
    else:

        conversation_history = [
            {"role": "system", "content": "You must greet and welcome the user when they greet you, and keep it conversational. Your responses should be conversational."},
            {"role": "user", "content": user_query}
        ]
        

        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=conversation_history,
            temperature=0.5
        )

        # Assuming the last message is the chatbot's response
        chat_response = response.choices[0].message.content
        response_dict = {"answer": chat_response}
    # Save the response to files and return the response dictionary.
    saved_response = save_response_to_file(response_dict)
    return saved_response

# Example usage (you would integrate this logic with your FastAPI endpoint).
def handle_user_interaction(file_path, user_query):
        response = process_combined_query(file_path, user_query)
        print(json.dumps(response, indent=4))
