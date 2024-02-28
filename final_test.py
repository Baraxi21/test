import pandas as pd
import json
import matplotlib.pyplot as plt
from main_test import create_pd_agent, query_pd_agent
from datetime import datetime 
import os 

def save_response_to_file(response_dict):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    if "answer" in response_dict:
        answer_filename = f"answer_{timestamp}.txt"
        with open(answer_filename, "w") as file:
            file.write(response_dict["answer"])
        print(f"Answer saved at {answer_filename}")

    if "chart" in response_dict:
        chart_filename = f"chart_{timestamp}.png"
        os.rename("./chart_image/chart.png", f"./chart_image/{chart_filename}")
        print(f"Chart saved at ./chart_image/{chart_filename}")

    if "table" in response_dict:
        table_filename = f"table_data_{timestamp}.csv"
        data = response_dict["table"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        df.to_csv(table_filename, index=False)
        print(f"Table data saved as {table_filename}")

if __name__ == "__main__":
    file_path = "Company_Dataset.csv" 
    agent = create_pd_agent(file_path)

    while True:
        user_query = input("Enter your query (or type 'exit' to quit): ")
        if user_query.lower() == "exit":
            break
        response_str = query_pd_agent(agent, user_query)
        response_dict = json.loads(response_str)
        save_response_to_file(response_dict)
