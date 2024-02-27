import openai
import pandas as pd

#Put api key here

def read_and_summarize_dataset(file_path):
    data = pd.read_csv(file_path)
    data_string = data.to_string()

    ques = f'''Give me a summary based on the following for this dataset: {data_string}
    Using advanced analytical techniques, perform a thorough analysis of the provided dataset. This dataset encompasses various attributes, including but not limited to, numerical data and categorical data.
    
    Your analysis should meticulously cover the following areas, with a keen focus on providing actionable insights and uncovering underlying patterns:

   1. **Descriptive Statistics:**
   - Calculate and interpret measures like mean, median, mode, minimum, maximum, and standard deviation for relevant numerical columns.
   - Provide a brief explanation of what these statistics reveal about the dataset, with particular attention to salary distributions and variations.

2. **Detailed Enumeration of Attributes:**
   - For each categorical attribute in the dataset, meticulously calculate and list the total number of individuals present for each and EVERY category. Ensure your output is structured for clarity:
     - Category1: 
       - SubCategory1 = [Total Number], SubCategory2 = [Total Number], ...
     - Category2: 
       - SubCategory1 = [Total Number], SubCategory2 = [Total Number], ...
     - Category3: 
       - Subcategory1 = [Total Number], SubCategory2 = [Total Number], ...
   - This enumeration should extend to providing a detailed breakdown within each category and subcategory, illustrating disparities or trends based on other attributes.

3. **Key Metrics and Analysis:**
   - Beyond basic descriptive statistics, identify and calculate key metrics that summarize the dataset, with an emphasis on uncovering insights related to disparities, trends, and patterns.
   - Specifically, analyze descriptive statistics within each categorical grouping including 'Salary' to reveal any significant insights or trends.

    Your analysis should be comprehensive, aiming not only to quantify the dataset but also to provide qualitative insights into patterns, trends, and disparities, especially concerning salaries. Ensure your findings are structured, clear, and actionable, utilizing the detailed enumeration format provided for clarity. Your goal is to uncover not just the surface-level statistics but also the deeper insights that these numbers reflect about the workforce dynamics, salary structures, and any potential areas for improvement or further investigation.'''
    
    insight = openai.chat.completions.create(
            model="gpt-3.5-turbo", 
            messages=[{"role": "system", "content": ques}],
            temperature=0.5,  
            stop=None
        )

    ans =  insight.choices[0].message.content.strip()
    return ans

def generate_analytics(summary):
    prompt = f'''You are an Insightful AI Chatbot, you have the ability to answer any data analytical related questions or queries that is given by the user. 
    The questions can be related to the summary that is provided to you. This is the summary: \n{summary}\n
    The questions can be anything based on the summary of the dataset but mostly analytical, descriptive. 
    You should improvise and answer the queries according to the user input or queries with respect to any kind of summary provided to you
    '''

    messages = [{"role": "system", "content": prompt}]

    print("You can start asking your analytical questions. Type 'exit' to end the conversation.")

    while True:
        user_query = input("You: ")
        if user_query.lower() == "exit":
            break

        messages.append({"role": "user", "content": user_query})

        # Generate the response from OpenAI
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo", 
            messages=messages,
            temperature=0.5,  
            stop=None
        )


        ai_response = response.choices[0].message.content.strip()
        print(f"AI: {ai_response}")

        messages.append({"role": "assistant", "content": ai_response})

if __name__ == "__main__":
    file_path = "Company_Dataset.csv"  # Path to your uploaded CSV file
    output = read_and_summarize_dataset(file_path)
    print(output)
    generate_analytics(output)