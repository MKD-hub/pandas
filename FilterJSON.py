import pandas as pd
import re
from datetime import datetime

def contains_amharic(text):
    # Unicode range for Amharic characters
    amharic_regex = r'[\u1200-\u137F]'
    # Search for any Amharic characters in the text
    if re.search(amharic_regex, text):
        return True
    
    return False



# Function to detect if a text is in English
def is_english(text):
    return contains_amharic(text) == False



def filter(file):

    df_filtered = pd.DataFrame()
    description_pattern = r"Description: (.*)"
    title_pattern = r"Job Title: (.*?) Job Type:"
    # Load the JSON file
    
    df = pd.read_json(file)
  

    # Convert the JSON data to a pandas DataFrame
    df = df['data']

    for job in df:
        try:
            content = job.get('content')
            link = job.get('link')
            date = job.get('date')

            timestamp = date * 1000
            date = datetime.fromtimestamp(timestamp / 1000)
            
            if content:
                description_match = re.search(description_pattern, content)
                title_match = re.search(title_pattern, content)
                # print(description_match.group(1));

                description = description_match.group(1) or "" # in cases where there's no match
                title = title_match.group(1) or ""
                new_row = pd.DataFrame({'date': [date], 'job_title': [title], 'job_description': [description], 'link': [link]})  

            if is_english(description_match.group(1)):
                df_filtered = df_filtered._append(new_row, ignore_index=True)

        except KeyError:
            print(f"Missing 'job_description' attribute in row")
    
    filename = "./csv/filtered_data_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".csv"
    
    # Save the filtered DataFrame to a CSV file
    df_filtered.to_csv(filename, index=False)

    return str(filename)