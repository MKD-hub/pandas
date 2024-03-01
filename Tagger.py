import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
from datetime import datetime

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Comparison sentences, same as attributes on user profile

comparison_sentences = ["Software Design And Development", "Media And Communication", "Food And Drink Preparation Or Service", "Information Technology", "Accounting And Finance", "Creative Art And Design", "Architecture And Urban Planning", "Construction And Civil Engineering", "Health Care", "Hospitality And Tourism", "Translation And Transcription", "Manufacturing And Production",
"Logistics And Supply Chain", "Installation And Maintenance Technician", "Sales And Promotion", "Purchasing And Procurement", "Secretarial And Office Management", "Security And Safety", "Multimedia Content Production", "Horticulture", "Agriculture", "Livestock And Animal Husbandry", "Aeronautics And Aerospace", "Entertainment", "Fashion Design", "Clothing And Textile", "Project Management And Administration", "Business And Commerce", "Human Resources And Talent Management", "Mechanical And Electrical Engineering", "Chemical And Biomedical Engineering", "Environmental And Energy Engineering", "Research And Data Analytics", "Psychiatry, Psychology And Social Work", "Law", "Beauty And Grooming", "Labor Work And Masonry", "Teaching And Tutor", "Training And Mentorship", "Pharmaceutical", "Customer Service And Care", "Event Management And Organization", "Transportation And Delivery", "Woodwork And Carpentry", "Marketing And Advertisement", "Shop And Office Attendant", "Broker And Case Closer", "Advisory And Consultancy", "Janitorial And Other Office Services", "Documentation And Writing Services", "Veterinary", "Data Mining And Analytics", "Gardening And Landscaping" ]

# Tokenize and compute embeddings for comparison sentences
encoded_comparison_input = tokenizer(comparison_sentences, padding=True, truncation=True, return_tensors='pt')
with torch.no_grad():
    model_output_comparison = model(**encoded_comparison_input)
comparison_embeddings = mean_pooling(model_output_comparison, encoded_comparison_input['attention_mask'])
comparison_embeddings = F.normalize(comparison_embeddings, p=2, dim=1)

def get_similarity(target_sentence: str):
    # Tokenize and compute embeddings for the target sentence
    encoded_target_input = tokenizer([target_sentence], padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output_target = model(**encoded_target_input)
    target_embeddings = mean_pooling(model_output_target, encoded_target_input['attention_mask'])
    target_embeddings = F.normalize(target_embeddings, p=2, dim=1)

    # Calculate cosine similarity
    cosine_similarities = torch.matmul(target_embeddings, comparison_embeddings.t())
    similarities = cosine_similarities.tolist()

    # Pair each similarity score with its corresponding comparison sentence
    paired_similarities = list(zip(comparison_sentences, similarities[0]))

    # Sort the pairs by similarity score in descending order
    sorted_paired_similarities = sorted(paired_similarities, key=lambda x: x[1], reverse=True)

    # Filter to only include similarities above or equal to 0.33
    filtered_paired_similarities = [pair for pair in sorted_paired_similarities if pair[1] >= 0.27]

    # Extract the names of the most similar items
    most_similar_items = [item[0] for item in filtered_paired_similarities]

    return {"most_similar_items": most_similar_items}



# Read target sentences from CSV

import pandas as pd
import json
from datetime import datetime

def calculateSimilarity(filename):
    pathJSON = './JSON/similarity' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.json'

    df = pd.read_csv(filename)
    target_sentences = df['job_description'].tolist()
    links = df['link'].tolist() # Assuming the links are stored in a column named 'link'

    # Initialize an empty list to store the results
    results = []

    # Calculate similarities for each target sentence
    for i, target_sentence in enumerate(target_sentences):
        similarities = get_similarity(target_sentence)
        results.append({
            'description': target_sentence,
            'link': links[i], # Include the link for each job description
            'tags': similarities['most_similar_items']
        })

    # Format the results in the desired JSON structure
    output = {'jobs': results}

    # Write the results to a new JSON file
    with open(pathJSON, 'w') as outfile:
        json.dump(output, outfile, indent=4)

    return pathJSON

