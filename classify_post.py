import pandas as pd
import torch
from transformers import DistilBertModel, DistilBertTokenizer, DistilBertForSequenceClassification
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sia = SentimentIntensityAnalyzer()
class DistilBERTClassifier(nn.Module):
    def __init__(self, num_classes):
        super(DistilBERTClassifier, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(768, num_classes) 

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :] 
        x = self.dropout(pooled_output)
        x = self.fc(x)
        return x
num_classes = 3 
model = DistilBERTClassifier(num_classes).to(device)
# Load the pre-trained DistilBERT model and tokenizer for risk classification
model_path = 'distilbert_risk_classifier_model_final.pth'
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model.load_state_dict(torch.load(model_path, map_location=device))


# Load the model weights

model.to(device)
model.eval()

# Function for sentiment classification (using Vader)
def get_vader_sentiment(text):
    score = sia.polarity_scores(text)["compound"]
    if score > 0.2:
        return "Positive"
    elif score < -0.2:
        return "Negative"
    else:
        return "Neutral"

# Function for risk level classification (using DistilBERT)
def predict_risk(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs
    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class  # 0: Low risk, 1: Moderate risk, 2: High risk

# Function to classify the posts
def classify_posts(posts_df):
    """
    Classify the sentiment and risk level of each post in the DataFrame.
    Arguments:
    posts_df (pd.DataFrame): A DataFrame with a column 'Combined_Text' containing the text data.
    
    Returns:
    pd.DataFrame: The original DataFrame with additional columns 'Sentiment' and 'Risk_Level'.
    """
    posts_df["Sentiment"] = posts_df["Combined_Text"].apply(get_vader_sentiment)

    # Apply risk classification with the additional logic for positive sentiment
    posts_df["Risk_Level"] = posts_df.apply(lambda row: 0 if row["Sentiment"] == "Positive" else predict_risk(row["Combined_Text"]), axis=1)

    return posts_df

# Function to plot sentiment and risk level distribution
def plot_distributions(posts_df):
    """
    Generate and display the plots for the sentiment and risk level distributions.
    Arguments:
    posts_df (pd.DataFrame): The DataFrame with 'Sentiment' and 'Risk_Level' columns.
    """
    # Plot Sentiment Distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(data=posts_df, x='Sentiment', palette="Set2")
    plt.title("Sentiment Distribution of Posts")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.show()

    # Plot Risk Level Distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(data=posts_df, x='Risk_Level', palette="coolwarm")
    plt.title("Risk Level Distribution of Posts")
    plt.xlabel("Risk Level")
    plt.ylabel("Count")
    plt.xticks(ticks=[0, 1, 2], labels=['Low Risk', 'Moderate Risk', 'High Risk'])
    plt.show()

def save_results(posts_df, file_name="classified_posts.csv"):
    """
    Save the classified posts DataFrame to a CSV file.
    Arguments:
    posts_df (pd.DataFrame): The DataFrame with classified results.
    file_name (str): The file name to save the results as.
    """
    posts_df.to_csv(file_name, index=False)

# Function to classify posts and generate plots with optional saving
def process_and_visualize(posts_df, save_plots=False, save_file=False, file_name="classified_posts.csv"):
    """
    Process the posts, classify them, and generate plots.
    Arguments:
    posts_df (pd.DataFrame): A DataFrame with a column 'Combined_Text' containing the text data.
    save_plots (bool): Whether to save the plots as images.
    save_file (bool): Whether to save the classified posts DataFrame to a CSV file.
    file_name (str): The file name to save the results if save_file is True.
    
    Returns:
    pd.DataFrame: The classified posts DataFrame.
    """
    # Classify the posts
    classified_posts_df = classify_posts(posts_df)
    
    # Generate and display plots
    plot_distributions(classified_posts_df)

    if save_file:
        save_results(classified_posts_df, file_name)

    return classified_posts_df
