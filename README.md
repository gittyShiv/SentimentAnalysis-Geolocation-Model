# **Reddit Sentiment Analysis with Longitudinal & Geospatial Crisis Trend Analysis**  

## **Overview**  
A significant amount of research has been conducted on sentiment analysis of social media posts, particularly on platforms like Reddit and Twitter. Many state-of-the-art methods leverage deep learning techniques for enhanced sentiment classification.This project aims to analyze Reddit posts to classify sentiments and assess crisis-related discussions.  
In this project, I have fine-tuned a **DistilBERT** model on a dataset of 1,825 Reddit posts, collected using the **Reddit API**. The sentiment analysis is performed on the combined title and content of each post to gain deeper insights.
For sentiment classification, I have used **TextBlob** to categorize posts as positive, negative, or neutral. Additionally, the fine-tuned DistilBERT model is employed to classify the risk level of each post, helping to identify crisis-related discussions.( **Accuracy of 0.76**)
### **Important Files:**
- **Reddit_data.py**: Script for retrieving and storing filtered and cleaned social media posts ready for prediction
- **classify_post.py**: Script to predict sentiment, risk level and provide plot for the same.
- **crisis_geolocation.py**: Script which provides location heatmap and top 5 places.
- **Sentiment_analysis__final.ipynb**: Notebook containing detailed code of modeling
  
## **Features**  
- **Sentiment Analysis**: Classifies posts as **Positive, Negative, or Neutral** using TextBlob.  
- **Crisis Risk Classification**: Uses a **fine-tuned DistilBERT model** to categorize posts into different risk levels.  
- **Geolocation Mapping**: Extracts locations from posts and plots them on an **interactive heatmap** using Folium.  

## **Installation**  

### **1. Clone the Repository**  
Open a terminal and run:  
```bash
git clone https://github.com/N-0-MAD/Reddit-sentiment-analysis-with-Longitudinal-Geospatial-Crisis-Trend-Analysis.git
cd Reddit-sentiment-analysis-with-Longitudinal-Geospatial-Crisis-Trend-Analysis
```

### **2. Create and Activate a Virtual Environment**  
```bash
python -m venv venv  
source venv/bin/activate  # For Mac/Linux  
venv\Scripts\activate  # For Windows  
```

### **3. Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **4. Download Required NLTK & SpaCy Models**  
Run the provided script to automatically download the necessary models:  

```bash
python setup_nltk_spacy.py
```
This script will ensure that all required **NLTK** and **SpaCy** models are installed and ready for use.
Although I have mentioned all the external dependencies used in this project, issues like numpy version conflict may occur. If this error occurs then try downgrading numpy version  to 1.19.5. 
If any dependency is not installed then you may install it using pip.
## **Usage**  

Import this python script in your Jupyter Notebook and run "fetch_reddit_data()" function to fetch filtered reddit posts using API key. The cleaned dataset will be saved as "Cleaned_dataset.csv"
Your dataset must have the following columns: columns=["ID", "Timestamp", "Title", "Content", "Upvotes", "Comments", "URL"]

### **1. Fetch the data and save the cleaned dataset**  
```Jupyter notebook
import Reddit_data
reddit_df = Reddit_data.fetch_reddit_data()
```
This script cleans the dataset, prepares it for sentiment analysis [reddit_df] , and saves it as "Cleaned_dataset.csv"

### **2. Classify Sentiments & Risk Levels**  
```Jupyter Notebook
import classify_post
reddit_df_classified= classify_post.process_and_visualize(reddit_df)
reddit_df_classified.to_csv("classified_risk_levels.csv", index=False)
```
This script uses TextBlob and DistilBERT to classify the sentiment and crisis risk level of posts. 
reddit_df_classified is the final dataframe having sentiments and risk levels (0=low risk, 1=moderate concern, 2=high risk)

### **3. Perform Geolocation Analysis & Generate a Heatmap**  
```Jupyter Notebook
from crisis_geolocation import generate_crisis_heatmap
crisis_geolocation.generate_crisis_heatmap(reddit_df_classified)
```
This script extracts location-related information and plots high-risk discussion points on a heatmap (saved as `crisis_heatmap.html`). 
Also, it will give the top 5 locations as well.

## **Results**  
- **Crisis Heatmap**: Highlights key locations where crisis-related discussions are prominent.  
- **Sentiment Classification**: Categorizes Reddit posts based on polarity.  
- **Risk Classification**: Uses deep learning (DistilBERT) to determine the urgency of posts.   

## **Example**  
An example of execution of above steps is given in the file 'Experiment.ipynb'

## **Improvements**
The current model achieved an accuracy of **0.76**. We can improve this model further by integrating Bert-CNN and other deep learning approaches. 
Also I am planning to add more data regularly and re-tune the model for better accuracy once I have around 2-3 lakh posts data. I am sure that we can achieve an accuracy > 0.8 by adding more data.
