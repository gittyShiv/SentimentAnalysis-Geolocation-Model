import pandas as pd
import re
import time
import praw
import nltk
import emoji
nltk.download("stopwords")
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()  
    text = emoji.demojize(text)  
    text = re.sub(r"http\S+|www\S+", "<URL>", text) 
    text = re.sub(r"[^a-zA-Z\s]", "", text) 
    text = " ".join(word for word in text.split() if word not in STOPWORDS) 
    return text  
   

def preprocess_data(reddit_df):

    reddit_df["Cleaned_Title"] = reddit_df["Title"].apply(clean_text)
    reddit_df["Cleaned_Content"] = reddit_df["Content"].apply(clean_text)
    
    reddit_df = reddit_df.drop(columns=['Content', 'Title'])
    
  
    reddit_df = reddit_df.drop_duplicates(subset=["Cleaned_Title", "Cleaned_Content"], keep="first")
    

    reddit_df = reddit_df.dropna(subset=["Cleaned_Title", "Cleaned_Content"])
    
    reddit_df["Combined_Text"] = reddit_df["Cleaned_Title"] + " " + reddit_df["Cleaned_Content"]
    
    return reddit_df

def fetch_reddit_data():
    
    STOPWORDS = set(stopwords.words("english"))
    
    reddit = praw.Reddit(
        client_id="yourid",
        client_secret="yourSecret",
        user_agent="ModelAppSentimentDemo by u/username"
    )
    
    SUBREDDITS = ["depression", "mentalhealth", "addiction", "suicidewatch"]
    KEYWORDS = ["depressed", "hopeless", "numb", "empty", "worthless", "suicidal", 
                "want to die", "can't go on", "self-harm", "overwhelmed", 
                "anxious", "panic attack", "lost", "addiction", "withdrawal", 
                "relapse", "need help"]
    
    posts = []
    post_count = 0
    max_posts = 20000  
    
    for subreddit in SUBREDDITS:
        for post in reddit.subreddit(subreddit).new(limit=1000):  
            if any(keyword in post.title.lower() or keyword in post.selftext.lower() for keyword in KEYWORDS):
                posts.append([
                    post.id, post.created_utc, post.title, post.selftext, 
                    post.score, post.num_comments, post.permalink
                ])
                post_count += 1

            if post_count >= max_posts:
                break
        
        if post_count >= max_posts:
            break 

        time.sleep(5)

    reddit_df = pd.DataFrame(posts, columns=["ID", "Timestamp", "Title", "Content", "Upvotes", "Comments", "URL"])
    df_final = preprocess_data(reddit_df)
    df_final.to_csv("Cleaned_dataset.csv")
    print("dataset saved as \"Cleaned_dataset.csv\"")
    return df_final
