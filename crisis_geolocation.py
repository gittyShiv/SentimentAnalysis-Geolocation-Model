import pandas as pd
import folium
from collections import Counter
from geopy.geocoders import Nominatim
from nltk.tokenize import word_tokenize
import nltk
import spacy
spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")
nltk.download("punkt_tab")
geolocator = Nominatim(user_agent="geo_locator")

def extract_location(text):
   doc = nlp(text)
   for ent in doc.ents:
      if ent.label_ == "GPE":  # GPE = Geopolitical Entity (cities, countries, etc.)
         try:
             place = geolocator.geocode(ent.text, timeout=10)
             if place:
                 return place.address, place.latitude, place.longitude
         except:
            continue
   return None, None, None

def generate_crisis_heatmap(dataframe, output_html="crisis_heatmap.html"):

    high_risk_df = dataframe[dataframe["Risk_Level"] == 2].copy()
    locations = high_risk_df["Combined_Text"].apply(extract_location)
    
    if locations.isna().all():
        print(" No valid locations found. Exiting function.")
        return

    high_risk_df["Location"], high_risk_df["Lat"], high_risk_df["Lon"] = zip(*locations)
    geo_data = high_risk_df.dropna(subset=["Lat", "Lon"])

    if geo_data.empty:
        print(" No valid geographic data to plot.")
        return

    location_counts = Counter(geo_data["Location"])
    top_5_locations = location_counts.most_common(5)

    print("**Top 5 Crisis Locations:**")
    for location, count in top_5_locations:
        print(f" {location}: {count} mentions")

    m = folium.Map(location=[geo_data["Lat"].mean(), geo_data["Lon"].mean()], zoom_start=5)

    for _, row in geo_data.iterrows():
        folium.CircleMarker(
            location=(row["Lat"], row["Lon"]),
            radius=5,
            color="red",
            fill=True,
            fill_color="red",
            fill_opacity=0.5,
            popup=row["Location"]
        ).add_to(m)

    m.save(output_html)
    print(f"Heatmap saved as {output_html}")
    print(m)
