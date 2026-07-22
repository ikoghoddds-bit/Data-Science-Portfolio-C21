---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.3
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region id="CxYSWPCsM0YU" -->
# Natural Language Processing


<!-- #endregion -->

<!-- #region id="FGQCPUEuNezI" -->
This project will give you practical experience using Natural Language Processing techniques. This project is in three parts:
- in part 1) you will use a dataset in a CSV file
- in part 2) you will use the Wikipedia API to directly access content
on Wikipedia.
- in part 3) you will make your notebook interactive

<!-- #endregion -->

<!-- #region id="IP5iaS9lNamW" -->
## Part 1)


<!-- #endregion -->

<!-- #region id="twaXGmp6OAPK" -->
- The CSV file is available at https://ddc-datascience.s3.amazonaws.com/Projects/Project.5-NLP/Data/NLP.csv
- The file contains a list of famous people and a brief overview.
- The goal of part 1) is to ...
  1. Pick one person from the list ( the reference person ) and output 10 other people who's overview are "closest" to the reference person in a Natural Language Processing sense
  1. Also output the sentiment of the overview of the reference person


<!-- #endregion -->

```python id="H0zBN_V3Lp2I"
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

```

```python colab={"base_uri": "https://localhost:8080/", "height": 35} id="AqmkvPDiMYoO" outputId="a558c48b-b470-4c7c-ad2f-aa606458f8e6"
# Load data directly from S3

url = "https://ddc-datascience.s3.amazonaws.com/Projects/Project.5-NLP/Data/NLP.csv"
url
```

```python colab={"base_uri": "https://localhost:8080/", "height": 424} id="dmf0mN13MrPx" outputId="e8d211c8-9389-4d29-8af8-af56c5e5e479"
# Inspect structure

df = pd.read_csv(url)
df
```

```python colab={"base_uri": "https://localhost:8080/"} id="HQ2Yi_6gQcKr" outputId="15281630-52b8-49ba-b325-e74c140015a6"
df.shape
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="_K_5v478QeJA" outputId="755fbb58-5932-4150-b739-a306312e8807"
df.head()
```

<!-- #region id="Tu9AeC1UUQp6" -->
##### Data Cleaning & Preprocessing
<!-- #endregion -->

```python id="oQdMoq95UUvB"
# Drop any missing values or duplicates

df = df.dropna(subset=['name', 'text']).drop_duplicates(subset=['URI']).reset_index(drop=True)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 424} id="k-RFrrHsUnVN" outputId="967282eb-2f40-4a0e-cf6a-9a3ab8fc8b67"
df
```

```python id="c03DJxNOUZqh"
# Standardize text column

df['cleaned_text'] = df['text'].astype(str).str.lower().str.replace(r'[^a-z0-9\s]', '', regex=True)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 458} id="XYI_Jl3wUq5U" outputId="0ec8c79c-0ed6-47fa-f1d7-30db2eb24b9b"
df['cleaned_text']
```

<!-- #region id="QdRYFfg2UwSe" -->
### Vectorization & Nearest Neighbors Search
<!-- #endregion -->

```python id="Q5oai9DvU3TJ"
# Initialize and fit TF-IDF Vectorizer across the full dataset

tfidf = TfidfVectorizer(stop_words='english', max_features=10000)
tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])
```

```python id="vy067MGhVBK8"
# Pick a reference person (e.g., 'Motoaki Takenouchi')

ref_name = "Motoaki Takenouchi"
ref_idx = df[df['name'] == ref_name].index[0]
```

```python id="jMv6lK63VBIc"
# Compute cosine similarity between reference vector and all vectors
cosine_sims = cosine_similarity(tfidf_matrix[ref_idx], tfidf_matrix).flatten()
```

```python id="wR7xOXMYVBGP"
# Get top 10 closest individuals (excluding the reference person itself)
similar_indices = cosine_sims.argsort()[::-1]
top_10_indices = [i for i in similar_indices if i != ref_idx][:10]

```

```python id="OuFUTXwHVBDe"
# Build Part 1 summary results DataFrame
part1_results = df.iloc[top_10_indices][['name', 'text']].copy()
part1_results['part1_similarity'] = cosine_sims[top_10_indices]
part1_results['part1_rank'] = range(1, 11)

```

```python colab={"base_uri": "https://localhost:8080/"} id="zKzE33y8VBAM" outputId="d8161f40-fcc6-4efd-b7ea-911cf8a446ac"
print(f"Top 10 closest people to '{ref_name}' (Part 1):")
print(part1_results[['part1_rank', 'name', 'part1_similarity']])
```

<!-- #region id="1uANoRGMViym" -->
### Sentiment Analysis on Overview Text
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="VdRI9QQyVA4x" outputId="3471f3b6-1df8-4e20-b5a4-2805fcbc8627"
# Calculate sentiment polarity and subjectivity of the reference person’s overview text.

ref_overview = df.loc[ref_idx, 'text']
blob_part1 = TextBlob(ref_overview)

print(f"--- Sentiment Output for {ref_name} (Overview Text) ---\n")
print("Polarity (-1 to 1):", blob_part1.sentiment.polarity)
print("Subjectivity (0 to 1):", blob_part1.sentiment.subjectivity)
```

<!-- #region id="SyoC6PWeN9un" -->
## Part 2)


<!-- #endregion -->

<!-- #region id="AJxfrb48N-v9" -->
- For the same reference person that you chose in Part 1), use the Wikipedia API to access the whole content of the reference person's Wikipedia page.
- The goal of Part 2) is to ...
  1. Print out the text of the Wikipedia article for the reference person
  1. Determine the sentiment of the text of the Wikipedia page for the reference person
  1. Collect the text of the Wikipedia pages from the 10 nearest neighbors from Part 1)
  1. Determine the nearness ranking of these 10 people to your reference person based on their entire Wikipedia page
  1. Compare, i.e. plot,  the nearest ranking from Step 1) with the Wikipedia page nearness ranking.  A difference of the rank is one means of comparison.


<!-- #endregion -->

<!-- #region id="EBPLvKdzWCel" -->
### Wikipedia API Integration & Comparison
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="ChhMiPn3WH8Z" outputId="9fbb1fd4-940b-46ef-ba91-66c8a524d614"
# Install wikipedia-api

!pip install wikipedia-api

import wikipediaapi

```

```python colab={"base_uri": "https://localhost:8080/"} id="iQe-UveTWPn_" outputId="52354b25-78f1-4837-9f90-5c54fd1e9531"
# Initialize Wikipedia API with a user agent

wiki = wikipediaapi.Wikipedia(user_agent="DataScienceProject/1.0", language="en")
wiki
```

```python id="1gmGwd9WWPcz"
# Function to safely fetch page text

def get_wiki_text(person_name):
    page = wiki.page(person_name)
    if page.exists():
        return page.text
    return ""
```

```python id="XgEolr_cW2BD"
# Fetch full text for reference person

ref_wiki_text = get_wiki_text(ref_name)

# Fetch full text for the top 10 neighbors

part1_results['wiki_text'] = part1_results['name'].apply(get_wiki_text)
```

<!-- #region id="uSvh6cw3XMSy" -->
### Print Text & Sentiment of Wikipedia Article
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="LxmZCfT3XP8e" outputId="1012f6bb-db26-4b40-d057-14f0c7188892"
# Print full text (or first 1,000 characters for preview)

print(f"--- Full Wikipedia Text for {ref_name} ---")
print(ref_wiki_text[:1000] + "\n...")
```

```python colab={"base_uri": "https://localhost:8080/"} id="2iSQeyngXcCB" outputId="b745c85c-9ef1-48be-8812-cd758c6e7d9a"
# Determine sentiment of full Wikipedia article

blob_part2 = TextBlob(ref_wiki_text)
print("\n--- Sentiment Output (Full Wikipedia Article) ---\n")
print("Polarity:", blob_part2.sentiment.polarity)
print("Subjectivity:", blob_part2.sentiment.subjectivity)
```

<!-- #region id="pM9TPauDXmhv" -->
### Re-Ranking Nearest Neighbors Using Entire Wikipedia Pages
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="NLSDPkzpX1xh" outputId="b80641c1-cb74-4e40-f0b0-9e91755d5395"
# Combine reference person and 10 neighbors full wiki texts into a corpus

wiki_corpus = [ref_wiki_text] + part1_results['wiki_text'].tolist()
wiki_corpus
```

```python colab={"base_uri": "https://localhost:8080/"} id="z6gXz-N0X5AT" outputId="193bab48-c298-4e7f-fc67-5dc0dc3d2ef3"
# Vectorize using TF-IDF

wiki_tfidf = TfidfVectorizer(stop_words='english')
wiki_matrix = wiki_tfidf.fit_transform(wiki_corpus)

wiki_tfidf
wiki_matrix
```

```python id="HKeY-NjNX44n"
# Compute similarity between reference page (index 0) and
# the 10 neighbors (indices 1 to 10)

wiki_sims = cosine_similarity(wiki_matrix[0:1], wiki_matrix[1:]).flatten()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 432} id="_qBk9OeCX4uZ" outputId="db3b8a17-775b-4bc9-addb-5fbb73035085"
# Add full Wikipedia similarity scores and calculate Part 2 rank

part1_results['part2_similarity'] = wiki_sims
part1_results['part2_rank'] = part1_results['part2_similarity'].rank(ascending=False, method='min').astype(int)

print(wiki_sims)
part1_results['part2_rank']

```

```python colab={"base_uri": "https://localhost:8080/"} id="SJSqXj8IX4rH" outputId="85ff2b6a-d4c9-45c3-e30c-09569f8f3868"
# Sort by Part 2 rank

part2_results = part1_results.sort_values(by='part2_rank').reset_index(drop=True)
print(part2_results[['part2_rank', 'name', 'part2_similarity', 'part1_rank']])
```

<!-- #region id="l5DAubAsZWhw" -->
### Comparative Analysis & Visualization
<!-- #endregion -->

```python id="dksgPG7ZZfKF"
import matplotlib.pyplot as plt
import seaborn as sns
```

```python colab={"base_uri": "https://localhost:8080/", "height": 507} id="8sd-KNM8X4n9" outputId="9ef6d261-b805-405d-cf9c-8f412c450fb9"
# Calculate rank difference
part1_results['rank_difference'] = part1_results['part1_rank'] - part1_results['part2_rank']

# Plot rank comparison

plt.figure(figsize=(10, 5))
sns.barplot(data=part1_results, x='name', y='rank_difference', palette='coolwarm', hue='name')

plt.xticks(rotation=45, ha='right')
plt.title(f"Rank Shift for Top 10 Neighbors of '{ref_name}' (Part 1 vs Part 2)")
plt.xlabel("Person Name")
plt.ylabel("Rank Difference (Part 1 Rank - Part 2 Rank)")
plt.axhline(0, color='black', linestyle='--')
plt.tight_layout()
plt.show()
```

<!-- #region id="xEIw9hfFObw7" -->
## Part 3)

<!-- #endregion -->

<!-- #region id="SVW0y4Axty7A" -->
Make an interactive notebook where a user can choose or enter a name and the notebook displays the 10 closest individuals.

In addition to presenting the project slides, at the end of the presentation each student will demonstrate their code using a famous person suggested by the other students that exists in the DBpedia set.

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="t6dopgR3vfDU" outputId="6a4feaa8-e65f-405b-c751-462a19ed489e"
!curl -s https://ddc-datascience.s3.amazonaws.com/Projects/Project.5-NLP/Data/NLP.csv | wc -l
```

```bash id="yHBJAbrGviBT" colab={"base_uri": "https://localhost:8080/"} outputId="3f3135a9-1dbe-49af-871f-644ba10c294c"
curl -s https://ddc-datascience.s3.amazonaws.com/Projects/Project.5-NLP/Data/NLP.csv |
head -1 |
tr , '\n' |
cat -n

```

```bash colab={"base_uri": "https://localhost:8080/"} id="SHpINW480KdC" outputId="b7c946a4-4f35-4bf5-817e-e4f28a7b8988"
curl -s https://ddc-datascience.s3.amazonaws.com/Projects/Project.5-NLP/Data/NLP.csv |
head -2 |
tail -1 |
tr , '\n' |
cat -n

```

<!-- #region id="tgTophJ_a7dO" -->
### Build Interactive UI Widget
<!-- #endregion -->

```python id="A9dv-4f-bYL9"
import ipywidgets as widgets
from IPython.display import display, clear_output
```

```python id="wjxZ4RyJa69m"
# Create interactive widget so users can search any person in the DBpedia dataset

# Define the end-to-end lookup function

def find_closest_people(person_name):
    if person_name not in df['name'].values:
        print(f"\n'{person_name}' not found in dataset. Please try another name.")
        return

    # Locate index

    idx = df[df['name'] == person_name].index[0]

    # Compute similarity across full TF-IDF matrix

    sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    top_indices = [i for i in sims.argsort()[::-1] if i != idx][:10]

    # Display results dataframe

    results = df.iloc[top_indices][['name']].copy()
    results['Cosine Similarity'] = sims[top_indices]

    # Compute overview sentiment

    text = df.loc[idx, 'text']
    sentiment = TextBlob(text).sentiment

    print(f"\n'=== Results for: {person_name} ===")
    print(f"Overview Sentiment: Polarity={sentiment.polarity:.3f}, Subjectivity={sentiment.subjectivity:.3f}\n")
    display(results.reset_index(drop=True))

```

```python colab={"base_uri": "https://localhost:8080/", "height": 464, "referenced_widgets": ["dfaa00dad3fe4938b44611d69d4b4ee7", "465a13f84ce840479acebce3427d6920", "28561b95aca5406487189ea1298903bc", "1e7ac103e08e4c2087c4b0a998ae0a36", "16b4298ae8964e94a25bdccfba2517cc", "f97b08753ff44793b7c23d383523ffd6", "4dd97c4852de4e8d8e79ea473632e4d9"]} id="X9vYUL6obtiR" outputId="c041ec1d-0dbf-41df-b637-5d0103c7b6e4"
# Create Dropdown Widget with famous names from dataset

dropdown = widgets.Dropdown(
    options=sorted(df['name'].unique()[:100]), # Top 100 names for fast scrolling
    description='Select Person:',
    disabled=False,
    style={'description_width': 'initial'} #Ensures full label text displays
)

# Interactively display results

widgets.interact(find_closest_people, person_name=dropdown);
```
