import os
import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import nltk
from nltk.corpus import stopwords
from gensim import corpora, models
import re
from nltk.stem import WordNetLemmatizer

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

def load_messages(directory):
    messages = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for message in data:
                    content = message.get('content', '')
                    timestamp = message.get('timestamp', '')
                    author = message.get('author', {})
                    username = author.get('username', 'Unknown')
                    messages.append({'content': content, 'timestamp': timestamp, 'author': username})
    df = pd.DataFrame(messages)
    return df

def preprocess_messages(df):
    # Remove messages with missing or empty content
    df = df.dropna(subset=['content'])
    df = df[df['content'].str.strip() != '']
    df = df.reset_index(drop=True)
    return df

def generate_embeddings(df):
    # Use BERT embeddings
    # For a lighter model, you can use 'all-MiniLM-L6-v2'
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    embeddings = model.encode(df['content'].tolist(), show_progress_bar=True)
    df['embedding'] = embeddings.tolist()
    return df

def cluster_embeddings(df, num_clusters=10):
    embeddings = np.array(df['embedding'].tolist())

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    df['cluster'] = clusters
    print(f"K-Means clustering completed with {num_clusters} clusters.")

    return df

def assign_cluster_labels(df):
    cluster_labels = {}
    stop_words = set(stopwords.words('english'))
    custom_stop_words = set(['like', 'yeah', 'okay', 'don', 've', 're', 'll', 'im', 'oh'])
    stop_words.update(custom_stop_words)
    lemmatizer = WordNetLemmatizer()
    for cluster in df['cluster'].unique():
        cluster_df = df[df['cluster'] == cluster]
        texts = cluster_df['content'].tolist()
        # Handle possible empty cluster
        if len(texts) == 0:
            cluster_labels[cluster] = 'No Label'
            continue
        # Preprocess texts
        processed_texts = []
        for text in texts:
            # Remove non-alphabetic characters and tokenize
            tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
            # Remove stopwords
            tokens = [word for word in tokens if word not in stop_words]
            # Optional lemmatization
            # tokens = [lemmatizer.lemmatize(word) for word in tokens]
            processed_texts.append(tokens)
        # Create dictionary and corpus for LDA
        dictionary = corpora.Dictionary(processed_texts)
        corpus = [dictionary.doc2bow(text) for text in processed_texts]
        # Train LDA model
        try:
            lda_model = models.LdaModel(corpus, num_topics=1, id2word=dictionary, passes=10)
            # Get the top words for the topic
            top_words = lda_model.show_topic(0, topn=3)
            label = ', '.join([word for word, prob in top_words])
            cluster_labels[cluster] = label
        except Exception as e:
            # Handle exceptions (e.g., not enough data)
            cluster_labels[cluster] = 'No Label'
            print(f"Could not generate label for cluster {cluster}: {e}")
    df['cluster_label'] = df['cluster'].map(cluster_labels)
    return df

def reduce_dimensions(df):
    embeddings = np.array(df['embedding'].tolist())
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1)
    embeddings_2d = tsne.fit_transform(embeddings)
    df['x'] = embeddings_2d[:, 0]
    df['y'] = embeddings_2d[:, 1]
    return df

def prepare_visualization_data(df, output_json):
    # Sample data if it's too large
    max_points = 5000  # Adjust as needed
    if len(df) > max_points:
        df_sampled = df.sample(n=max_points, random_state=42)
    else:
        df_sampled = df

    vis_data = df_sampled[['content', 'timestamp', 'author', 'cluster', 'cluster_label', 'x', 'y']].to_dict(orient='records')
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(vis_data, f, ensure_ascii=False)
    print(f"Visualization data saved to {output_json} (sampled {len(df_sampled)} points)")

if __name__ == "__main__":
    data_dir = 'DiscordChat/dumbassnamedtuna_8b1cfad5-4618-4ac3-8acd-90ac1d281cbf'
    output_json = 'app/vis_data.json'

    print("Loading messages...")
    df = load_messages(data_dir)
    print(f"Loaded {len(df)} messages.")

    print("Preprocessing messages...")
    df = preprocess_messages(df)
    print(f"After preprocessing, {len(df)} messages remain.")

    print("Generating embeddings...")
    df = generate_embeddings(df)

    print("Clustering embeddings...")
    df = cluster_embeddings(df, num_clusters=10)  # Adjust num_clusters as needed

    print("Assigning cluster labels...")
    df = assign_cluster_labels(df)

    print("Reducing dimensions for visualization...")
    df = reduce_dimensions(df)

    print("Preparing visualization data...")
    prepare_visualization_data(df, output_json)

    print("Data processing complete.")