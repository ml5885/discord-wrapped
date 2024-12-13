import os
import json
import pickle
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import umap
import hdbscan
import spacy
from scipy.special import logsumexp
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.makedirs("app", exist_ok=True)

CONVERSATIONAL_STOPWORDS = {
    'a','about','above','after','again','against','all','also','am',
    'an','and','any','are',"aren't",'as','at','be','because','been',
    'before','being','below','between','both','but','by',"can't",
    'cannot','could',"couldn't",'did',"didn't",'do','does',
    "doesn't",'doing',"don't",'down','during','each','few','for',
    'from','further','had',"hadn't",'has',"hasn't",'have',
    "haven't",'having','he',"he'd","he'll","he's",'her','here',
    "here's",'hers','herself','him','himself','his','how',"how's",
    'i',"i'd","i'll","i'm","i've",'if','in','into','is',"isn't",
    'it',"it's",'its','itself',"let's",'me','more','most',
    "mustn't",'my','myself','no','nor','not','of','off','on',
    'once','only','or','other','ought','our','ours','ourselves',
    'out','over','own','same',"shan't",'she',"she'd","she'll",
    "she's",'should',"shouldn't",'so','some','such','than','that',
    "that's",'the','their','theirs','them','themselves','then',
    'there',"there's",'these','they',"they'd","they'll","they're",
    "they've",'this','those','through','to','too','under','until',
    'up','very','was',"wasn't",'we',"we'd","we'll","we're",
    "we've",'were',"weren't",'what',"what's",'when',"when's",
    'where',"where's",'which','while','who',"who's",'whom','why',
    "why's",'with',"won't",'would',"wouldn't",'you',"you'd",
    "you'll","you're","you've",'your','yours','yourself','yourselves',
    'didn','don','haven','cause','just','kinda','like','dont',
    'wasn','couldn','wouldn','shouldn','doesn','didnt','hasnt',
    'havent','isnt','arent','weren'
}

TRIVIAL_TERMS_FOR_LABELING = {"the", "and", "of", "to", "a", "in", "it", "is"}

def load_messages(directory):
    messages = []
    json_files = [f for f in os.listdir(directory) if f.endswith(".json")]
    for filename in tqdm(json_files, desc="Loading JSON files", unit="file"):
        filepath = os.path.join(directory, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                for message in data:
                    content = message.get("content", "").strip()
                    timestamp = message.get("timestamp", "").strip()
                    author = (
                        message.get("author", {}).get("username", "Unknown").strip()
                    )
                    if content:
                        messages.append(
                            {
                                "content": content,
                                "timestamp": timestamp,
                                "author": author,
                            }
                        )
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from file {filename}: {e}")
        except Exception as e:
            print(f"Unexpected error loading file {filename}: {e}")
    df = pd.DataFrame(messages)
    return df

def preprocess_messages(df):
    initial_count = len(df)
    df = df.dropna(subset=["content"])
    df = df[df["content"].str.strip() != ""]
    df = df.reset_index(drop=True)
    final_count = len(df)
    print(f"Preprocessing completed: {initial_count - final_count} messages removed.")
    return df

def preprocess_text(text, nlp):
    doc = nlp(text)
    tokens = [token.text.lower() for token in doc if len(token.text) > 1]
    return " ".join(tokens)

def process_text_data(df):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    tfidf_cache_file = "tfidf_cache.pkl"
    if (os.path.exists(tfidf_cache_file)):
        print("Loading TF-IDF matrix and feature names from cache...")
        with open(tfidf_cache_file, "rb") as f:
            tfidf_matrix, feature_names = pickle.load(f)
        if (tfidf_matrix.shape[0] != len(df)):
            print("Cached TF-IDF matrix does not match current data length. Regenerating TF-IDF...")
            os.remove(tfidf_cache_file)
            return process_text_data(df)
    else:
        print("Generating TF-IDF matrix...")
        tfidf = TfidfVectorizer(
            max_features=10000, min_df=5, max_df=0.8, ngram_range=(1, 2), token_pattern=r"\b\w+\b", stop_words=list(CONVERSATIONAL_STOPWORDS),
        )
        tqdm.pandas(desc="Preprocessing text")
        df["processed_text"] = df["content"].progress_apply(preprocess_text, nlp=nlp)
        tfidf_matrix = tfidf.fit_transform(tqdm(df["processed_text"], desc="Fitting TF-IDF"))
        feature_names = tfidf.get_feature_names_out()
        with open(tfidf_cache_file, "wb") as f:
            pickle.dump((tfidf_matrix, feature_names), f)
        print("TF-IDF matrix and feature names cached.")

    if (tfidf_matrix.shape[0] != len(df)):
        raise ValueError("TF-IDF rows do not match DF length after processing.")
    return tfidf_matrix, feature_names

def generate_embeddings(df, model_name, force_new_embeddings=False):
    embeddings_file = f"{model_name.replace('/', '_')}_embeddings.pkl"
    if (os.path.exists(embeddings_file) and not force_new_embeddings):
        print(f"Loading {model_name} embeddings from cache...")
        with open(embeddings_file, "rb") as f:
            embeddings = pickle.load(f)
        embeddings = np.array(embeddings, dtype=np.float32)
        if (embeddings.shape[0] != len(df)):
            print("Cached embeddings do not match current data length. Regenerating embeddings...")
            os.remove(embeddings_file)
            return generate_embeddings(df, model_name, force_new_embeddings=True)
    else:
        print(f"Generating embeddings using {model_name}...")
        model = SentenceTransformer(model_name, trust_remote_code=True)
        content_list = df["content"].tolist()
        embeddings = model.encode(content_list, show_progress_bar=True, convert_to_numpy=True, dtype=np.float32)
        with open(embeddings_file, "wb") as f:
            pickle.dump(embeddings, f)
        print(f"{model_name} embeddings cached.")

    if (embeddings.shape[0] != len(df)):
        raise ValueError("Embeddings length does not match DataFrame length after generation.")
    return embeddings

def reduce_dimensionality(embeddings, dim=20, method="umap"):
    print(f"Reducing embeddings to {dim} dimensions using {method.upper()}...")
    if (method.lower() == "umap"):
        reducer = umap.UMAP(n_components=dim, random_state=42, verbose=False, n_neighbors=15, min_dist=0.1, metric='cosine')
    elif (method.lower() == "t-sne"):
        tsne_method = "exact" if (dim > 3) else "barnes_hut"
        reducer = TSNE(n_components=dim, method=tsne_method, random_state=42, verbose=1)
    else:
        raise ValueError("Method must be 'umap' or 't-sne'")
    reduced_embeddings = reducer.fit_transform(embeddings)
    if (reduced_embeddings.shape[0] != embeddings.shape[0]):
        raise ValueError("Reduced embeddings length does not match original embeddings length.")
    return reduced_embeddings

def cluster_documents_hdbscan(embeddings, reduction_method="umap"):
    reduced_embeddings = reduce_dimensionality(embeddings, dim=50, method="umap").astype(np.float32)
    print("Clustering with HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(metric="euclidean", cluster_selection_method="leaf")
    clusterer.fit(reduced_embeddings)
    labels = clusterer.labels_
    return labels

def cluster_documents_kmeans(embeddings, n_clusters=50):
    print(f"Clustering with KMeans (n_clusters={n_clusters})...")
    clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, verbose=1)
    clusterer.fit(embeddings)
    labels = clusterer.labels_
    return labels

def cluster_documents_gmm(embeddings, n_components=50, max_iter=100, tol=1e-4):
    print(f"Clustering with GMM (n_components={n_components})...")
    gmm = GaussianMixture(n_components=n_components, max_iter=max_iter, tol=tol, random_state=42, verbose=1)
    gmm.fit(embeddings)
    labels = gmm.predict(embeddings)
    return labels

def create_cluster_labels(df, tfidf_matrix, feature_names):
    print("Creating cluster labels...")
    cluster_labels = {}
    unique_clusters = df["cluster"].unique()

    for cluster_id in unique_clusters:
        mask = df["cluster"] == cluster_id
        cluster_docs = tfidf_matrix[mask]
        if (cluster_docs.shape[0] == 0 or cluster_id == -1):
            cluster_labels[cluster_id] = "Misc"
            continue

        centroid = cluster_docs.mean(axis=0).A1
        top_indices = centroid.argsort()[::-1]
        chosen_label = None
        for idx in top_indices:
            term = feature_names[idx]
            if (term not in TRIVIAL_TERMS_FOR_LABELING and len(term) > 2):
                chosen_label = term.title()
                break
        if (not chosen_label):
            chosen_label = "Misc"

        cluster_labels[cluster_id] = chosen_label

    df["cluster_label"] = df["cluster"].map(cluster_labels)
    return df

def prepare_visualization_data(df, embeddings_2d, output_json="app/vis_data.json"):
    print("Preparing visualization data...")
    if (embeddings_2d.shape[0] != len(df)):
        raise ValueError("Visualization embeddings length does not match dataframe length.")
    df["x"] = embeddings_2d[:, 0]
    df["y"] = embeddings_2d[:, 1]
    vis_data = df[["content", "timestamp", "author", "cluster", "cluster_label", "x", "y"]].to_dict(orient="records")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(vis_data, f, ensure_ascii=False, indent=4)
    print(f"Visualization data saved to '{output_json}' (total {len(df)} points)")

def save_clusters(df, output_file="app/final_clusters.json"):
    print("Saving final clusters...")
    clusters = df.groupby("cluster").apply(lambda x: x.to_dict(orient="records")).to_dict()
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(clusters, f, ensure_ascii=False, indent=4)
    print(f"Clusters saved to '{output_file}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster_method", type=str, default="kmeans", choices=["hdbscan", "kmeans", "gmm"], help="Clustering method")
    parser.add_argument("--n_clusters", type=int, default=50, help="Number of clusters for kmeans or gmm")
    parser.add_argument("--reduction_method", type=str, default="t-sne", choices=["umap", "t-sne"], help="Dimensionality reduction method for clustering and visualization")
    parser.add_argument("--embedding_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="SentenceTransformer model name for embeddings")
    parser.add_argument("--force_new_embeddings", action="store_true", help="Force regeneration of embeddings even if cached")
    args = parser.parse_args()

    data_dir = "DiscordChat/dumbassnamedtuna_8b1cfad5-4618-4ac3-8acd-90ac1d281cbf"

    print("\n=== Loading and Preprocessing Messages ===")
    df = load_messages(data_dir)
    df = preprocess_messages(df)
    print(f"After preprocessing, {len(df)} messages remain.")

    print("\n=== Processing Text Data ===")
    tfidf_matrix, feature_names = process_text_data(df)

    print("\n=== Generating Embeddings ===")
    embeddings = generate_embeddings(df, model_name=args.embedding_model, force_new_embeddings=args.force_new_embeddings)

    print("\n=== Clustering Documents ===")
    if (args.cluster_method == "hdbscan"):
        cluster_labels = cluster_documents_hdbscan(embeddings, reduction_method=args.reduction_method)
    elif (args.cluster_method == "kmeans"):
        cluster_labels = cluster_documents_kmeans(embeddings, n_clusters=args.n_clusters)
    else:
        cluster_labels = cluster_documents_gmm(embeddings, n_components=args.n_clusters)

    df["cluster"] = cluster_labels
    print(f"Clustering completed. {len(df['cluster'].unique())} unique clusters found.")

    print("\n=== Creating Cluster Labels ===")
    df = create_cluster_labels(df, tfidf_matrix, feature_names)

    print("\n=== Reducing Embeddings for Visualization ===")
    embeddings_2d = reduce_dimensionality(embeddings, dim=2, method=args.reduction_method)

    print("\n=== Preparing Visualization Data ===")
    prepare_visualization_data(df, embeddings_2d, "app/vis_data.json")

    print("\n=== Saving Final Clusters ===")
    save_clusters(df, "app/final_clusters.json")

    print("\n=== Processing Complete ===")
