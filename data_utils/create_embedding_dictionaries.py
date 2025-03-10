import json
import os
import pickle
from collections import defaultdict, Counter
import numpy as np
import time
import argparse

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Create embedding dictionaries for citation network data")
    
    parser.add_argument("--dataset_path", type=str, default="data/test_dataset.json", 
                        help="Path to the citation network dataset JSON file")
    parser.add_argument("--output_path", type=str, default="embedding_dictionaries.pkl",
                        help="Path to save the embedding dictionaries")
    parser.add_argument("--limit_embeddings", action="store_true", default=False,
                        help="Limit to only the top N most frequent topics and keywords")
    parser.add_argument("--top_n_topics", type=int, default=2000,
                        help="Number of top topics to keep when limit_embeddings is True")
    parser.add_argument("--top_n_keywords", type=int, default=3000,
                        help="Number of top keywords to keep when limit_embeddings is True")
    
    return parser.parse_args()

# Main execution
if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Track execution time
    start_time = time.time()
    print("Starting extraction process...")

    # Load the citation network dataset
    DATASET_PATH = args.dataset_path
    print(f"Loading dataset from {DATASET_PATH}...")

    # Instead of loading the entire dataset at once (which might cause memory issues),
    # we'll process it line by line for topics and keywords
    topics_dict = {}  # Map topic ID to index
    keywords_dict = {}  # Map keyword ID to index
    topic_display_names = {}  # Map topic ID to display name
    keyword_display_names = {}  # Map keyword ID to display name

    # Set to track all unique topics and keywords
    all_topics = set()
    all_keywords = set()
    
    # Counters to track frequency
    topic_counter = Counter()
    keyword_counter = Counter()

    # Process the file in chunks to avoid memory issues
    print("Processing dataset to extract topics and keywords...")
    chunk_size = 1000  # Number of papers to process at a time
    paper_count = 0
    line_count = 0

    # First pass: Identify all unique topics and keywords and count frequencies
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        # Check the file structure - is it JSON Lines or a single JSON object?
        first_char = f.read(1)
        f.seek(0)  # Reset to beginning of file
        
        if first_char == '{':
            # Single JSON object
            print("Detected single JSON object format")
            data = json.load(f)
            if 'papers' in data:
                print(f"Found {len(data['papers'])} papers in dataset")
                for paper_id, paper_data in data['papers'].items():
                    paper_count += 1
                    
                    # Extract topics
                    if 'topics' in paper_data:
                        for topic in paper_data['topics']:
                            topic_id = topic.get('id', None)
                            display_name = topic.get('displayName', None)
                            
                            if topic_id:
                                all_topics.add(topic_id)
                                topic_counter[topic_id] += 1
                                # Save display name if available
                                if display_name and topic_id not in topic_display_names:
                                    topic_display_names[topic_id] = display_name
                    
                    # Extract keywords
                    if 'keywords' in paper_data:
                        for keyword in paper_data['keywords']:
                            keyword_id = keyword.get('id', None)
                            display_name = keyword.get('displayName', None)
                            
                            if keyword_id:
                                all_keywords.add(keyword_id)
                                keyword_counter[keyword_id] += 1
                                # Save display name if available
                                if display_name and keyword_id not in keyword_display_names:
                                    keyword_display_names[keyword_id] = display_name
                    
                    if paper_count % 1000 == 0:
                        print(f"Processed {paper_count} papers, found {len(all_topics)} topics and {len(all_keywords)} keywords")
                        
            else:
                print("Invalid file structure - 'papers' key not found")
        else:
            # JSON Lines format
            print("Detected JSON Lines format")
            for line in f:
                line_count += 1
                try:
                    paper_data = json.loads(line)
                    paper_count += 1
                    
                    # Extract topics
                    if 'topics' in paper_data:
                        for topic in paper_data['topics']:
                            topic_id = topic.get('id', None)
                            display_name = topic.get('displayName', None)
                            
                            if topic_id:
                                all_topics.add(topic_id)
                                topic_counter[topic_id] += 1
                                # Save display name if available
                                if display_name and topic_id not in topic_display_names:
                                    topic_display_names[topic_id] = display_name
                    
                    # Extract keywords
                    if 'keywords' in paper_data:
                        for keyword in paper_data['keywords']:
                            keyword_id = keyword.get('id', None)
                            display_name = keyword.get('displayName', None)
                            
                            if keyword_id:
                                all_keywords.add(keyword_id)
                                keyword_counter[keyword_id] += 1
                                # Save display name if available
                                if display_name and keyword_id not in keyword_display_names:
                                    keyword_display_names[keyword_id] = display_name
                    
                    if paper_count % 1000 == 0:
                        print(f"Processed {paper_count} papers, found {len(all_topics)} topics and {len(all_keywords)} keywords")
                except json.JSONDecodeError:
                    print(f"Error parsing line {line_count}, skipping")

print(f"Completed extraction. Found {len(all_topics)} unique topics and {len(all_keywords)} unique keywords across {paper_count} papers.")

# Filter to top N if limit_embeddings is True
filtered_topics = all_topics
filtered_keywords = all_keywords

if args.limit_embeddings:
    # Get top N topics
    top_n_topics = min(args.top_n_topics, len(all_topics))
    top_topics = [topic for topic, _ in topic_counter.most_common(top_n_topics)]
    filtered_topics = set(top_topics)
    
    # Get top N keywords
    top_n_keywords = min(args.top_n_keywords, len(all_keywords))
    top_keywords = [keyword for keyword, _ in keyword_counter.most_common(top_n_keywords)]
    filtered_keywords = set(top_keywords)
    
    print(f"Limited to top {len(filtered_topics)} topics and top {len(filtered_keywords)} keywords")
    print(f"Original counts: {len(all_topics)} topics, {len(all_keywords)} keywords")
    
    # Calculate proportion of coverage
    total_topic_occurrences = sum(topic_counter.values())
    covered_topic_occurrences = sum(topic_counter[topic] for topic in filtered_topics)
    topic_coverage = covered_topic_occurrences / total_topic_occurrences if total_topic_occurrences > 0 else 0
    
    total_keyword_occurrences = sum(keyword_counter.values())
    covered_keyword_occurrences = sum(keyword_counter[keyword] for keyword in filtered_keywords)
    keyword_coverage = covered_keyword_occurrences / total_keyword_occurrences if total_keyword_occurrences > 0 else 0
    
    print(f"Topic coverage: {topic_coverage:.2%}")
    print(f"Keyword coverage: {keyword_coverage:.2%}")

# Create dictionaries mapping IDs to indices
print("Creating mapping dictionaries...")
topic_id_to_index = {topic_id: idx for idx, topic_id in enumerate(sorted(filtered_topics))}
keyword_id_to_index = {keyword_id: idx for idx, keyword_id in enumerate(sorted(filtered_keywords))}

# Create inverse mappings
index_to_topic_id = {idx: topic_id for topic_id, idx in topic_id_to_index.items()}
index_to_keyword_id = {idx: keyword_id for keyword_id, idx in keyword_id_to_index.items()}

# Create index to display name mappings
index_to_topic_display = {idx: topic_display_names.get(topic_id, "Unknown") 
                          for idx, topic_id in index_to_topic_id.items()}
index_to_keyword_display = {idx: keyword_display_names.get(keyword_id, "Unknown") 
                           for idx, keyword_id in index_to_keyword_id.items()}

# Package dictionaries for saving
embedding_data = {
    'topic_id_to_index': topic_id_to_index,
    'keyword_id_to_index': keyword_id_to_index,
    'index_to_topic_id': index_to_topic_id,
    'index_to_keyword_id': index_to_keyword_id,
    'index_to_topic_display': index_to_topic_display,
    'index_to_keyword_display': index_to_keyword_display,
    'topic_count': len(filtered_topics),
    'keyword_count': len(filtered_keywords),
    # Add frequency information
    'topic_frequencies': {topic: topic_counter[topic] for topic in filtered_topics},
    'keyword_frequencies': {keyword: keyword_counter[keyword] for keyword in filtered_keywords},
    'limited_embeddings': args.limit_embeddings
}

# Save dictionaries to pickle file
SAVE_PATH = args.output_path
with open(SAVE_PATH, 'wb') as f:
    pickle.dump(embedding_data, f)

print(f"Dictionaries saved to {SAVE_PATH}")

# Save a human-readable summary
SUMMARY_PATH = f"{os.path.splitext(SAVE_PATH)[0]}_summary.txt"
with open(SUMMARY_PATH, 'w', encoding='utf-8') as f:
    f.write(f"Topic and Keyword Embedding Summary\n")
    f.write(f"================================\n\n")
    f.write(f"Total unique topics: {len(filtered_topics)}\n")
    f.write(f"Total unique keywords: {len(filtered_keywords)}\n\n")
    
    if args.limit_embeddings:
        f.write(f"Note: Limited to top {args.top_n_topics} topics and top {args.top_n_keywords} keywords\n")
        f.write(f"Topic coverage: {topic_coverage:.2%}\n")
        f.write(f"Keyword coverage: {keyword_coverage:.2%}\n\n")
    
    f.write(f"Sample Topics (first 20):\n")
    f.write(f"--------------------\n")
    for idx, topic_id in list(index_to_topic_id.items())[:20]:
        display_name = index_to_topic_display[idx]
        frequency = topic_counter[topic_id]
        f.write(f"[{idx}] {display_name} ({topic_id}) - Frequency: {frequency}\n")
    
    f.write(f"\nSample Keywords (first 20):\n")
    f.write(f"--------------------\n")
    for idx, keyword_id in list(index_to_keyword_id.items())[:20]:
        display_name = index_to_keyword_display[idx]
        frequency = keyword_counter[keyword_id]
        f.write(f"[{idx}] {display_name} ({keyword_id}) - Frequency: {frequency}\n")

# Calculate and print execution time
end_time = time.time()
execution_time = end_time - start_time
print(f"Process completed in {execution_time:.2f} seconds")

def create_one_hot_encoding(embedding_data=None):
    """
    Creates a function that converts paper data to one-hot encodings for topics and keywords.
    
    Args:
        embedding_data: Dictionary containing the embedding mappings. If None, tries to load from file.
        
    Returns:
        function: A function that takes paper_data and returns topic_vector and keyword_vector
    """
    if embedding_data is None:
        # Load dictionaries if not provided
        try:
            with open("embedding_dictionaries.pkl", 'rb') as f:
                embedding_data = pickle.load(f)
        except Exception as e:
            print(f"Error loading embedding dictionaries: {e}")
            return None
    
    # Get the necessary dictionaries
    topic_id_to_index = embedding_data['topic_id_to_index']
    keyword_id_to_index = embedding_data['keyword_id_to_index']
    topic_count = embedding_data['topic_count']
    keyword_count = embedding_data['keyword_count']
    
    def paper_to_one_hot(paper_data):
        """
        Convert paper data to one-hot encoded vectors for topics and keywords.
        
        Args:
            paper_data: Dictionary containing paper information with 'topics' and 'keywords'
            
        Returns:
            tuple: (topic_vector, keyword_vector) as numpy arrays
        """
        # Create one-hot vectors
        topic_vector = np.zeros(topic_count, dtype=np.float32)
        keyword_vector = np.zeros(keyword_count, dtype=np.float32)
        
        # Fill topic vector
        if 'topics' in paper_data:
            for topic in paper_data['topics']:
                if 'id' in topic and topic['id'] in topic_id_to_index:
                    idx = topic_id_to_index[topic['id']]
                    score = topic.get('score', 1.0)  # Use score if available, else 1.0
                    topic_vector[idx] = score
        
        # Fill keyword vector
        if 'keywords' in paper_data:
            for keyword in paper_data['keywords']:
                if 'id' in keyword and keyword['id'] in keyword_id_to_index:
                    idx = keyword_id_to_index[keyword['id']]
                    score = keyword.get('score', 1.0)  # Use score if available, else 1.0
                    keyword_vector[idx] = score
        
        return topic_vector, keyword_vector

    return paper_to_one_hot

print(f"Execution completed in {time.time() - start_time:.2f} seconds")

# Add example code to demonstrate using these dictionaries
print("\nExample usage:")
print("from create_embedding_dictionaries import create_one_hot_encoding")
print("import pickle")
print("")
print("# Load dictionaries")
print("with open('embedding_dictionaries.pkl', 'rb') as f:")
print("    embedding_data = pickle.load(f)")
print("")
print("# Get dictionary mappings")
print("topic_id_to_index = embedding_data['topic_id_to_index']")
print("index_to_topic_display = embedding_data['index_to_topic_display']")
print("")
print("# Create one-hot encoding function")
print("paper_to_one_hot = create_one_hot_encoding()")
print("")
print("# Convert paper to one-hot vectors")
print("paper_data = {'topics': [{'id': 'T12345', 'score': 0.8}]}")
print("topic_vector, keyword_vector = paper_to_one_hot(paper_data)")
print("")
print("# Use these vectors for feature initialization")
print("features = np.concatenate([topic_vector, keyword_vector])")
print("")
print("# To get human-readable info")
print("for idx, value in enumerate(topic_vector):")
print("    if value > 0:")
print("        topic_id = embedding_data['index_to_topic_id'][idx]")
print("        display_name = embedding_data['index_to_topic_display'][idx]")
print("        print(f'Topic {topic_id} ({display_name}): {value}')") 