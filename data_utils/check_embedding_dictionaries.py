import pickle
import sys

# Load the dictionaries
print("Loading embedding dictionaries...")
try:
    with open("embedding_dictionaries.pkl", "rb") as f:
        embedding_data = pickle.load(f)
    
    # Print the structure of the data
    print("\nEmbedding Dictionary Structure:")
    for key, value in embedding_data.items():
        if isinstance(value, dict):
            print(f"- {key}: Dictionary with {len(value)} entries")
        else:
            print(f"- {key}: {value}")
    
    # Print some statistics
    print("\nStatistics:")
    print(f"Number of topics: {embedding_data['topic_count']}")
    print(f"Number of keywords: {embedding_data['keyword_count']}")
    
    # Check if the dimensions match what's expected
    print("\nVerifying dimensions:")
    if len(embedding_data['topic_id_to_index']) == embedding_data['topic_count']:
        print("✓ Topic dimensions match")
    else:
        print("✗ Topic dimensions mismatch")
    
    if len(embedding_data['keyword_id_to_index']) == embedding_data['keyword_count']:
        print("✓ Keyword dimensions match")
    else:
        print("✗ Keyword dimensions mismatch")
    
    # Print a few sample entries to verify content
    print("\nSample Topic Mappings:")
    sample_topics = list(embedding_data['topic_id_to_index'].items())[:3]
    for topic_id, idx in sample_topics:
        display_name = embedding_data['index_to_topic_display'][idx]
        print(f"Topic ID {topic_id} -> Index {idx} -> Display Name: {display_name}")
    
    print("\nSample Keyword Mappings:")
    sample_keywords = list(embedding_data['keyword_id_to_index'].items())[:3]
    for keyword_id, idx in sample_keywords:
        display_name = embedding_data['index_to_keyword_display'][idx]
        print(f"Keyword ID {keyword_id} -> Index {idx} -> Display Name: {display_name}")
    
    print("\nDictionary verification completed successfully.")
    
except Exception as e:
    print(f"Error loading or processing embedding dictionaries: {e}")
    sys.exit(1) 