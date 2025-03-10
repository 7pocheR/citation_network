import pickle
import numpy as np
import json
from create_embedding_dictionaries import create_one_hot_encoding

# Load the embedding dictionaries
print("Loading embedding dictionaries...")
with open("embedding_dictionaries.pkl", "rb") as f:
    embedding_data = pickle.load(f)

# Create the one-hot encoding function
paper_to_one_hot = create_one_hot_encoding(embedding_data)

# Load a sample paper to test on
print("Loading test paper...")
try:
    with open("test_paper.json", "r", encoding="utf-8") as f:
        test_paper = json.load(f)
    
    # Display paper basic info
    print(f"\nTest Paper: {test_paper.get('title', 'Unknown title')}")
    print(f"Publication date: {test_paper.get('publication_date', 'Unknown')}")
    
    # Extract topics and keywords
    topics = test_paper.get('topics', [])
    keywords = test_paper.get('keywords', [])
    
    print(f"\nTopics in paper ({len(topics)}):")
    for topic in topics:
        topic_id = topic.get('id', 'Unknown')
        display_name = topic.get('display_name', 'Unknown')
        score = topic.get('score', 1.0)
        print(f"- {display_name} (ID: {topic_id}, Score: {score:.2f})")
    
    print(f"\nKeywords in paper ({len(keywords)}):")
    for keyword in keywords:
        keyword_id = keyword.get('id', 'Unknown')
        display_name = keyword.get('display_name', 'Unknown')
        score = keyword.get('score', 1.0)
        print(f"- {display_name} (ID: {keyword_id}, Score: {score:.2f})")
    
    # Generate one-hot encoding
    print("\nGenerating one-hot encoding...")
    topic_vector, keyword_vector = paper_to_one_hot(test_paper)
    
    # Check the dimensions
    print(f"Topic vector dimensions: {topic_vector.shape}")
    print(f"Keyword vector dimensions: {keyword_vector.shape}")
    
    # Count non-zero elements
    topic_nonzero = np.count_nonzero(topic_vector)
    keyword_nonzero = np.count_nonzero(keyword_vector)
    
    print(f"Non-zero elements in topic vector: {topic_nonzero} (matches topics in paper: {topic_nonzero == len(topics)})")
    print(f"Non-zero elements in keyword vector: {keyword_nonzero} (matches keywords in paper: {keyword_nonzero == len(keywords)})")
    
    # Print the non-zero elements to verify correct mapping
    print("\nNon-zero elements in topic vector:")
    for idx, value in enumerate(topic_vector):
        if value > 0:
            topic_id = embedding_data['index_to_topic_id'][idx]
            display_name = embedding_data['index_to_topic_display'][idx]
            print(f"- Index {idx}: {display_name} (ID: {topic_id}, Value: {value:.2f})")
    
    print("\nNon-zero elements in keyword vector:")
    for idx, value in enumerate(keyword_vector):
        if value > 0:
            keyword_id = embedding_data['index_to_keyword_id'][idx]
            display_name = embedding_data['index_to_keyword_display'][idx]
            print(f"- Index {idx}: {display_name} (ID: {keyword_id}, Value: {value:.2f})")
    
    # Combined feature vector
    combined_vector = np.concatenate([topic_vector, keyword_vector])
    print(f"\nCombined feature vector dimensions: {combined_vector.shape}")
    print(f"Non-zero elements in combined vector: {np.count_nonzero(combined_vector)}")
    
    print("\nOne-hot encoding validation successful!")
    
except Exception as e:
    print(f"Error testing one-hot encoding: {e}") 