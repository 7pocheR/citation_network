import pickle
import numpy as np
import json
from create_embedding_dictionaries import create_one_hot_encoding

# Load the embedding dictionaries
print("Loading embedding dictionaries...")
with open("embedding_dictionaries.pkl", "rb") as f:
    embedding_data = pickle.load(f)

# Get some real topic and keyword IDs from our dictionaries
topic_ids = list(embedding_data['topic_id_to_index'].keys())[:5]  # Take first 5 topics
keyword_ids = list(embedding_data['keyword_id_to_index'].keys())[:3]  # Take first 3 keywords

# Create a mock paper with real topic and keyword IDs
mock_paper = {
    "title": "Mock Paper for Testing One-Hot Encoding",
    "publication_date": "2025-03-05",
    "topics": [
        {
            "id": topic_ids[0],
            "display_name": embedding_data['index_to_topic_display'][embedding_data['topic_id_to_index'][topic_ids[0]]],
            "score": 0.9
        },
        {
            "id": topic_ids[1],
            "display_name": embedding_data['index_to_topic_display'][embedding_data['topic_id_to_index'][topic_ids[1]]],
            "score": 0.8
        },
        {
            "id": topic_ids[2],
            "display_name": embedding_data['index_to_topic_display'][embedding_data['topic_id_to_index'][topic_ids[2]]],
            "score": 0.7
        }
    ],
    "keywords": [
        {
            "id": keyword_ids[0],
            "display_name": embedding_data['index_to_keyword_display'][embedding_data['keyword_id_to_index'][keyword_ids[0]]],
            "score": 0.85
        },
        {
            "id": keyword_ids[1],
            "display_name": embedding_data['index_to_keyword_display'][embedding_data['keyword_id_to_index'][keyword_ids[1]]],
            "score": 0.75
        }
    ]
}

# Create the one-hot encoding function
paper_to_one_hot = create_one_hot_encoding(embedding_data)

# Display mock paper basic info
print(f"\nMock Paper: {mock_paper['title']}")
print(f"Publication date: {mock_paper['publication_date']}")

# Display topics and keywords
topics = mock_paper['topics']
keywords = mock_paper['keywords']

print(f"\nTopics in paper ({len(topics)}):")
for topic in topics:
    topic_id = topic['id']
    display_name = topic['display_name']
    score = topic['score']
    print(f"- {display_name} (ID: {topic_id}, Score: {score:.2f})")

print(f"\nKeywords in paper ({len(keywords)}):")
for keyword in keywords:
    keyword_id = keyword['id']
    display_name = keyword['display_name']
    score = keyword['score']
    print(f"- {display_name} (ID: {keyword_id}, Score: {score:.2f})")

# Generate one-hot encoding
print("\nGenerating one-hot encoding...")
topic_vector, keyword_vector = paper_to_one_hot(mock_paper)

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

# Save this vector to demonstrate the one-hot encoding
np.save("mock_paper_one_hot.npy", combined_vector)
print("\nSaved combined one-hot vector to 'mock_paper_one_hot.npy'")

print("\nOne-hot encoding validation successful!") 