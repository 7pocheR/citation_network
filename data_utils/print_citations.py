import json
import sys
from pprint import pprint

# Load the citation network data
try:
    with open('test_dataset.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded test_dataset.json with {len(data['papers'])} papers")
except Exception as e:
    print(f"Error loading test_dataset.json: {e}")
    sys.exit(1)

# Load the generated paper
try:
    with open('test_paper.json', 'r', encoding='utf-8') as f:
        paper = json.load(f)
    print(f"Loaded test_paper.json with keys: {list(paper.keys())}")
except Exception as e:
    print(f"Error loading test_paper.json: {e}")
    sys.exit(1)

# Print paper information
print("\n" + "="*80)
print("GENERATED PAPER INFORMATION")
print("="*80)
print(f"Title: {paper.get('title', 'Unknown')}")
print(f"ID: {paper.get('id', 'Unknown')}")
print(f"Publication Date: {paper.get('publication_date', 'Unknown')}")

# Print topics
print("\n" + "-"*80)
print("TOPICS")
print("-"*80)
if 'topics' in paper and paper['topics']:
    for i, topic in enumerate(paper['topics']):
        print(f"Topic {i+1}:")
        print(f"  ID: {topic.get('id', 'Unknown')}")
        print(f"  Display Name: {topic.get('display_name', 'Unknown')}")
        print(f"  Field: {topic.get('field', 'Unknown')}")
        print(f"  Score: {topic.get('score', 0):.4f}")
else:
    print("No topics found")

# Print keywords
print("\n" + "-"*80)
print("KEYWORDS")
print("-"*80)
if 'keywords' in paper and paper['keywords']:
    for i, keyword in enumerate(paper['keywords']):
        print(f"Keyword {i+1}: {keyword}")
else:
    print("No keywords found")

# Print feature vector summary (if present)
print("\n" + "-"*80)
print("FEATURE VECTOR SUMMARY")
print("-"*80)
if 'features' in paper and paper['features']:
    features = paper['features'][0]  # First (and usually only) feature vector
    # Print summary statistics rather than the full vector
    print(f"Feature vector dimension: {len(features)}")
    print(f"Mean value: {sum(features)/len(features):.4f}")
    print(f"Min value: {min(features):.4f}")
    print(f"Max value: {max(features):.4f}")
    # Print top 5 features by absolute magnitude
    feature_magnitudes = [(i, abs(val)) for i, val in enumerate(features)]
    feature_magnitudes.sort(key=lambda x: x[1], reverse=True)
    print("\nTop 5 features by magnitude:")
    for i, (idx, magnitude) in enumerate(feature_magnitudes[:5]):
        print(f"  Feature {idx}: {features[idx]:.4f} (magnitude: {magnitude:.4f})")
else:
    print("No feature vector found")

# Print latent vector summary (if present)
if 'z' in paper and paper['z']:
    print("\n" + "-"*80)
    print("LATENT VECTOR SUMMARY")
    print("-"*80)
    z = paper['z'][0]  # First (and usually only) latent vector
    print(f"Latent vector dimension: {len(z)}")
    print(f"Mean value: {sum(z)/len(z):.4f}")
    print(f"Min value: {min(z):.4f}")
    print(f"Max value: {max(z):.4f}")
else:
    print("\nNo latent vector found")

# Get the top citations
print("\n" + "="*80)
print("TOP CITATIONS")
print("="*80)
if 'top_citations' in paper and len(paper['top_citations']) > 0:
    top_citations = paper['top_citations'][0]
    print(f"Found {len(top_citations)} top citations")
    
    # Print information about each cited paper
    for i, idx in enumerate(top_citations):
        print(f"\nCitation {i+1} (Index {idx}):")
        
        # We need to map the index to a paper ID
        papers = list(data['papers'].items())
        if idx < len(papers):
            paper_id, paper_info = papers[idx]
            print(f"  Paper ID: {paper_id}")
            print(f"  Title: {paper_info.get('title', 'Unknown')}")
            print(f"  Publication Date: {paper_info.get('publication_date', 'Unknown')}")
            
            # Print topics for this cited paper if available
            if 'topics' in paper_info and paper_info['topics']:
                topic_names = [t.get('display_name', 'Unknown') for t in paper_info['topics']]
                print(f"  Topics: {', '.join(topic_names)}")
            
            # Print authors
            authors = paper_info.get('authors', [])
            if authors:
                author_names = [a.get('name', 'Unknown') for a in authors]
                print(f"  Authors: {', '.join(author_names)}")
            else:
                print("  Authors: None")
                
            # If citation scores are available, print them
            if 'citation_scores' in paper:
                # The citation_scores structure can be complex, handle with care
                try:
                    # If it's a nested list, get the appropriate score
                    if isinstance(paper['citation_scores'], list) and len(paper['citation_scores']) > 0:
                        if isinstance(paper['citation_scores'][0], list):
                            score = paper['citation_scores'][0][i] if i < len(paper['citation_scores'][0]) else "N/A"
                        else:
                            score = paper['citation_scores'][i] if i < len(paper['citation_scores']) else "N/A"
                        
                        if isinstance(score, (int, float)):
                            print(f"  Citation Score: {score:.4f}")
                        else:
                            print(f"  Citation Score: {score}")
                except Exception as e:
                    print(f"  Citation Score: Error retrieving ({str(e)})")
        else:
            print(f"  Invalid index {idx} (out of range, max index is {len(papers)-1})")
else:
    print("No top_citations field found in the generated paper") 