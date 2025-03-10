import json

# Load the generated paper
with open('test_paper.json', 'r', encoding='utf-8') as f:
    paper = json.load(f)

print(f"\nPaper: {paper.get('title', 'Unknown')} (ID: {paper.get('id', 'Unknown')})")
print(f"Publication Date: {paper.get('publication_date', 'Unknown')}")

# Print topics with interpretation
print("\nTopics:")
if 'topics' in paper and paper['topics']:
    # Sort topics by score to see the most relevant ones first
    sorted_topics = sorted(paper['topics'], key=lambda x: x.get('score', 0), reverse=True)
    
    print("  Primary research areas (sorted by relevance):")
    for i, topic in enumerate(sorted_topics):
        score = topic.get('score', 0)
        relevance = "High" if score > 0.6 else "Medium" if score > 0.4 else "Low"
        print(f"  {i+1}. {topic.get('display_name', 'Unknown')} - {relevance} relevance ({score:.4f})")
    
    # Find the dominant field
    fields = [topic.get('field', 'Unknown') for topic in paper['topics']]
    dominant_field = max(set(fields), key=fields.count)
    print(f"\n  Primary field: {dominant_field}")
    
    # Calculate average topic score
    avg_score = sum(topic.get('score', 0) for topic in paper['topics']) / len(paper['topics'])
    print(f"  Average topic relevance score: {avg_score:.4f}")
    
    # Interpret topic distribution
    score_variance = sum((topic.get('score', 0) - avg_score)**2 for topic in paper['topics']) / len(paper['topics'])
    if score_variance < 0.01:
        print("  Topic interpretation: This paper has evenly distributed topics, suggesting interdisciplinary research.")
    elif len([t for t in paper['topics'] if t.get('score', 0) > 0.6]) > 2:
        print("  Topic interpretation: This paper has multiple strong focus areas, suggesting a multidisciplinary approach.")
    elif len([t for t in paper['topics'] if t.get('score', 0) > 0.6]) == 1:
        print("  Topic interpretation: This paper has a singular strong focus area, suggesting specialized research.")
    else:
        print("  Topic interpretation: This paper has a moderate distribution of topics.")
else:
    print("  None found")

# Print keywords
print("\nKeywords:")
if 'keywords' in paper and paper['keywords']:
    for keyword in paper['keywords']:
        print(f"  - {keyword}")
else:
    print("  None found")

# Load the citation network data for paper information
try:
    with open('test_dataset.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    papers_dict = data['papers']
    papers_list = list(papers_dict.items())
except Exception as e:
    print(f"Error loading citation data: {e}")
    papers_dict = {}
    papers_list = []

# Print top citations with years
print("\nTop Citations:")
if 'top_citations' in paper and len(paper['top_citations']) > 0:
    top_citations = paper['top_citations'][0]
    
    # Group citations by year
    years = {}
    for i, idx in enumerate(top_citations):
        if idx < len(papers_list):
            paper_id, paper_info = papers_list[idx]
            year = paper_info.get('publication_date', '')[:4]
            if year:
                years[year] = years.get(year, 0) + 1
    
    # Print citations
    for i, idx in enumerate(top_citations):
        if idx < len(papers_list):
            paper_id, paper_info = papers_list[idx]
            year = paper_info.get('publication_date', 'Unknown')[:4]
            print(f"  {i+1}. {paper_info.get('title', 'Unknown')} ({year})")
        else:
            print(f"  {i+1}. Unknown paper (Index {idx})")
    
    # Print year distribution
    if years:
        print("\n  Citation year distribution:")
        for year in sorted(years.keys()):
            print(f"    {year}: {years[year]} papers")
        
        # Calculate recency of citations
        current_year = 2025  # The publication year of the generated paper
        avg_year = sum(int(year) * count for year, count in years.items()) / sum(years.values())
        avg_age = current_year - avg_year
        print(f"\n  Average citation age: {avg_age:.1f} years")
        if avg_age < 3:
            print("  Citation recency: Very recent literature (cutting edge)")
        elif avg_age < 5:
            print("  Citation recency: Recent literature")
        else:
            print("  Citation recency: Includes older foundational works")
else:
    print("  None found")

print("\nFeature Vector:")
if 'features' in paper and paper['features']:
    features = paper['features'][0]
    print(f"  Dimension: {len(features)}")
    feature_magnitudes = [(i, abs(val)) for i, val in enumerate(features)]
    feature_magnitudes.sort(key=lambda x: x[1], reverse=True)
    print("  Top 5 features by magnitude:")
    for i, (idx, magnitude) in enumerate(feature_magnitudes[:5]):
        print(f"    Feature {idx}: {features[idx]:.4f} (magnitude: {magnitude:.4f})")
else:
    print("  None found") 