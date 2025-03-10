import requests
import json
import time
from collections import deque
from typing import Dict, Set, List, Optional
from datetime import datetime
from tqdm import tqdm

class OpenAlexCollector:
    """Collects citation network data from OpenAlex API"""
    
    def __init__(self):
        self.collected_papers: Dict = {}
        self.processed_ids: Set[str] = set()
        self.api_base = "https://api.openalex.org/works"
        
        # Configuration
        self.MAX_PRIMARY_PAPERS = 2000
        self.MAX_REFS_PER_PAPER = 0
        self.MIN_CITATIONS = 50
        self.MAX_ITERATIONS = 2
        self.SLEEP_TIME = 1
        self.START_YEAR = 2014  # Added time filter
        
        # Sources (top ML/AI/CV venues)
        self.SOURCES = [
            "S4306420609",  # NeurIPS (Neural Information Processing Systems)
            "S4306419644",  # ICML
            "S4306419637",  # ICLR
            "S4306417987",  # CVPR
            "S4363607701",  # CVPR 2022
            "S4363607795",  # CVPR 2009
            "S1960151631"   # Arxiv
        ]
        
        # Required fields for API
        self.FIELDS = [
            'doi',
            'title',
            'display_name',
            'publication_date',
            'cited_by_count',
            'citation_normalized_percentile',
            'topics',
            'keywords',
            'referenced_works_count',
            'referenced_works',
            'related_works',
            'abstract_inverted_index',
            'cited_by_api_url',
            'counts_by_year'
        ]
    
    def get_paper_data(self, work_id: Optional[str] = None, params: Optional[dict] = None) -> Optional[dict]:
        """Fetch paper data from OpenAlex API"""
        try:
            if work_id:
                response = requests.get(f"{self.api_base}/{work_id}")
            else:
                response = requests.get(self.api_base, params=params)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None
    
    def extract_paper_info(self, paper: dict) -> Optional[dict]:
        """Extract GNN-relevant fields from paper data"""
        if not paper:
            return None
        
        topics = paper.get('topics', [])
        return {
            'doi': paper.get('doi'),
            'title': paper.get('title'),
            'publication_date': paper.get('publication_date'),
            'cited_by_count': paper.get('cited_by_count'),
            'topics': [
                {
                    'id': t.get('id'),
                    'display_name': t.get('display_name'),
                    'score': t.get('score'),
                    'field': t.get('field', {}).get('display_name')
                }
                for t in topics
            ],
            'keywords': [
                {
                    'id': k.get('id'),
                    'display_name': k.get('display_name'),
                    'score': k.get('score')
                }
                for k in paper.get('keywords', [])
            ],
            'referenced_works': paper.get('referenced_works', []),
            'counts_by_year': paper.get('counts_by_year', [])
        }
    
    def is_valid_reference(self, paper: dict) -> bool:
        """Check if paper meets criteria"""
        if not paper:
            return False
        return (
            paper.get('cited_by_count', 0) >= self.MIN_CITATIONS
            and any(
                topic.get('field', {}).get('display_name') == "Computer Science"
                for topic in paper.get('topics', [])
            )
        )
    
    def collect_primary_papers(self) -> List[dict]:
        """Collect initial set of primary papers"""
        primary_papers = []
        
        # Fix filter syntax
        sources_filter = '|'.join(self.SOURCES)
        filter_string = f"primary_location.source.id:{sources_filter},publication_year:{self.START_YEAR}-"
        print(f"Using filter: {filter_string}")
        
        params = {
            'filter': filter_string,
            'sort': 'cited_by_count:desc',
            'per-page': 200,
            'cursor': '*',  # Start cursor
            'select': 'id,doi,title,publication_date,cited_by_count,topics,keywords,referenced_works,counts_by_year'
        }
        
        with tqdm(total=self.MAX_PRIMARY_PAPERS, desc="Collecting primary papers") as pbar:
            while len(primary_papers) < self.MAX_PRIMARY_PAPERS:
                try:
                    data = self.get_paper_data(params=params)
                    if not data or not data.get('results'):
                        break
                    
                    # Get all results from this page
                    new_papers = data['results']
                    if not new_papers:
                        break
                    
                    # Add papers from this page
                    remaining_slots = self.MAX_PRIMARY_PAPERS - len(primary_papers)
                    papers_to_add = new_papers[:remaining_slots]
                    primary_papers.extend(papers_to_add)
                    
                    # Update progress
                    pbar.update(len(papers_to_add))
                    
                    if len(primary_papers) >= self.MAX_PRIMARY_PAPERS:
                        break
                    
                    # Get next cursor
                    if data.get('meta', {}).get('next_cursor'):
                        params['cursor'] = data['meta']['next_cursor']
                    else:
                        break
                    
                    time.sleep(self.SLEEP_TIME)
                    
                except Exception as e:
                    print(f"Error: {e}")
                    break
        
        print(f"Final primary papers count: {len(primary_papers)}")
        return primary_papers
    
    def collect_network(self):
        """Main method to collect the citation network"""
        primary_papers = self.collect_primary_papers()
        print(f"Collected {len(primary_papers)} primary papers")  # Debug print
        
        papers_to_process = deque([(p, 0, True) for p in primary_papers])
        total_to_process = len(papers_to_process)
        
        with tqdm(total=total_to_process, desc="Building citation network") as pbar:
            while papers_to_process:
                paper, iteration, is_primary = papers_to_process.popleft()
                
                # Use the OpenAlex URL directly as ID
                paper_id = paper.get('id')
                
                if paper_id not in self.processed_ids:
                    # Process paper
                    paper_info = self.extract_paper_info(paper)
                    if paper_info:
                        paper_info['is_primary'] = is_primary
                        self.collected_papers[paper_id] = paper_info
                        self.processed_ids.add(paper_id)
                        pbar.update(1)
                    
                    # Add references if not at max iteration
                    if iteration < self.MAX_ITERATIONS:
                        ref_count = 0
                        for ref_id in paper.get('referenced_works', []):
                            if ref_count >= self.MAX_REFS_PER_PAPER:
                                break
                            if ref_id in self.processed_ids:
                                continue
                            
                            ref_paper = self.get_paper_data(work_id=ref_id)
                            time.sleep(self.SLEEP_TIME)
                            
                            if ref_paper and self.is_valid_reference(ref_paper):
                                papers_to_process.append((ref_paper, iteration + 1, False))
                                ref_count += 1
                                total_to_process += 1
                                pbar.total = total_to_process
        
        # Save network
        filename = f'citation_network_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'primary_papers': self.MAX_PRIMARY_PAPERS,
                    'max_refs_per_paper': self.MAX_REFS_PER_PAPER,
                    'min_citations': self.MIN_CITATIONS,
                    'max_iterations': self.MAX_ITERATIONS,
                    'sources': self.SOURCES,
                    'total_papers': len(self.collected_papers)
                },
                'papers': self.collected_papers
            }, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    collector = OpenAlexCollector()
    collector.collect_network()
