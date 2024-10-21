import pandas as pd
import chromadb
import uuid


class Portfolio:
    def __init__(self, file_path="Webapp\Resource\skills_portfolio_links.xlsx"):
        self.file_path = file_path
        self.data = pd.read_excel(file_path)
        self.chroma_client = chromadb.PersistentClient('vectorstore')
        self.collection = self.chroma_client.get_or_create_collection(name="portfolio")

    def load_portfolio(self):
        if not self.collection.count():
            for _, row in self.data.iterrows():
                    self.collection.add(documents=row["Skills"],
                       metadatas={"links": row["Portfolio and Project Links"]},
                       ids=[str(uuid.uuid4())])
    
    def query_links(self, skills):
        return self.collection.query(query_texts=skills, n_results=2).get('metadatas', [])