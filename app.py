import sqlite3
import contextlib
import json
from pydantic import BaseModel, Field
from typing import List, Dict, Tuple, Optional
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from together import Together
import numpy as np
from collections import defaultdict
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Knowledge Graph API")

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_CONFIG = {
    "triplets_db": os.path.join(BASE_DIR, "triplets_new.db"),  
    "definitions_db": os.path.join(BASE_DIR, "relations_new.db"),
    "news_db": os.path.join(BASE_DIR, "cnnhealthnews2.db"),
    "news_faiss": os.path.join(BASE_DIR, "news_index_compressed"),
    "triplets_faiss": os.path.join(BASE_DIR, "triplets_index_compressed"),
    "triplets_table": "triplets", 
    "definitions_table": "relations", 
    "head_column": "head_entity",
    "relation_column": "relation", 
    "tail_column": "tail_entity",
    "definition_column": "definition",
    "link_column": "link",
    "title_column": "column",
    "content_column": "content"
}

class GraphNode(BaseModel):
    id: str
    label: str
    type: str = "entity"

class GraphEdge(BaseModel):
    source: str
    target: str
    relation: str
    definition: Optional[str] = None

class GraphData(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]

class TripletData(BaseModel):
    head: str
    relation: str
    tail: str

class RelationDefinition(BaseModel):
    relation: str
    definition: str

class RetrieveTripletsResponse(BaseModel):
    triplets: List[TripletData]
    relations: List[RelationDefinition]

class NewsItem(BaseModel):
    url: str
    content: str
    preview: str
    title: str

class QueryRequest(BaseModel):
    query: str
    
class QueryResponse(BaseModel):
    answer: str
    triplets: List[TripletData]
    relations: List[RelationDefinition]
    news_items: List[NewsItem]
    graph_data: GraphData
    
class ExtractedInformationNews(BaseModel):
    extracted_information: str = Field(description="Extracted information")
    links: list = Field(description="citation links")

class ExtractedInformation(BaseModel):
    extracted_information: str = Field(description="Extracted information")

@contextlib.contextmanager
def get_triplets_db():
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_CONFIG["triplets_db"])
        yield conn
    finally:
        if conn:
            conn.close()

@contextlib.contextmanager
def get_news_db():
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_CONFIG["news_db"])
        yield conn
    finally:
        if conn:
            conn.close()

@contextlib.contextmanager
def get_definitions_db():
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_CONFIG["definitions_db"])
        yield conn
    finally:
        if conn:
            conn.close()

def retrieve_triplets(query: str) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str]]]:
    """
    Args:
        query (str): User query
        
    Returns:
        Tuple containing:
        - List of triplets: [(head, relation, tail), ...]
        - List of relations with definitions: [(relation, definition), ...]
    """
    load_dotenv() 
    API_KEY = os.environ.get("TOGETHER_API_KEY")
    client = Together(api_key = API_KEY)
    
    dummy_embeddings = FakeEmbeddings(size=768)
    triplets_store = FAISS.load_local(
    DATABASE_CONFIG["triplets_faiss"], dummy_embeddings, allow_dangerous_deserialization=True
    )
    triplets_store.index.nprobe = 100
    triplets_store._normalize_L2 = True
    triplets_store.distance_strategy = DistanceStrategy.COSINE

    response = client.embeddings.create(
      model = "Alibaba-NLP/gte-modernbert-base",
      input = query
    )

    emb = np.array(response.data[0].embedding)
    emb = emb / np.linalg.norm(emb)
    
    related_head_entity = []
    result_triplets = triplets_store.similarity_search_with_score_by_vector(emb, k=100)
    for res, score in result_triplets:
        if score > 0.7:
            related_head_entity.append(res)
            
    try:
        all_triplets = []
        with get_triplets_db() as conn:
            head_col = DATABASE_CONFIG["head_column"]
            rel_col = DATABASE_CONFIG["relation_column"] 
            tail_col = DATABASE_CONFIG["tail_column"]
        
            for head_entity in related_head_entity:
                he = head_entity.page_content
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM triplets WHERE head_entity = (?)", ([he]))
                rows = cursor.fetchall()
                triplets = [(str(row[0]), str(row[1]), str(row[2])) for row in rows]
                all_triplets += triplets
            
        all_relations = []
        relations = [relation for _, relation, _ in all_triplets]
        with get_definitions_db() as conn:
            rel_col = DATABASE_CONFIG["relation_column"] 
            def_col = DATABASE_CONFIG["definition_column"]
        
            for rel in set(relations):
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM relations WHERE relation = (?)", ([rel]))
                rows = cursor.fetchall()
                relation = [(str(row[0]), str(row[1])) for row in rows]
                all_relations += relation

        return all_triplets, all_relations
        
    except Exception as e:
        print(f"Error in retrieve_triplets: {e}")
        return [], []

def retrieve_news(query: str) -> Dict[str, str]:
    """
    Args:
        query (str): User query
        
    Returns: Tuple
        - Related content
        - Links of the related content
    """
    load_dotenv() 
    API_KEY = os.environ.get("TOGETHER_API_KEY")
    client = Together(api_key = API_KEY)
    
    dummy_embeddings = FakeEmbeddings(size=768)
    news_store = FAISS.load_local(
        DATABASE_CONFIG["news_faiss"], dummy_embeddings, allow_dangerous_deserialization=True
    )
    news_store.index.nprobe = 100
    news_store._normalize_L2 = True
    news_store.distance_strategy = DistanceStrategy.COSINE
    
    news_store._normalize_L2 = True
    news_store.distance_strategy = DistanceStrategy.COSINE

    response = client.embeddings.create(
      model = "Alibaba-NLP/gte-modernbert-base",
      input = query
    )

    emb = np.array(response.data[0].embedding)
    emb = emb / np.linalg.norm(emb)

    related_news_content = []
    result_news= news_store.similarity_search_with_score_by_vector(emb, k=500)
    for res, score in result_news:
        if score > 0.7:
            print(score)
            related_news_content.append(res)
    
    news_dict = defaultdict(list)
    links = [res.metadata["link"] for res in related_news_content]
    for idx, link in enumerate(links):
        news_dict[link].append(related_news_content[idx].page_content)
        
    content_only = [". ".join(sentences) for sentences in news_dict.values()]
    
    return content_only, links


def extract_information_from_triplets(query: str,
                                      triplets: List[Tuple[str, str, str]], 
                                      relations: List[Tuple[str, str]]) -> str:
    """
    Args:
        triplets: List of triplets from retrieve_triplets
        relations: List of relation definitions from retrieve_triplets
        
    Returns:
        str: Extracted information from triplets
    """
    system_prompt = f'''Given a a list of relational triplets and a list of relation and its definition. Extract the information from the triplets to answer query question.
    If there is no related or useful information can be extracted from the triplets to answer the query question, inform "No related information found."
    Give the output in paragraphs form narratively, you can explain the reason behind your answer in detail."
    '''
    
    user_prompt = f'''
    query question: {query}
    list of triplets: {triplets}
    list of relations and their definition: {relations}
    extracted information:
    '''
    
    load_dotenv() 
    API_KEY = os.environ.get("TOGETHER_API_KEY")
    client = Together(api_key = API_KEY)

    response = client.chat.completions.create(
        model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        temperature = 0,
        messages=[{
            "role": "system",
            "content": [
                {"type": "text", "text":system_prompt}
            ]
        },
                {
           "role": "user",
           "content": [
               {"type": "text", "text":user_prompt},
            ]
        }]
    )
    
    return response.choices[0].message.content

def extract_information_from_news(query: str, news_list: Dict[str, str]) -> Tuple[str, List[str]]:
    """   
    Args:
        news_list: List from retrieve_news
        
    Returns:
        Extracted information string
    """
    
    system_prompt = f'''Given a list of some information related to the query, extract all important information from the list to answer query question.
    Every item in the list represent one information, if the information is ambiguous (e.g. contains unknown pronoun to which it refers), do not use that information to answer the query.
    You don't have to use all the information, only use the information that has clarity and a good basis, but try to use as many information as possible.
    If there is no related or useful information can be extracted from the news information to answer the query question, write "No related information found." as the extracted_information output.
    Give the extracted_information output in paragraphs form detailedly.
    The output must be in this form: {{"extracted_information": <output paragraphs>}}
    '''
    
    user_prompt = f'''
    query: {query}
    news list: {news_list}
    output:
    '''

    load_dotenv() 
    API_KEY = os.environ.get("TOGETHER_API_KEY")
    client = Together(api_key = API_KEY)

    response = client.chat.completions.create(
       model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
       response_format={
            "type": "json_schema",
            "schema": ExtractedInformation.model_json_schema(),
       },
       temperature = 0,
       messages=[{
           "role": "system",
           "content": [
               {"type": "text", "text":system_prompt}
           ]
       },
                {
           "role": "user",
           "content": [
               {"type": "text", "text":user_prompt},
           ]
       }]
    )
    response = json.loads(response.choices[0].message.content)
    info = response['extracted_information']
    
    return info

def extract_information(query:str, triplet_info: str, news_info: str, language:str) -> str:
    """   
    Args:
        triplet_info: Information extracted from triplets
        news_info: Information extracted from news
        
    Returns:
        str: Final answer for the user
    """
    system_prompt = f'''Given information from two sources, combine the information and make a comprehensive and informative paragraph that answer the query.
    Make sure the output paragraph includes all crucial information and given in detail.
    If there is no related or useful information can be extracted from the triplets to answer the query question, inform "No related information found."
    Remember this paragraph will be shown to user, so make sure it is based on facts and data, also use appropriate language.
    The output must be in this form and in {language} language: {{"extracted_information": <output paragraphs>}}
    '''
    
    user_prompt = f'''
    query: {query}
    first source: {triplet_info}
    second source: {news_info}
    extracted information:
    '''

    load_dotenv() 
    API_KEY = os.environ.get("TOGETHER_API_KEY")
    client = Together(api_key = API_KEY)

    response = client.chat.completions.create(
       model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
       response_format={
            "type": "json_schema",
            "schema": ExtractedInformation.model_json_schema(),
       },
       temperature = 0,
       messages=[{
           "role": "system",
           "content": [
               {"type": "text", "text":system_prompt}
           ]
       },
                {
           "role": "user",
           "content": [
               {"type": "text", "text":user_prompt},
           ]
       }]
    )
    
    response = json.loads(response.choices[0].message.content)
    answer = response["extracted_information"]
    return answer

def news_preview(links: list[str]) -> Tuple[str, str, str]:
    try:
        preview_contents = []
        with get_news_db() as conn:
            for i in links:
                cursor = conn.cursor()
                cursor.execute("SELECT link, title, content FROM CNNHEALTHNEWS2 WHERE link = (?)", ([i]))
                rows = cursor.fetchall()
                prevs = [(str(row[0]), str(row[1]), str(row[2])) for row in rows]
                preview_contents += prevs

        return preview_contents
    
    except Exception as e:
        print(f"Error in news_preview: {e}")
        return ("", "", "")

class Language(BaseModel):
    query: str = Field(description="Translated query")
    language: str = Field(description="Query's language")
    
def query_language(query):
    system_prompt = f'''Your task is to determine what language the question is written in and translate it to english if it is not in English.
    The output must be in this form: {{query: <translated query>, language: <query's language>}}
    '''
    
    user_prompt = f'''
    query: {query}
    output:
    '''
    
    load_dotenv() 
    API_KEY = os.environ.get("TOGETHER_API_KEY")
    client = Together(api_key = API_KEY)
    
    response = client.chat.completions.create(
       model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
       response_format={
            "type": "json_schema",
            "schema": Language.model_json_schema(),
       },
       temperature = 0,
       messages=[{
           "role": "system",
           "content": [
               {"type": "text", "text":system_prompt}
           ]
       },
                {
           "role": "user",
           "content": [
               {"type": "text", "text":user_prompt},
           ]
       }])

    return json.loads(response.choices[0].message.content)

#API ENDPOINTS

@app.get("/", response_class=FileResponse)
def serve_index():
    return FileResponse("index.html")

@app.get("/explorepage.html", response_class=FileResponse)
def serve_explore_page():
    return FileResponse("explorepage.html")

@app.get("/search.html", response_class=FileResponse)
def serve_search_page():
    return FileResponse("search.html")

@app.post("/api/query", response_model=QueryResponse)
def process_query(request: QueryRequest):
    """Process user query and return response"""
    try:
        # Step 1: Retrieve triplets
        query = request.query
        query = query_language(query)
        
        triplets_data, relations_data = retrieve_triplets(query['query'])
        
        # Step 2: Retrieve news
        news_list, news_links = retrieve_news(query['query'])
        
        # Step 3: Extract information from triplets
        triplet_info = extract_information_from_triplets(query['query'], triplets_data, relations_data)
        
        # Step 4: Extract information from news
        news_info = extract_information_from_news(query['query'], news_list)
        
        # Step 5: Generate final answer
        final_answer = extract_information(query['query'], triplet_info, news_info, query['language'])
        
        # Convert triplets to response format
        triplets = [TripletData(head=t[0], relation=t[1], tail=t[2]) for t in triplets_data]
        relations = [RelationDefinition(relation=r[0], definition=r[1]) for r in relations_data]
        
        # Convert news to response format with previews
        news_prev = news_preview(news_links)
        news_items = []
        for url, title, content in news_prev:
            preview = content[:300] + "..." if len(content) > 300 else content
            news_items.append(NewsItem(
                url=url,
                content=content,
                preview=preview,
                title=title
            ))
        
        # Create mini graph data for visualization
        nodes_set = set()
        edges = []
        
        for triplet in triplets_data:
            head, relation, tail = triplet
            nodes_set.add(head)
            nodes_set.add(tail)
            
            definition = "No definition available"
            for rel, def_text in relations_data:
                if rel == relation:
                    definition = def_text
                    break
            
            edges.append(GraphEdge(
                source=head,
                target=tail,
                relation=relation,
                definition=definition
            ))
        
        nodes = [GraphNode(id=node, label=node) for node in nodes_set]
        graph_data = GraphData(nodes=nodes, edges=edges)
        
        return QueryResponse(
            answer=final_answer,
            triplets=triplets,
            relations=relations,
            news_items=news_items,
            graph_data=graph_data
        )
        
    except Exception as e:
        print(f"Error in process_query: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.get("/api/graph", response_model=GraphData)
def get_graph_data(
    search: Optional[str] = None
):
    """Get complete graph data for explore page"""
    
    try:
        # Build dynamic query based on configuration
        table = DATABASE_CONFIG["triplets_table"]
        head_col = DATABASE_CONFIG["head_column"]
        rel_col = DATABASE_CONFIG["relation_column"] 
        tail_col = DATABASE_CONFIG["tail_column"]
        
        base_query = f"SELECT {head_col}, {rel_col}, {tail_col} FROM {table}"
        params = []
        
        if search:
            base_query += f" WHERE {head_col} LIKE ? OR {tail_col} LIKE ? OR {rel_col} LIKE ?"
            search_term = f"%{search}%"
            params = [search_term, search_term, search_term]
        
        base_query += " LIMIT 1000"
        
        # Get triplets
        with get_triplets_db() as conn:
            cursor = conn.execute(base_query, params)
            triplets = cursor.fetchall()
        
        # Get definitions
        with get_definitions_db() as conn:
            def_table = DATABASE_CONFIG["definitions_table"]
            def_col = DATABASE_CONFIG["definition_column"]
            rel_col_def = DATABASE_CONFIG["relation_column"]
            
            def_cursor = conn.execute(f"SELECT {rel_col_def}, {def_col} FROM {def_table}")
            definitions = {row[0]: row[1] for row in def_cursor.fetchall()}
        
        # Build nodes and edges
        nodes_set = set()
        edges = []
        
        for triple in triplets:
            head = triple[0]
            relation = triple[1]
            tail = triple[2]
            
            # Add entities to nodes set
            nodes_set.add(head)
            nodes_set.add(tail)
            
            # Create edge with definition
            edge = GraphEdge(
                source=head,
                target=tail,
                relation=relation,
                definition=definitions.get(relation, "No definition available")
            )
            edges.append(edge)
        
        # Convert nodes set to list of GraphNode objects
        nodes = [GraphNode(id=node, label=node) for node in nodes_set]
        
        return GraphData(nodes=nodes, edges=edges)
        
    except Exception as e:
        print(f"Error in get_graph_data: {e}")
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")

if __name__ == "__main__":
    print("Starting Knowledge Graph API...")
    print(f"Triplets DB: {DATABASE_CONFIG['triplets_db']}")
    print(f"Definitions DB: {DATABASE_CONFIG['definitions_db']}")
    
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

    