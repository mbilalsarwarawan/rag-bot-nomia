from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.vectorstores import VectorStoreRetriever
from typing import Any, Dict, List
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_core.vectorstores import VectorStoreRetriever
from pydantic import Field, BaseModel
from langchain.schema import Document
import json
from langchain_text_splitters import RecursiveJsonSplitter
from qdrant_client.http.models import Distance, VectorParams
import requests
from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy import create_engine
import psycopg2


from langchain_community.utilities import SQLDatabase

db2 = SQLDatabase.from_uri("sqlite:///rag_app.db")


llm=ChatOllama(model="llama3.2")
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain import hub

prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")

toolkit2 = SQLDatabaseToolkit(db=db2, llm=llm)
from langgraph.prebuilt import create_react_agent

agent_executor2 = create_react_agent(llm, toolkit2.get_tools(), prompt=prompt_template.format(dialect=f"SQLite with table names {db2.get_context()}", top_k=100))
qdrant_client = QdrantClient(
    url="http://localhost:6333"
)

text_splitter = RecursiveJsonSplitter(max_chunk_size=2500, min_chunk_size=1500)
embedding_function = OllamaEmbeddings(model="nomic-embed-text")


def get_org_workspace_vectorstore(organization_id: str, workspace_id: str):
    collection_name = f"org_{organization_id}_workspace_{workspace_id}"
    collection = qdrant_client.collection_exists(collection_name=collection_name)
    if not collection:
        qdrant_client.create_collection(
            collection_name=collection_name, vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )

    vectorstore = QdrantVectorStore(client=qdrant_client, collection_name=collection_name, embedding=embedding_function)
    return vectorstore


# --- Part 1: Your existing RAG chain code ---

class FilteredVectorStoreRetrieverWithFilter(VectorStoreRetriever):
    metadata_filter: Dict[str, Any] = Field(..., description="Simple metadata filter to apply.")

    def _build_qdrant_filter(self, metadata_filter: Dict[str, Any]) -> Dict[str, Any]:
        return {"must": [{"key": key, "match": {"value": value}} for key, value in metadata_filter.items()]}

    def _get_relevant_documents(self, query: str, *, run_manager: Any, **kwargs: Any) -> List[Document]:
        qdrant_filter = self._build_qdrant_filter(self.metadata_filter)
        kwargs = {**kwargs, "filter": qdrant_filter}
        return super()._get_relevant_documents(query, run_manager=run_manager, **kwargs)

    async def _aget_relevant_documents(self, query: str, *, run_manager: Any, **kwargs: Any) -> List[Document]:
        qdrant_filter = self._build_qdrant_filter(self.metadata_filter)
        kwargs = {**kwargs, "filter": qdrant_filter}
        return await super()._aget_relevant_documents(query, run_manager=run_manager, **kwargs)


output_parser = StrOutputParser()

qa_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an AI assistant whose job is to reply to user queries with detailed explanation based on the context provided.
Your task is to generate an answer using only the details in the context. Do not add external information or assumptions.

'user query'
'context'

Instructions:
1. If no context is provided, reply with: "Do not have enough information".
2. Verify that the context is relevant to the user query. If not, simply reply "Do not have enough information".
3. If the context is relevant, generate a very detailed answer covering all points, citing the 'template_id' and 'filename' from the context.
4. Start your answer directly.
5. From the context, cite the 'template_id' and 'filename' which justify your answer.
"""
    ),
    ("system", "context: {context}"),
    ("human", "{input}")
])


def format_documents_with_metadata(docs: List[Document]) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs):
        template_id = doc.metadata.get("template_id", "N/A")
        filename = doc.metadata.get("filename", "N/A")
        content = doc.page_content
        formatted_docs.append(
            f"Document {i}:\n\"template_id\": \"{template_id}\"\n\"filename\": \"{filename}\"\n\"content\": \"{content}\"\n"
        )
    return "\n".join(formatted_docs)


def get_rag_chain(query: str, organization_id: str, workspace_id: str, model: str = "llama3.2", k: int = 3,
                  file_id: str | None = None):
    if not (organization_id and workspace_id):
        raise ValueError("Both organization_id and workspace_id are required.")
    # Initialize the language model for retrieval
    llm = ChatOllama(model=model)
    workspace_vectorstore = get_org_workspace_vectorstore(organization_id, workspace_id)
    search_kwargs = {"k": k}
    retriever = workspace_vectorstore.as_retriever(search_kwargs=search_kwargs)
    if file_id:
        retriever = FilteredVectorStoreRetrieverWithFilter(
            vectorstore=workspace_vectorstore,
            search_kwargs=search_kwargs,
            metadata_filter={"metadata.template_id": int(file_id)},
            search_type=retriever.search_type  # carry over search configuration
        )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    def invoke_chain(input_dict: Dict[str, str]):
        docs = retriever.invoke(input_dict["input"])
        formatted_context = format_documents_with_metadata(docs)
        return rag_chain.invoke({"input": formatted_context})

    return invoke_chain


# --- Part 2: SQL chain implementation ---

# Assume you have a SQL database connection.
from langchain_community.utilities import SQLDatabase

# Replace with your actual connection string (ensure it uses read-only credentials)
sql_db = SQLDatabase.from_uri("sqlite:///rag_app.db")


def run_sql_chain(query: str) -> str:
    # Customize a prompt that forces the generation of only SELECT queries.
    safe_sql_prompt = f"""
You are a SQL expert. Given the following question, generate a syntactically correct SQL SELECT query to retrieve data.
Do not generate any DML commands (INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE).
If the question is not about data retrieval, output "I don't know".
Question: {query}
"""
    # Use ChatOllama to generate the SQL query.
    generated_sql = ChatOllama(model="llama3.2").invoke(safe_sql_prompt).content.strip()
    # Check for forbidden keywords
    print(generated_sql)
    forbidden_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE"]
    for kw in forbidden_keywords:
        if kw in generated_sql.upper():
            return f"Error: Generated query contains forbidden keyword {kw}"
    # Execute the query on the SQL database.
    result = sql_db.run(generated_sql)
    return result


# --- Part 3: Agentic router to select between RAG and SQL chains ---

def agent_router(query: str) -> str:
    router_prompt = f"""
You are an expert query router. Given the following query: "{query}"
Decide whether to answer this query by retrieving background knowledge from documents (RAG) or by querying the live SQL database for real-time data.
Respond with exactly one word: "RAG" or "SQL". Do not include any extra text. if query contain live, sql or any similar word go with SQL otherwise RAG
"""
    response = ChatOllama(model="llama3.2").invoke(router_prompt).content.strip().upper()
    print(f"Hello {response}")
    if response not in ["RAG", "SQL"]:
        return "RAG"  # default to RAG if unclear
    return response


def run_agentic_query(query: str, organization_id: str, workspace_id: str, file_id: str | None = None) -> str:
    # Use the router to decide which chain to run.
    route = agent_router(query)
    if route == "SQL":
        query = f"{query} organization_id = {organization_id} and workspace_id = {workspace_id}"
        events = agent_executor2.stream(
            {"messages": [("user", query)]},
            stream_mode="values",
        )
        for event in events:
            event["messages"][-1].pretty_print()
        return event["messages"][-1].content
    else:
        rag_chain = get_rag_chain(query, organization_id, workspace_id, model="llama3.2", k=3, file_id=file_id)
        # Invoke the rag chain with the query as input.
        return rag_chain({"input": query})


# --- Example usage ---
organization_id = "5"  # Replace with actual organization ID
workspace_id = "723"  # Replace with actual workspace ID
# For a query likely needing real-time data:
sql_query = "what is  MUTUAL CONFIDENTIALITY AND NONDISCLOSURE AGREEMENT"
result_sql = run_agentic_query(sql_query, organization_id, workspace_id)
print("SQL Chain Result:", result_sql)
