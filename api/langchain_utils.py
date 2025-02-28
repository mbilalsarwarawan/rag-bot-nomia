from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from qdrant_utils import get_org_workspace_vectorstore
from langchain_core.vectorstores import VectorStoreRetriever
from typing import Any, Dict, List
from pydantic import Field
from langchain.schema import Document

class FilteredVectorStoreRetrieverWithFilter(VectorStoreRetriever):
    metadata_filter: Dict[str, Any] = Field(..., description="Simple metadata filter to apply.")

    def _build_qdrant_filter(self, metadata_filter: Dict[str, Any]) -> Dict[str, Any]:

        return {"must": [{"key": key, "match": {"value": value}} for key, value in metadata_filter.items()]}

    def _get_relevant_documents(
        self, query: str, *, run_manager: Any, **kwargs: Any
    ) -> List[Document]:
        # Build a Qdrant-compatible filter from our simple metadata filter.
        qdrant_filter = self._build_qdrant_filter(self.metadata_filter)
        # Merge with any additional kwargs.
        kwargs = {**kwargs, "filter": qdrant_filter}
        return super()._get_relevant_documents(query, run_manager=run_manager, **kwargs)

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: Any, **kwargs: Any
    ) -> List[Document]:
        qdrant_filter = self._build_qdrant_filter(self.metadata_filter)
        kwargs = {**kwargs, "filter": qdrant_filter}
        return await super()._aget_relevant_documents(query, run_manager=run_manager, **kwargs)

output_parser = StrOutputParser()

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an AI assistant whose job is to reply to user queries with detailed explanation based on the context provided to you. Your task is to generate an answer to the query using only the details in the context. Do not add any external information or assumptions.

'user query'
'context'

Instructions:
1. If no context is provided, reply with: "Do not have enough information".
2. First, verify if the provided context contains sufficient and relevant information to answer the user query. if the context provided not match the user query simply reply "Do no have enough information". Do not make suppositions when there is no direct information.
3. If the context is relevant and detailed enough, generate a very detailed answer covering all the points also citing context provided to you.
4. Start your answer directly, do not include phrases like according to given context and etc.


Proceed with the response.
"""),
    ("system", "context: {context}"),
    ("human", "{input}")
])

def get_rag_chain(query, organization_id: str, workspace_id: str, model: str = "llama3.2", k: int = 3, file_id: str | None = None):
    if not (organization_id and workspace_id):
        raise ValueError("Both organization_id and workspace_id are required.")

    llm = OllamaLLM(model="llama3.2")

    workspace_vectorstore = get_org_workspace_vectorstore(organization_id, workspace_id)

    search_kwargs = {"k": k}
    retriever = workspace_vectorstore.as_retriever(search_kwargs=search_kwargs)
    print(retriever.invoke(query))
    print(file_id)
    print("\n before")
    if file_id:
        retriever = FilteredVectorStoreRetrieverWithFilter(
            vectorstore=workspace_vectorstore,
            search_kwargs=search_kwargs,
            metadata_filter={"metadata.template_id": int(file_id)},
            search_type=retriever.search_type  # carry over any search configuration
        )
        print(retriever.invoke(query))


    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain
