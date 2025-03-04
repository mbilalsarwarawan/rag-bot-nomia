from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from qdrant_utils import get_org_workspace_vectorstore
from langchain_core.vectorstores import VectorStoreRetriever
from typing import Any, Dict, List
from pydantic import Field
from langchain.schema import Document
import json
from pydantic import BaseModel, Field,validator
from typing import List



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
    ("system","""You are an AI assistant whose job is to reply to user queries with detailed explanation based on the context provided to you. Your task is to generate an answer to the query using only the details in the context. Do not add any external information or assumptions.

'user query'
'context'

Instructions:
1. If no context is provided, reply with: "Do not have enough information".
2. First, verify if the provided context contains sufficient and relevant information to answer the user query. if the context provided not match the user query simply reply "Do no have enough information". Do not make suppositions when there is no direct information.
3. If the context is relevant and detailed enough, generate a very detailed answer covering all the points also citing template_id and filename which justify the answer.
4. Start your answer directly, do not include phrases like according to given context and etc.
5. From the context cite the 'template_id' and 'filename' which justify your answer.

Response Format:
##Answer:
##Reference:
Template_id:
Filename:

"""),
    ("system", "context: {context}  "),
    ("human", "{input}")
])



def get_rag_chain(query, organization_id: str, workspace_id: str, model: str = "llama3.2", k: int = 3, file_id: str | None = None):
    if not (organization_id and workspace_id):
        raise ValueError("Both organization_id and workspace_id are required.")

    # Initialize the language model
    llm = ChatOllama(model=model)
    workspace_vectorstore = get_org_workspace_vectorstore(organization_id, workspace_id)

    search_kwargs = {"k": k}
    retriever = workspace_vectorstore.as_retriever(search_kwargs=search_kwargs)

    # If file_id is provided, filter the vectorstore using metadata
    if file_id:
        retriever = FilteredVectorStoreRetrieverWithFilter(
            vectorstore=workspace_vectorstore,
            search_kwargs=search_kwargs,
            metadata_filter={"metadata.template_id": int(file_id)},
            search_type=retriever.search_type  # carry over any search configuration
        )

    # Helper: format each retrieved document with its metadata (template_id and filename)
    def format_documents_with_metadata(docs):
        formatted_docs = []
        for i, doc in enumerate(docs):
            template_id = doc.metadata.get("template_id", "N/A")
            filename = doc.metadata.get("filename", "N/A")
            # You can decide whether to use the full content or just a snippet
            content = doc.page_content
            formatted_docs.append(
                f"Document {i}:\n"
                f"\"template_id\": \"{template_id}\"\n"
                f"\"filename\": \"{filename}\"\n"
                f"\"content\": \"{content}\"\n"
            )
        return "\n".join(formatted_docs)


    # Create the question-answer chain with the custom prompt
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Create the retrieval chain that combines the retriever and QA chain.
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # (Optional) You might want to wrap rag_chain.invoke so that it first formats the retrieved documents.
    # For example:
    def invoke_chain(input_dict):
        # First, use the retriever to get a list of documents
        docs = retriever.invoke(input_dict["input"])
        # Format the retrieved docs with metadata
        formatted_context = format_documents_with_metadata(docs)
        # Now, add the formatted context to the input and invoke the chain
        return rag_chain.invoke({"input":formatted_context+input_dict["input"]})

    # Return the modified chain (or a wrapped function) that now uses the formatted context
    return invoke_chain
