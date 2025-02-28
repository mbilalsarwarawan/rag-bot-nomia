from langchain_ollama import OllamaLLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from filter_retriever import FilteredVectorStoreRetrieverWithFilter
from prompt import qa_prompt
from qdrant_utils import get_vectorstore





def get_rag_chain(query, organization_id: int, workspace_id: int, model: str = "llama3.2", k: int = 3, file_id: int | None = None):
    if not (organization_id and workspace_id):
        raise ValueError("Both organization_id and workspace_id are required.")

    llm = OllamaLLM(model=model)

    workspace_vectorstore = get_vectorstore(organization_id, workspace_id)

    search_kwargs = {"k": k}
    retriever = workspace_vectorstore.as_retriever(search_kwargs=search_kwargs)
    print(retriever.invoke(query))
    print(file_id)
    print("\n before")
    if file_id:
        retriever = FilteredVectorStoreRetrieverWithFilter(
            vectorstore=workspace_vectorstore,
            search_kwargs=search_kwargs,
            metadata_filter={"metadata.template_id": file_id},
            search_type=retriever.search_type  # carry over any search configuration
        )
        print(retriever.invoke(query))


    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain
