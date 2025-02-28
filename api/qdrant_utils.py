from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from typing import List
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_community.document_loaders import TextLoader




qdrant_client = QdrantClient(
url="http://localhost:6333"
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)

embedding_function = OllamaEmbeddings(model="nomic-embed-text")

def get_vectorstore(organization_id: int, workspace_id: int):
    collection_name = f"org_{organization_id}_workspace_{workspace_id}"
    collection=qdrant_client.collection_exists(collection_name=collection_name)
    if not collection:
        qdrant_client.create_collection(
            collection_name=collection_name, vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )

    vectorstore=QdrantVectorStore(client=qdrant_client, collection_name=collection_name, embedding=embedding_function)
    return vectorstore

def load_and_split_document(file_path: str) -> List[Document]:
    try:
        if file_path.endswith('.txt'):
            loader = TextLoader(file_path)
            documents = loader.load()
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
        documents = text_splitter.split_documents(documents)
        return documents
    except Exception as e:
        print(f"Error loading document: {e}")



def index_document_to_chroma(file_path: str, organization_id: int, workspace_id: int , file_id: int, file_name: str) -> bool:
    try:
        splits = load_and_split_document(file_path)
        vectorstore = get_vectorstore(organization_id, workspace_id)
        for split in splits:
            split.metadata['file_id'] = file_id
            split.metadata['file_name'] = file_name
            split.metadata['organization_id'] = organization_id
            split.metadata['workspace_id'] = workspace_id

        vectorstore.add_documents(splits)
        return True
    except Exception as e:
        print(f"Error indexing document: {e}")
        return False



def delete_doc_from_chroma(organization_id: int, workspace_id: int, file_id: int) -> bool:
    try:
        vectorstore = get_vectorstore(organization_id, workspace_id)

        delete_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.template_id",
                    match=models.MatchValue(value=file_id),
                )
            ]
        )

        deletion_result = vectorstore.client.delete(
            collection_name=vectorstore.collection_name,
            points_selector=models.FilterSelector(
                filter=delete_filter
            )
        )
        return True

    except Exception as e:
        print(f"Error deleting document with file_id {file_id} from Qdrant: {e}")
        return False


def update_document_splits(file_path: str, organization_id: int, workspace_id: int, file_id: int, file_name: str) -> bool:
    try:
        if not delete_doc_from_chroma(organization_id, workspace_id, file_id):
            print(f"Failed to delete existing splits for file_id {file_id}")
            return False

        splits = load_and_split_document(file_path)
        if not splits:
            print("No splits returned from the document loader.")
            return False

        vectorstore = get_vectorstore(organization_id, workspace_id)
        for split in splits:
            split.metadata['file_id'] = file_id
            split.metadata['file_name'] = file_name
            split.metadata['organization_id'] = organization_id
            split.metadata['workspace_id'] = workspace_id

        vectorstore.add_documents(splits)
        return True
    except Exception as e:
        print(f"Error updating splits for file_id {file_id}: {e}")
        return False
