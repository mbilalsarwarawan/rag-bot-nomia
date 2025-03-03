import json
import time
from fastapi import FastAPI
from pydantic_models import QueryInput, QueryResponse, DocumentInfo, DeleteFileRequest, ListDoc, FileUpload, ListDoc, \
    FileRecord
from langchain_utils import get_rag_chain
from db_utils import get_all_documents, insert_document, \
    delete_document_record, update_document_record, get_all_organizations, get_all_workspaces
from qdrant_utils import index_document_to_chroma, delete_doc_from_chroma, update_document_splits
import uuid
import logging
from fastapi import UploadFile, File, HTTPException
import os
import shutil
logging.basicConfig(filename='app.log', level=logging.INFO)
app = FastAPI()
import re

def remove_think_tags(text):
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned_text.strip()

@app.post("/chat", response_model=QueryResponse)
def chat(query_input: QueryInput):
    session_id = query_input.session_id
    logging.info(f"Session ID: {session_id}, User Query: {query_input.question}, Model: {query_input.model.value}")
    if not session_id:
        session_id = str(uuid.uuid4())

    

    if query_input.file_id:
        rag_chain = get_rag_chain(
            query=query_input.question,
            organization_id=query_input.organization_id,
            workspace_id=query_input.workspace_id,
            file_id=query_input.file_id,
            model=query_input.model.value
        )
    else:
        rag_chain = get_rag_chain(
            query=query_input.question,
            organization_id=query_input.organization_id,
            workspace_id=query_input.workspace_id,
            model=query_input.model.value,
            file_id=None
        )

    answer = rag_chain({
        "input": query_input.question,
    })
    answer=answer["answer"]
    answer = remove_think_tags(answer)

    logging.info(f"Session ID: {session_id}, AI Response: {answer}")
    return QueryResponse(answer=answer, session_id=session_id, model=query_input.model)


@app.post("/upload-doc")
def upload_and_index_document(file: FileUpload):
    temp_file_path = f"temp_{file.organization_id}_{file.workspace_id}.json"

    try:
        # Ensure required attributes exist
        if not file.organization_id or not file.workspace_id or not file.file:
            raise HTTPException(status_code=400, detail="Missing required JSON fields.")



        with open(temp_file_path, "w", encoding="utf-8") as json_file:
            json.dump(file.model_dump(), json_file, ensure_ascii=False, indent=4)
        # Insert the document record into the database
        file_id = insert_document(
            file_id=file.file_id,
            filename=temp_file_path,  # Store the temp file name as filename
            organization_id=file.organization_id,
            workspace_id=file.workspace_id
        )

        # Index the document in ChromaDB (modify this function to read JSON if needed)
        success = index_document_to_chroma(temp_file_path, file.organization_id, file.workspace_id, file.file_id)

        if not success:
            delete_document_record(file_id, file.organization_id, file.workspace_id)
            raise HTTPException(status_code=500, detail="Failed to index JSON data.")

        return {
            "message": "JSON data has been successfully uploaded and indexed.",
            "file_id": file_id
        }

    except HTTPException as http_err:
        raise http_err  # Re-raise FastAPI HTTPExceptions

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

    finally:
        # Clean up the temporary JSON file after processing
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.get("/list-docs/organization/{organization_id}/workspace/{workspace_id}", response_model=list[DocumentInfo])
def list_documents(organization_id: str, workspace_id: str):
    return get_all_documents(organization_id, workspace_id)

@app.post("/delete-doc")
def delete_document(request: DeleteFileRequest):
    # Delete from Chroma
    chroma_delete_success = delete_doc_from_chroma(request.organization_id, request.workspace_id,request.file_id)

    if chroma_delete_success:
        # If successfully deleted from Chroma, delete from our database
        db_delete_success = delete_document_record(request.file_id,  request.organization_id, request.workspace_id)
        if db_delete_success:
            return {"message": f"Successfully deleted document with file_id {request.file_id} from the system."}
        else:
            return {"error": f"Deleted from Chroma but failed to delete document with file_id {request.file_id} from the database."}
    else:
        return {"error": f"Failed to delete document with file_id {request.file_id} from Chroma."}


@app.post("/update-doc")
def update_document(file : FileUpload):

    temp_file_path = f"temp_{file.filename}.json"

    try:
        # Save the uploaded file to a temporary file
        with open(temp_file_path, "w", encoding="utf-8") as json_file:
            json.dump(file.model_dump(), json_file, ensure_ascii=False, indent=4)

        update_document_record(file.file_id, file.organization_id,file.workspace_id,file.filename)

        success = update_document_splits(temp_file_path, file.organization_id, file.workspace_id, file.file_id)

        if success:
            return {
                "message": f"File {file.filename} has been successfully updated and indexed.",
                "file_id": file.file_id
            }
        else:
            delete_document_record(file.file_id, file.organization_id, file.workspace_id)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to index {file.filename}."
            )
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# --- FastAPI Endpoints ---
@app.post("/add-document/")
def add_document(doc: FileRecord):
    """ API endpoint to add a document, ensuring organization/workspace exists """
    try:
        return {"message": "Document added successfully",
                "file_id": insert_document(doc.file_id, doc.filename, doc.organization_id, doc.workspace_id)}

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- FastAPI Endpoints ---
@app.get("/organizations/")
def fetch_organizations():
    """ API endpoint to get all organizations """
    orgs = get_all_organizations()
    return orgs if orgs else {"message": "No organizations found"}

@app.get("/workspaces/{organization_id}")
def fetch_workspaces(organization_id: str):
    """ API endpoint to get all workspaces for a given organization """
    workspaces = get_all_workspaces(organization_id)
    return workspaces if workspaces else {"message": "No workspaces found"}