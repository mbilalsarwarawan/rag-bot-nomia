import requests
import streamlit as st

def get_api_response(question, session_id, model, organization_id, workspace_id, file_id=None):
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    data = {
        "question": question,
        "model": model,
        "organization_id":organization_id,
        "workspace_id": workspace_id
    }
    if session_id:
        data["session_id"] = session_id
    if file_id:
        data["file_id"]=file_id
        print(file_id)

    try:
        response = requests.post("http://localhost:8000/chat", headers=headers, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API request failed with status code {response.status_code}: {response.text}")
            return None
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

def upload_document(organization_id,workspace_id, file_id, file):
    print("Uploading file...")
    try:
        files = {"file": (file.name, file, file.type)}
        response = requests.post("http://localhost:8000/upload-doc", files=files, data={"organization_id":organization_id, "workspace_id": workspace_id, "file_id":file_id})
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to upload file. Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"An error occurred while uploading the file: {str(e)}")
        return None

def list_documents(organization_id,workspace_id):
    try:
        response = requests.get(f"http://localhost:8000/list-docs/organization/{organization_id}/workspace/{workspace_id}")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch document list. Error: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        st.error(f"An error occurred while fetching the document list: {str(e)}")
        return []

def delete_document(organization_id,workspace_id, file_id):
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    data = {"organization_id":organization_id, "workspace_id": workspace_id, "file_id":file_id}

    try:
        response = requests.post("http://localhost:8000/delete-doc", headers=headers, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to delete document. Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"An error occurred while deleting the document: {str(e)}")
        return None

def fetch_organizations():
    """ Fetch all organizations from API """
    response = requests.get("http://localhost:8000/organizations/")
    if response.status_code == 200:
        return response.json()
    return []

def fetch_workspaces(organization_id):
    """ Fetch all workspaces from API based on selected organization """
    response = requests.get(f"http://localhost:8000/workspaces/{organization_id}")
    if response.status_code == 200:
        return response.json()
    return []