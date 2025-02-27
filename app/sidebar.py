import streamlit as st
import uuid  # To generate unique file IDs
from api_utils import upload_document, list_documents, delete_document, fetch_organizations, fetch_workspaces


def display_sidebar():
    # Sidebar: Model Selection
    model_options = ["llama3.2"]
    st.sidebar.selectbox("Select Model", options=model_options, key="model")

    # Fetch organizations dynamically
    organizations = fetch_organizations()
    org_options = ["Select a company"] + [org["id"] for org in organizations]

    # Organization Selection
    selected_organization = st.sidebar.selectbox(
        "Choose a company:",
        org_options,
        index=0,
        key="selected_organization"
    )

    # Store selected organization ID
    if selected_organization != "Select a company":
        st.session_state["organization_id"] = selected_organization
    else:
        st.session_state["organization_id"] = None

    # Fetch workspaces dynamically when an organization is selected
    workspaces = fetch_workspaces(st.session_state["organization_id"]) if st.session_state["organization_id"] else []
    workspace_options = ["Select a workspace"] + [ws["id"] for ws in workspaces]

    # Workspace Selection
    selected_workspace = st.sidebar.selectbox(
        "Choose a workspace:",
        workspace_options,
        index=0,
        key="selected_workspace"
    )

    # Store selected workspace ID
    if selected_workspace != "Select a workspace":
        st.session_state["workspace_id"] = selected_workspace
    else:
        st.session_state["workspace_id"] = None



 # Refresh list after upload

    # Sidebar: List Documents
    st.sidebar.header("Uploaded Documents")
    if st.sidebar.button("Refresh Document List"):
        with st.spinner("Refreshing..."):
            st.session_state["documents"] = list_documents(
                st.session_state["organization_id"], st.session_state["workspace_id"]
            )
            st.session_state["selected_document_id"]= None

    # Initialize document list if not present
    if "documents" not in st.session_state:
        st.session_state["documents"] = list_documents(
            st.session_state["organization_id"], st.session_state["workspace_id"]
        )

    documents = st.session_state["documents"]
    if documents:
        for doc in documents:
            st.sidebar.text(f"{doc['filename']} (ID: {doc['file_id']}, Uploaded: {doc['upload_timestamp']})")


        selected_file_id = st.sidebar.selectbox(
            "Select a particular document to chat.",
            options=["None"]+[doc['file_id'] for doc in documents],
            format_func=lambda x: "None" if x=="None" else next(doc['filename'] for doc in documents if doc['file_id'] == x),
            key="selected_file_id"  # Unique key added
        )

        st.session_state["selected_document_id"]=None if selected_file_id =="None" else selected_file_id