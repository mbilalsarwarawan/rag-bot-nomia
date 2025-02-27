import sqlite3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

DB_NAME = "rag_app.db"

app = FastAPI()


def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


# --- Create Tables ---
def create_tables():
    conn = get_db_connection()

    # Create organizations table
    conn.execute('''CREATE TABLE IF NOT EXISTS organization (
                        id TEXT PRIMARY KEY,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )''')

    # Create workspaces table
    conn.execute('''CREATE TABLE IF NOT EXISTS workspace (
                        id TEXT PRIMARY KEY,
                        organization_id TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (organization_id) REFERENCES organization(id) ON DELETE CASCADE
                    )''')

    # Create document store
    conn.execute('''CREATE TABLE IF NOT EXISTS document_store (
                        file_id TEXT PRIMARY KEY,
                        filename TEXT,
                        organization_id TEXT NOT NULL,
                        workspace_id TEXT NOT NULL,
                        upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (organization_id) REFERENCES organization(id) ON DELETE CASCADE,
                        FOREIGN KEY (workspace_id) REFERENCES workspace(id) ON DELETE CASCADE
                    )''')

    conn.commit()
    conn.close()


create_tables()


# --- Existing Functions ---
def insert_document_record(file_id, filename, organization_id, workspace_id):
    """ Inserts a document record """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        'INSERT INTO document_store (file_id, filename, organization_id, workspace_id) VALUES (?, ?, ?, ?)',
        (file_id, filename, organization_id, workspace_id)
    )
    conn.commit()
    conn.close()
    return file_id


def update_document_record(file_id, organization_id, workspace_id, new_filename):
    """ Updates the filename of a document """
    conn = get_db_connection()
    conn.execute(
        'UPDATE document_store SET filename = ? WHERE file_id = ? AND organization_id = ? AND workspace_id = ?',
        (new_filename, file_id, organization_id, workspace_id)
    )
    conn.commit()
    conn.close()
    return True


def delete_document_record(file_id, organization_id, workspace_id):
    """ Deletes a document record """
    conn = get_db_connection()
    conn.execute(
        'DELETE FROM document_store WHERE file_id = ? AND organization_id = ? AND workspace_id = ?',
        (file_id, organization_id, workspace_id)
    )
    conn.commit()
    conn.close()
    return True


def get_all_documents(organization_id, workspace_id):
    """ Fetch all documents for a workspace """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''SELECT file_id, filename, organization_id, workspace_id, upload_timestamp 
                      FROM document_store 
                      WHERE organization_id = ? AND workspace_id = ? 
                      ORDER BY upload_timestamp DESC''',
                   (organization_id, workspace_id))
    documents = cursor.fetchall()
    conn.close()
    return [dict(doc) for doc in documents]


# --- Helper Functions ---
def ensure_organization_exists(organization_id: str):
    """ Ensure organization exists; insert if not exists """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM organization WHERE id = ?", (organization_id,))
    if not cursor.fetchone():
        cursor.execute("INSERT INTO organization (id) VALUES (?)", (organization_id,))
        conn.commit()

    conn.close()


def ensure_workspace_exists(workspace_id: str, organization_id: str):
    """ Ensure workspace exists; insert if not exists """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM workspace WHERE id = ?", (workspace_id,))
    if not cursor.fetchone():
        cursor.execute("INSERT INTO workspace (id, organization_id) VALUES (?, ?)", (workspace_id, organization_id))
        conn.commit()

    conn.close()


# --- Insert Document with Auto-Ensure ---
def insert_document(file_id: str, filename: str, organization_id: str, workspace_id: str):
    """ Ensures organization and workspace exist, then inserts a document """
    # Ensure organization exists
    ensure_organization_exists(organization_id)

    # Ensure workspace exists
    ensure_workspace_exists(workspace_id, organization_id)

    # Insert document
    return insert_document_record(file_id, filename, organization_id, workspace_id)


def get_all_organizations():
    """ Fetch all organizations """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, created_at FROM organization ORDER BY created_at DESC")
    organizations = cursor.fetchall()
    conn.close()
    return [dict(org) for org in organizations]

# --- Fetch All Workspaces for an Organization ---
def get_all_workspaces(organization_id):
    """ Fetch all workspaces under a specific organization """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, organization_id, created_at FROM workspace WHERE organization_id = ? ORDER BY created_at DESC", (organization_id,))
    workspaces = cursor.fetchall()
    conn.close()
    return [dict(ws) for ws in workspaces]








