from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime
from typing import List, Optional

class ModelName(str, Enum):
    Llama3 = "llama3.2"

class QueryInput(BaseModel):
    question: str
    session_id: str = Field(default=None)
    model: ModelName = Field(default=ModelName.Llama3)
    organization_id: int
    workspace_id: int
    file_id: Optional[int] = Field(default=None)

class QueryResponse(BaseModel):
    answer: str
    session_id: str
    model: ModelName

class DocumentInfo(BaseModel):
    file_id: int
    filename: str
    organization_id: int
    workspace_id: int
    upload_timestamp: datetime

class DeleteFileRequest(BaseModel):
    organization_id: int
    workspace_id: int
    file_id: int


class ListDoc(BaseModel):
    organization_id: int
    workspace_id: int


class FileItem(BaseModel):
    heading: str
    content: Optional[str] = None  # Content can be null

class FileUpload(BaseModel):
    file_id: int
    filename: str
    organization_id: int
    workspace_id: int
    file: List[FileItem]

class FileRecord(BaseModel):
    file_id: int
    filename: str
    organization_id: int
    workspace_id: int
