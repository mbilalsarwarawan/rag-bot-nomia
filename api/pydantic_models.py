from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime
from typing import List, Optional

class ModelName(str, Enum):
    DEEPSEEK_R1 = "deepseek-r1:1.5b"

class QueryInput(BaseModel):
    question: str
    session_id: str = Field(default=None)
    model: ModelName = Field(default=ModelName.DEEPSEEK_R1)
    organization_id: str
    workspace_id: str
    file_id: Optional[str] = Field(default=None)

class QueryResponse(BaseModel):
    answer: str
    session_id: str
    model: ModelName

class DocumentInfo(BaseModel):
    file_id: str
    filename: str
    organization_id: str
    workspace_id: str
    upload_timestamp: datetime

class DeleteFileRequest(BaseModel):
    organization_id: str
    workspace_id: str
    file_id: str


class ListDoc(BaseModel):
    organization_id: str
    workspace_id: str


class FileItem(BaseModel):
    heading: str
    content: Optional[str] = None  # Content can be null

class FileUpload(BaseModel):
    file_id: str
    filename: str
    organization_id: str
    workspace_id: str
    file: List[FileItem]

class FileRecord(BaseModel):
    file_id: str
    filename: str
    organization_id: str
    workspace_id: str
