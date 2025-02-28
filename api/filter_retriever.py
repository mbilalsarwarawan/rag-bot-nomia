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
        qdrant_filter = self._build_qdrant_filter(self.metadata_filter)
        kwargs = {**kwargs, "filter": qdrant_filter}
        return super()._get_relevant_documents(query, run_manager=run_manager, **kwargs)

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: Any, **kwargs: Any
    ) -> List[Document]:
        qdrant_filter = self._build_qdrant_filter(self.metadata_filter)
        kwargs = {**kwargs, "filter": qdrant_filter}
        return await super()._aget_relevant_documents(query, run_manager=run_manager, **kwargs)
