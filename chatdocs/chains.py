from typing import Any, Dict, Optional

from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQA

from .llms import get_llm
from .vectorstores import get_vectorstore


def get_retrieval_qa(
    config: Dict[str, Any],
    *,
    selected_llm_index: int = 0,
    callbacks: Optional[list[BaseCallbackHandler]] = None,
) -> RetrievalQA:
    db = get_vectorstore(config)
    retriever = db.as_retriever(**config["retriever"])
    llm = get_llm(config, selected_llm_index=selected_llm_index, callbacks=callbacks)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
