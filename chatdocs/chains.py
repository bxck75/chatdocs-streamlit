from typing import Any, Dict, Optional

from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQA

from .llms import get_llm
from .vectorstores import get_vectorstore


def get_retrieval_qa(
    config: Dict[str, Any],
    *,
    callbacks: Optional[list[BaseCallbackHandler]] = None,
) -> RetrievalQA:
    db = get_vectorstore(config)
    retriever = db.as_retriever(**config["retriever"])
    llm = get_llm(config, callbacks=callbacks)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
