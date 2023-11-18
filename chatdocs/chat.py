from typing import Any, Dict, Optional

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.messages import AIMessage, HumanMessage
from rich import print
from rich.markup import escape
from rich.panel import Panel

from .chains import make_conversation_chain


def print_answer(text: str) -> None:
    print(f"[bright_cyan]{escape(text)}", end="", flush=True)


class PrintCallback(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print_answer(token)


def chat(config: Dict[str, Any], query: Optional[str] = None) -> None:
    llm = make_conversation_chain(config)
    messages = []

    interactive = not query
    print()
    if interactive:
        print("Type your query below and press Enter.")
        print("Type 'exit' or 'quit' or 'q' to exit the application.\n")
    while True:
        print("[bold]Q: ", end="", flush=True)
        if interactive:
            query = input()
        else:
            print(escape(query))
        print()
        if query.strip() in ["exit", "quit", "q"]:
            print("Exiting...\n")
            break
        print("[bold]A:", end="", flush=True)

        res = llm(
            { "question": query, "chat_history": messages },
            callbacks=[PrintCallback],
        )

        print()
        for doc in res["source_documents"]:
            source, content = doc.metadata["source"], doc.page_content
            print(
                Panel(
                    f"[bright_blue]{escape(source)}[/bright_blue]\n\n{escape(content)}"
                )
            )
        
        print()
        print_answer(res["answer"])
        
        if not interactive:
            break

        messages.append(HumanMessage(content=query))
        messages.append(AIMessage(content=res["answer"]))
