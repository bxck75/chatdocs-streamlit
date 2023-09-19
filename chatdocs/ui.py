# allow relative imports when running with streamlit
from streamlit import runtime
if runtime.exists() and not __package__:
    from pathlib import Path
    __package__ = Path(__file__).parent.name

import argparse
from typing import Union

import streamlit as st
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.streamlit import StreamlitCallbackHandler

from .config import get_config
from .chains import get_retrieval_qa


def init_messages() -> None:
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = []


def print_state_messages():
    def find_role(message: Union[SystemMessage, HumanMessage, AIMessage]) -> str:
        """
        Identify role name from langchain.schema object.
        """
        if isinstance(message, SystemMessage):
            return "system"
        if isinstance(message, HumanMessage):
            return "user"
        if isinstance(message, AIMessage):
            return "assistant"
        raise TypeError("Unknown message type.")

    for message in st.session_state.messages:
        with st.chat_message(find_role(message)):
            st.markdown(message.content)


@st.cache_data
def load_config(config_path):
    return get_config(config_path)


@st.cache_resource
def load_qa_chain(config):
    print_callback = StreamingStdOutCallbackHandler()
    st_callback = StreamlitCallbackHandler(st.container())
    return get_retrieval_qa(config, callbacks=[print_callback, st_callback])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, nargs='?',
                    help='Custom path to a chatdocs.yml configuration file.')
    args = parser.parse_args()
    
    st.set_page_config(
        page_title="ChatDocs",
        page_icon="📚"
    )
    st.title("ChatDocs")
    st.sidebar.title("Options")

    init_messages()
    print_state_messages()

    config = load_config(args.config_path)
    qa = load_qa_chain(config)

    if prompt := st.chat_input("Enter a query"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append(HumanMessage(content=prompt))

        with st.spinner("LLM is typing ..."):
            placeholder = st.empty()
            full_response = ""
            response = qa(prompt)

            with st.chat_message("assistant"):
                if isinstance(response, str):
                    full_response += response
                    placeholder.markdown(full_response)
                else:
                    full_response = response["result"]
                    placeholder.markdown(full_response)
            st.session_state.messages.append(AIMessage(content=full_response))

            for doc in response["source_documents"]:
                source, content = doc.metadata["source"], doc.page_content
                with st.expander(label=source):
                    st.markdown(content)


if __name__ == "__main__":
    main()