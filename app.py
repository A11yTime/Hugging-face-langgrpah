
import streamlit as st

from rag_pipeline import (
    ask_question,
    build_vector_store,
    load_documents,
)

st.set_page_config(
    page_title="Document RAG Chatbot",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Document RAG Chatbot")
st.caption(
    "Ask questions about your PDFs and text files "
    "using LangGraph and Hugging Face."
)

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Knowledge base")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if st.button("Build knowledge base", use_container_width=True):
        if not uploaded_files:
            st.warning("Upload at least one document first.")
        else:
            try:
                with st.spinner("Reading and indexing documents..."):
                    documents = load_documents(uploaded_files)
                    store = build_vector_store(documents)

                    st.session_state.vector_store = store
                    st.session_state.indexed_files = [
                        f.name for f in uploaded_files
                    ]
                    st.session_state.messages = []

                st.success(
                    f"Indexed {len(documents)} document sections."
                )

            except Exception as exc:
                st.error(f"Could not index documents: {exc}")

    if st.session_state.indexed_files:
        st.subheader("Indexed files")
        for filename in st.session_state.indexed_files:
            st.write(f"📄 {filename}")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if st.session_state.vector_store is None:
    st.info(
        "Upload documents in the sidebar and click "
        "'Build knowledge base' to begin."
    )

# Display previous messages.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("View retrieved sources"):
                for source in message["sources"]:
                    page = source.get("page")
                    location = (
                        f", page {page}" if page else ""
                    )

                    st.markdown(
                        f"**[{source['chunk']}] "
                        f"{source['source']}{location}**"
                    )
                    st.caption(
                        source["content"][:1200]
                    )

question = st.chat_input(
    "Ask a question about your documents...",
    disabled=st.session_state.vector_store is None,
)

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching documents and generating answer..."):
                result = ask_question(
                    question,
                    st.session_state.vector_store,
                )

            st.markdown(result["answer"])

            if result["sources"]:
                with st.expander("View retrieved sources"):
                    for source in result["sources"]:
                        page = source.get("page")
                        location = (
                            f", page {page}" if page else ""
                        )

                        st.markdown(
                            f"**[{source['chunk']}] "
                            f"{source['source']}{location}**"
                        )
                        st.caption(
                            source["content"][:1200]
                        )

            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
            })

        except Exception as exc:
            st.error(
                "The request failed. Check your Hugging Face "
                "token, model availability, and application logs."
            )
            st.exception(exc)
