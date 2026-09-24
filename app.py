import hashlib

import streamlit as st

from agents import CATEGORY_TO_DEPARTMENT
from config import (
    EMBEDDING_MODEL,
    MODEL,
    SIMILARITY_THRESHOLD,
    TOP_K,
)
from graph import create_graph
from rag import (
    create_chunks,
    load_documents,
    load_uploaded_pdf,
    VectorStore,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Enterprise Knowledge Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM UI
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.4rem;
        font-weight: 750;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #6b7280;
        margin-bottom: 1.2rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    """
    <div class="main-title">
        AI Enterprise Knowledge Assistant
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Grounded enterprise Q&A using RAG,
        FAISS, specialist agents and LangGraph.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []


# ============================================================
# BUILD KNOWLEDGE BASE
# ============================================================

@st.cache_resource(
    show_spinner="Building knowledge base..."
)
def build_resources(
    upload_payloads
):

    # --------------------------------------------------------
    # BASE DOCUMENTS
    # --------------------------------------------------------

    documents = load_documents(
        "documents"
    )

    # --------------------------------------------------------
    # UPLOADED DOCUMENTS
    # --------------------------------------------------------

    for payload in upload_payloads:

        filename = payload[0]
        department = payload[1]
        file_bytes = payload[2]

        uploaded_documents = (
            load_uploaded_pdf(
                file_bytes,
                filename,
                department
            )
        )

        documents.extend(
            uploaded_documents
        )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not documents:
        raise ValueError(
            "No PDF documents were found."
        )

    # --------------------------------------------------------
    # CHUNKING
    # --------------------------------------------------------

    chunks, metadata = create_chunks(
        documents
    )

    if not chunks:
        raise ValueError(
            "The PDF documents contain no extractable text."
        )

    # --------------------------------------------------------
    # VECTOR STORE
    # --------------------------------------------------------

    vector_store = VectorStore(
        chunks,
        metadata
    )

    # --------------------------------------------------------
    # LANGGRAPH
    # --------------------------------------------------------

    graph = create_graph(
        chunks,
        metadata,
        vector_store.index
    )

    return (
        documents,
        chunks,
        metadata,
        vector_store,
        graph,
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "Knowledge Base"
    )

    uploaded_files = st.file_uploader(
        "Upload additional PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help=(
            "Uploaded PDFs are indexed "
            "for this browser session."
        ),
    )

    department = st.selectbox(
        "Department for uploaded documents",
        [
            "hr",
            "technical",
            "projects",
        ],
    )

    # --------------------------------------------------------
    # PROCESS UPLOADS
    # --------------------------------------------------------

    if uploaded_files:

        existing_files = {
            (
                item["name"],
                item["department"]
            )
            for item
            in st.session_state.uploaded_files
        }

        changed = False

        for uploaded_file in uploaded_files:

            key = (
                uploaded_file.name,
                department
            )

            if key in existing_files:
                continue

            st.session_state.uploaded_files.append(
                {
                    "name":
                        uploaded_file.name,

                    "department":
                        department,

                    "bytes":
                        uploaded_file.getvalue(),
                }
            )

            changed = True

        if changed:

            build_resources.clear()

            st.rerun()

    # --------------------------------------------------------
    # CLEAR UPLOADS
    # --------------------------------------------------------

    if st.button(
        "Clear uploaded documents",
        use_container_width=True
    ):

        st.session_state.uploaded_files = []

        build_resources.clear()

        st.rerun()

    # --------------------------------------------------------
    # CLEAR CHAT
    # --------------------------------------------------------

    if st.button(
        "Clear conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

    st.divider()

    st.caption(
        f"LLM: {MODEL}"
    )

    st.caption(
        f"Embeddings: {EMBEDDING_MODEL}"
    )

    st.caption(
        f"Top-K: {TOP_K}"
    )

    st.caption(
        f"Similarity threshold: "
        f"{SIMILARITY_THRESHOLD}"
    )


# ============================================================
# PREPARE UPLOAD PAYLOAD
# ============================================================

upload_payloads = tuple(
    (
        item["name"],
        item["department"],
        item["bytes"],
    )
    for item
    in st.session_state.uploaded_files
)


# ============================================================
# INITIALIZE KNOWLEDGE BASE
# ============================================================

try:

    (
        documents,
        chunks,
        metadata,
        vector_store,
        graph,
    ) = build_resources(
        upload_payloads
    )

except Exception as error:

    st.error(
        "Knowledge base initialization failed: "
        f"{error}"
    )

    st.stop()


# ============================================================
# METRICS
# ============================================================

unique_documents = {
    document["source"]
    for document in documents
}

unique_departments = {
    document["department"]
    for document in documents
}

col1, col2, col3, col4 = st.columns(4)


with col1:
    st.metric(
        "PDF documents",
        len(unique_documents)
    )


with col2:
    st.metric(
        "Indexed chunks",
        len(chunks)
    )


with col3:
    st.metric(
        "Top-K",
        TOP_K
    )


with col4:
    st.metric(
        "Departments",
        len(unique_departments)
    )


# ============================================================
# ARCHITECTURE
# ============================================================

st.divider()

with st.expander(
    "How this system works"
):

    st.markdown(
        """
        **User → Manager Agent → Specialist Agent
        → Semantic Retrieval → Grounded Answer**

        **HR Specialist**
        - Leave
        - Attendance
        - Employee policies

        **Technical Specialist**
        - Programming
        - APIs
        - Deployment
        - Infrastructure

        **Project Specialist**
        - Project ownership
        - Project architecture
        - Project technologies

        **General Specialist**
        - Cross-domain company questions

        **RAG**
        - Sentence Transformers
        - FAISS
        - Similarity filtering

        **LLM**
        - Groq

        **Workflow**
        - LangGraph
        """
    )


# ============================================================
# DISPLAY OLD MESSAGES
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"]
            == "assistant"
            and message.get("sources")
        ):

            with st.expander(
                "Sources"
            ):

                for source in message[
                    "sources"
                ]:

                    st.markdown(
                        f"""
                        **{source['source']}**

                        Page: {source['page']}

                        Department: {source['department']}

                        Similarity:
                        {source['similarity']:.3f}
                        """
                    )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask about HR policies, technical documentation, or company projects..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )

    # --------------------------------------------------------
    # CONVERSATION HISTORY
    # --------------------------------------------------------

    history = (
        st.session_state.messages[:-1]
    )

    # --------------------------------------------------------
    # ASSISTANT
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Searching the knowledge base..."
        ):

            try:

                result = graph.invoke(
                    {
                        "question":
                            question,

                        "history":
                            history,
                    }
                )

                answer = result.get(
                    "answer",
                    (
                        "I could not find this "
                        "information in the available "
                        "company documents."
                    ),
                )

                sources = result.get(
                    "sources",
                    []
                )

                category = result.get(
                    "category",
                    "GENERAL"
                )

                # ------------------------------------------------
                # ANSWER
                # ------------------------------------------------

                st.markdown(
                    answer
                )

                st.caption(
                    f"Routed to: {category} • "
                    f"Department: "
                    f"{CATEGORY_TO_DEPARTMENT.get(category) or 'all'}"
                )

                # ------------------------------------------------
                # SOURCES
                # ------------------------------------------------

                if sources:

                    with st.expander(
                        "Sources"
                    ):

                        for source in sources:

                            st.markdown(
                                f"""
                                **{source['source']}**

                                Page: {source['page']}

                                Department:
                                {source['department']}

                                Similarity:
                                {source['similarity']:.3f}
                                """
                            )

                # ------------------------------------------------
                # SAVE MESSAGE
                # ------------------------------------------------

                st.session_state.messages.append(
                    {
                        "role":
                            "assistant",

                        "content":
                            answer,

                        "sources":
                            sources,
                    }
                )

            except Exception as error:

                error_message = (
                    "Something went wrong while "
                    f"processing the question: {error}"
                )

                st.error(
                    error_message
                )

                st.session_state.messages.append(
                    {
                        "role":
                            "assistant",

                        "content":
                            error_message,

                        "sources":
                            [],
                    }
                )