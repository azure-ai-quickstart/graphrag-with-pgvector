import asyncio
import logging
import sys
import tracemalloc
import tiktoken
import pandas as pd
import streamlit as st
import zipfile
import os
from dotenv import load_dotenv
from streamlit.runtime.uploaded_file_manager import UploadedFile

from libs.markdown import deal_md
from libs.pg_local_search import PgLocalSearch
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.indexer_adapters import (
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.input.loaders.dfs import store_entity_semantic_embeddings
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
from libs.pgvector import PgVectorStore
from libs.common import run_command, list_subdirectories, create_session_files
from io import StringIO

tracemalloc.start()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

notebook_dir = os.path.abspath("")
parent_dir = os.path.dirname(notebook_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)

llm = ChatOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    model=os.getenv('OPENAI_CHAT_MODEL_ID', 'gpt-4o-mini'),
    api_type=OpenaiApiType.OpenAI,
    max_retries=20,
)

with open("./dataset/data.zip", 'rb') as f:
    dataset_zip = f.read()

with open("./dataset/daqin.md", 'rb') as f:
    dataset_md = f.read()

with open("./dataset/daqin.txt", 'rb') as f:
    dataset_txt = f.read()

local_context_params = {
    "text_unit_prop": 0.5,
    "community_prop": 0.1,
    "conversation_history_max_turns": 5,
    "conversation_history_user_turns_only": True,
    "top_k_mapped_entities": 10,
    "top_k_relationships": 10,
    "include_entity_rank": True,
    "include_relationship_weight": True,
    "include_community_rank": False,
    "return_candidate_context": False,
    "embedding_vectorstore_key": EntityVectorStoreKey.ID,
    "max_tokens": 12_000,
}

llm_params = {
    "max_tokens": 2_000,
    "temperature": 0.0,
}


def deal_zip(uploaded_file: UploadedFile):
    extract_dir = f'/tmp/workshop/input/'
    if uploaded_file is not None:

        # Ensure the extraction directory exists
        os.makedirs(extract_dir, exist_ok=True)

        # Unzip the file directly from the upload buffer
        with zipfile.ZipFile(uploaded_file) as zip_ref:
            zip_ref.extractall(extract_dir)

        # List extracted files
        files = os.listdir(extract_dir)
        for f in files:
            if f.endswith(".md"):
                deal_md(extract_dir, f)


def dataset():
    st.markdown("### Step 1: Prepare your file")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            label="Download Sample ZIP (Markdown & Images)",
            data=dataset_zip,
            file_name="data.zip",
            mime='application/zip'
        )
    with col2:
        st.download_button(
            label="Download Sample Markdown",
            data=dataset_md,
            file_name="data.md",
            mime='application/text'
        )
    with col3:
        st.download_button(
            label="Download Sample TXT",
            data=dataset_txt,
            file_name="data.txt",
            mime='application/text'
        )


def upload_file():
    st.markdown("### Step 2: Upload your file")
    uploaded_file = st.file_uploader(
        label="file_uploader",
        type=['txt', 'md', 'zip'],
        accept_multiple_files=False,
        label_visibility="hidden"
    )

    if uploaded_file is not None:
        create_session_files()

        if uploaded_file.name.endswith('.zip'):
            deal_zip(uploaded_file, )

        if uploaded_file.name.endswith('.txt') or uploaded_file.name.endswith('.md'):
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            string_data = stringio.read()

            with open(f"/tmp/workshop/input/input.txt", "w") as f:
                f.write(string_data)
                st.success('File uploaded successfully.')

                if uploaded_file.name.endswith('.md'):
                    extract_dir = f'/tmp/workshop/input/'
                    deal_md(extract_dir, 'input.txt')


def build_index():
    base_path = f"/tmp/workshop"
    st.markdown("### Step 3: Build your Index")
    if st.button('Build Index'):
        with st.chat_message("user", avatar="avatars/ms.svg"):
            if not os.path.exists(base_path):
                st.error("Please upload a file first")
                return

            with st.spinner(f'Running the index pipeline ...'):
                run_command(f"python -m graphrag.index --verbose --root {base_path}")

            subdirectories = list_subdirectories(path=f"{base_path}/output")
            if len(subdirectories) == 0:
                raise Exception("No output by graphrag.index, please check log.")

            create_final_entities = f"{base_path}/output/{subdirectories[0]}/artifacts/create_final_entities.parquet"
            if not os.path.exists(create_final_entities):
                raise Exception(f"No {create_final_entities} by graphrag.index, please check log.")

            df = pd.read_parquet(create_final_entities)
            with st.expander(f"Entities {len(df)} items: Expand/Collapse"):
                st.write(df.head(n=1000))

            df = pd.read_parquet(f"{base_path}/output/{subdirectories[0]}/artifacts/create_final_relationships.parquet")
            with st.expander(f"Relationships {len(df)} items: Expand/Collapse"):
                st.write(df.head(n=1000))

            index_log_file = f"{base_path}/output/{subdirectories[0]}/reports/indexing-engine.log"
            lines = []
            with open(index_log_file, 'r') as f:
                log_content = f.read()
                for line in log_content.split('\n'):
                    if line.strip():
                        if "output_tokens" in line:
                            line = line.replace("graphrag.llm.base.rate_limiting_llm INFO perf - ", "")
                            lines.append(line)
            with st.expander(f"LLM Logs {len(lines)} items: Expand/Collapse"):
                st.write(lines)

            subdirectories = list_subdirectories(path=f"/tmp/workshop/output")
            if len(subdirectories) == 0:
                st.error("Your need to build index first.")
                return

            entities = get_entities(subdirectories[0])
            embedding_store = get_pg_store()
            embedding_store.truncate_table()
            store_entity_semantic_embeddings(
                entities=entities,
                vectorstore=embedding_store
            )


def get_pg_store():
    description_embedding_store = PgVectorStore(
        collection_name=f"entity_embeddings_workshop",
    )

    description_embedding_store.connect(
        host=os.getenv('POSTGRES_HOST'),
        password=os.getenv('POSTGRES_PASSWORD'),
        dbname=os.getenv('POSTGRES_DB'),
        user=os.getenv('POSTGRES_USER'),
        port=os.getenv('POSTGRES_PORT', '5432'),
    )

    return description_embedding_store


def get_entities(index_report):
    input_dir = f"/tmp/workshop/output/{index_report}/artifacts"
    entity_df = pd.read_parquet(f"{input_dir}/create_final_nodes.parquet")
    entity_embedding_df = pd.read_parquet(f"{input_dir}/create_final_entities.parquet")
    return read_indexer_entities(final_nodes=entity_df, final_entities=entity_embedding_df, community_level=2)


def get_context_builder(index_report):
    input_dir = f"/tmp/workshop/output/{index_report}/artifacts"
    entity_df = pd.read_parquet(f"{input_dir}/create_final_nodes.parquet")
    entities = get_entities(index_report)

    description_embedding_store = get_pg_store()
    st.write(f"Connected Postgres")

    entity_df.head()

    relationship_df = pd.read_parquet(f"{input_dir}/create_final_relationships.parquet")
    relationships = read_indexer_relationships(relationship_df)

    relationship_df.head()

    report_df = pd.read_parquet(f"{input_dir}/create_final_community_reports.parquet")
    reports = read_indexer_reports(final_community_reports=report_df, final_nodes=entity_df, community_level=2)

    report_df.head()

    text_unit_df = pd.read_parquet(f"{input_dir}/create_final_text_units.parquet")
    text_units = read_indexer_text_units(text_unit_df)

    text_unit_df.head()

    embedding_model = os.environ["GRAPHRAG_EMBEDDING_MODEL"]

    text_embedder = OpenAIEmbedding(
        api_key=os.environ["OPENAI_API_KEY"],
        api_base=None,
        api_type=OpenaiApiType.OpenAI,
        model=embedding_model,
        deployment_name=embedding_model,
        max_retries=20,
    )

    return LocalSearchMixedContext(
        community_reports=reports,
        text_units=text_units,
        entities=entities,
        relationships=relationships,
        entity_text_embeddings=description_embedding_store,
        embedding_vectorstore_key=EntityVectorStoreKey.ID,
        text_embedder=text_embedder,
        token_encoder=tiktoken.get_encoding("cl100k_base"),
    )


async def search():
    st.markdown("### Step 4: Search")
    query = st.text_area(label="search",
                         label_visibility='hidden',
                         max_chars=1000,
                         placeholder="Enter your query here ...",
                         value="")
    if st.button('Search'):
        with st.chat_message("system", avatar="avatars/postgres.svg"):
            if not query:
                st.error("Please enter a query")
                return

            if not os.path.exists(f"/tmp/workshop/output"):
                st.error("Please build index first")
                return

            subdirectories = list_subdirectories(path=f"/tmp/workshop/output")
            if len(subdirectories) == 0:
                st.error("Your need to build index first.")
                return

            with st.spinner(f'Connecting Postgres ...'):

                context_builder: LocalSearchMixedContext = get_context_builder(subdirectories[0])

                search_engine = PgLocalSearch(
                    llm=llm,
                    context_builder=context_builder,
                    token_encoder=context_builder.token_encoder,
                    llm_params=llm_params,
                    context_builder_params=local_context_params,
                    response_type="multiple paragraphs",
                )

        await search_engine.asearch(query)


if __name__ == "__main__":
    try:
        title = "GraphRAG with Azure Database for PostgreSQL(pgvector) and OpenAI GPT-4o mini"
        st.set_page_config(page_title=title,
                           page_icon="avatars/favicon.ico",
                           layout="wide",
                           initial_sidebar_state='expanded')
        st.image("avatars/logo.svg", width=150)
        st.title(title)
        dataset()
        st.markdown("----------------------------")
        upload_file()
        st.markdown("----------------------------")
        build_index()
        st.markdown("----------------------------")
        asyncio.run(search())
    except Exception as e:
        logger.exception(e)
        raise e
