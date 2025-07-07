import sys
from pathlib import Path

# Get parent directory (Thesis-Edvin)
sys.path.append(str(Path.cwd().parent))

from utils import *
from rag import *

from langchain.schema import Document
import re

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict
import hashlib
from enum import Enum

import os
import os

from langchain.vectorstores import FAISS, Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma as LangchainChroma
from langchain_community.vectorstores.faiss import FAISS as LangchainFAISS

from langchain_huggingface import HuggingFaceEmbeddings

# mypy.ini

from sklearn.metrics import accuracy_score, classification_report

import itertools
from pydantic import BaseModel, field_validator

from tqdm import tqdm

tqdm.pandas()

from pandarallel import pandarallel
import numpy as np

from rank_bm25 import BM25Okapi


class CWE_doc_density(Enum):
    LIGHT = 0
    MEDIUM = 1
    NORM = 2
    HEAVY = 3


class CWE_abstraction(Enum):
    PILLAR = 4
    CLASS = 3
    BASE = 2
    VARIANT = 1
    COMPOUND = 0


class Vectorstore(Enum):
    FAISS = 0
    CHROMADB = 1


def remove_subtitle(markdown_text, subtitle_names):
    # Pattern to match the subtitle and all content until the next subtitle or end
    ret = markdown_text
    for name in subtitle_names:
        pattern = r"## " + re.escape(name) + r"\b.*?(?=\n## |\Z)"
        ret = re.sub(pattern, "", ret, flags=re.DOTALL | re.IGNORECASE)

    # Remove with flags for dot matching newline and case sensitivity
    return ret

def create_ordered_cwe_dict(weakness, doc_density=CWE_doc_density.LIGHT, old=False):
    ret =  {
        "ID": weakness["ID"],
        "Description": weakness['Description'],
        **({"ExtendedDescription": weakness['ExtendedDescription']} if "ExtendedDescription" in weakness else {}),

    }
    
    if doc_density in [CWE_doc_density.NORM, CWE_doc_density.HEAVY] and "AlternateTerms" in weakness :
        ret["AlternateTerms"] = weakness["AlternateTerms"]
    
    if doc_density in [CWE_doc_density.HEAVY] :
        if "CommonConsequences" in weakness:
            ret["CommonConsequences"] = weakness["CommonConsequences"]
        if "AffectedResources" in weakness:
            ret["AffectedResources"] = weakness["AffectedResources"]
        if "ModesOfIntroduction" in weakness:
            ret["ModesOfIntroduction"] = weakness["ModesOfIntroduction"]
        if "BackgroundDetails" in weakness:
            ret["BackgroundDetails"] = weakness["BackgroundDetails"]
    

    if doc_density in [CWE_doc_density.NORM, CWE_doc_density.HEAVY]:
        if "Notes" in weakness:
            ret["Notes"] = weakness["Notes"]
        if "ObservedExamples" in weakness:
            ret["ObservedExamples"] = [ ex['Description'] for ex in weakness["ObservedExamples"]]        

    if "DemonstrativeExamples" in weakness:
        if doc_density in [CWE_doc_density.LIGHT, CWE_doc_density.NORM] :
            ret["DemonstrativeExample"] = weakness['DemonstrativeExamples'][0]
        else :
            ret["DemonstrativeExamples"] = weakness['DemonstrativeExamples']
    
    return ret

def create_ordered_cwe_document(weakness, doc_density=CWE_doc_density.LIGHT, old=False):
    # Core Metadata (Filterable Fields)
    metadata = {
        "CWE_ID": weakness["ID"],
        "Name": weakness["Name"],
        "Abstraction": CWE_abstraction[weakness["Abstraction"].upper()].value,
        "FunctionalAreas": weakness.get("FunctionalAreas", []),
        "AffectedResources": weakness.get("AffectedResources", []),
        "ExploitLikelihood": weakness.get("LikelihoodOfExploit"),
        "MappingUsage": weakness.get("MappingNotes", {}).get("Usage"),
        "Code_Languages": set(),  # Track all languages found in examples
    }

    if not old:
        metadata["Child_IDs"] = weakness["Children"]

    # Core content structure
    content_parts = [
        f"# CWE-{weakness['ID']}: {weakness['Name']}",
        f"\n## Description\n{weakness['Description']}",  # Core Definition
    ]
    if weakness.get("ExtendedDescription"):
        content_parts.append(
            f"\n## Extended Description\n{weakness.get('ExtendedDescription', '')}"
        )

    if doc_density in [CWE_doc_density.NORM, CWE_doc_density.HEAVY] and weakness.get(
        "AlternateTerms"
    ):
        content_parts.append("\n## Alternate Terms")
        content_parts.extend(
            f"\n- **{term['Term']}**: {term.get('Description', 'No description available')}"
            for term in weakness.get("AlternateTerms", [])
        )

    if doc_density in [CWE_doc_density.HEAVY]:
        if weakness.get("CommonConsequences"):
            content_parts.append("\n## Potential Consequences")
            for cons in weakness.get("CommonConsequences", []):
                content_parts.append(
                    f"\n- **Scope**: {', '.join(cons.get('Scope', []))}\n"
                    f"  **Impact**: {', '.join(cons.get('Impact', []))}\n"
                    f"  **Note**: {cons.get('Note', 'No additional notes')}"
                )
        # Then add remaining elements in logical order
        if weakness.get("AffectedResources"):
            content_parts.extend(
                [
                    f"\n## Affected Resources\n{', '.join(weakness.get('AffectedResources', []))}"
                ]
            )
        if weakness.get("ModesOfIntroduction"):
            content_parts.extend(
                [
                    f"\n## Modes of Introduction\n"
                    + "\n".join(
                        f"- {mode['Phase']}: {mode.get('Note','')}"
                        for mode in weakness.get("ModesOfIntroduction", [])
                    )
                ]
            )
        if weakness.get("BackgroundDetails"):
            bg_details = weakness["BackgroundDetails"]
            if isinstance(bg_details, list):
                bg_details = "\n".join(bg_details)
            content_parts.append(f"\n## Background\n{bg_details[:2000]}")

    if doc_density in [CWE_doc_density.NORM, CWE_doc_density.HEAVY]:
        if weakness.get("Notes"):
            notes = [
                (note.get("Type", "General"), note["Note"])
                for note in weakness["Notes"]
                if note.get("Type")
                in [
                    "Technical",
                    "Relationship",
                    "Theoretical",
                    "Terminology",
                    "Applicable Platform",
                ]
            ]
            if notes:
                content_parts.append("\n## Notes")
                content_parts.extend(f"\n- **{ty}**: {note}" for ty, note in notes)

        # Observed Examples
        if weakness.get("ObservedExamples"):
            content_parts.append("\n## Observed Cases")
            content_parts.extend(
                f"\n- {ex['Description']}"  # removed (Ref: {ex.get('Reference','')})
                for ex in weakness["ObservedExamples"]
            )

    if weakness.get("DemonstrativeExamples"):
        content_parts.append("\n## Demonstrative Scenario")

        def format_code_block(code, language):
            # Normalize language tag
            language = (language or "").lower().strip()
            language = re.sub(r" ", "_", language)

            # Remove ALL leading/trailing backticks and whitespace
            code = re.sub(r"^[\s`]*", "", code)
            code = re.sub(r"[\s`]*$", "", code)

            # Remove empty lines at start/end
            code = re.sub(r"^\n+", "", code)
            code = re.sub(r"\n+$", "", code)

            code = re.sub(r"`{3,}", "", code)
            return (
                f"```{language}\n{code}\n```"
                if language and language != "other"
                else f"```\n{code}\n```"
            )

        examples = (
            [weakness["DemonstrativeExamples"][0]]
            if doc_density in [CWE_doc_density.LIGHT]
            else weakness["DemonstrativeExamples"]
        )
        for i, example_group in enumerate(examples):
            example_content = []
            pre_code_text = []
            post_code_text = []
            vulnerable_code = ""
            language = ""

            # First pass: Collect all components in order
            for entry in example_group.get("Entries", []):
                if "IntroText" in entry:
                    pre_code_text.append(entry["IntroText"])
                elif "BodyText" in entry:
                    if vulnerable_code:
                        post_code_text.append(entry["BodyText"])
                    else:
                        pre_code_text.append(entry["BodyText"])
                elif entry.get("Nature") == "Bad" and "ExampleCode" in entry:
                    vulnerable_code = entry["ExampleCode"]
                    language = entry.get("Language", "")
                    metadata["Code_Languages"].add(language)

            # Build the example with proper flow
            if pre_code_text:
                example_content.append("\n### Scenario")  # + str(i + 1)
                example_content.extend(f"\n{text}" for text in pre_code_text)

            if vulnerable_code:
                example_content.append(
                    "\n### Vulnerable"
                    + (" " + language if language and language != "other" else "")
                    + " Code"
                )
                example_content.append(format_code_block(vulnerable_code, language))

            if post_code_text:
                example_content.append("\n### Analysis")
                example_content.extend(f"\n{text}" for text in post_code_text)

            content_parts.extend(example_content)

    # Convert language set to list in metadata
    metadata["Code_Languages"] = list(metadata["Code_Languages"])

    return Document(page_content="\n".join(content_parts), metadata=metadata)


def split_cwe_document(doc: Document, max_tokens: int = 800) -> List[Document]:
    """
    Splits documents with chunk reassembly capability through metadata.
    Prioritizes:
    1. Keeping examples/scenarios intact
    2. Preserving section boundaries
    3. Adding reassembly metadata
    """
    if count_tokens(doc.page_content) <= max_tokens:
        # Add chunk metadata even for unsplit documents
        doc.metadata.update(
            {
                "chunk_id": f"{doc.metadata['CWE_ID']}_0",
                "total_chunks": 1,
                "chunk_hash": hashlib.md5(doc.page_content.encode()).hexdigest(),
            }
        )
        return [doc]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens,
        chunk_overlap=100,
        length_function=count_tokens,
        separators=[
            "\n\nScenario ",  # Split between demonstrative examples
            "\n\n## ",  # Section headers
            "\n\n",  # Paragraph breaks
            "```\n",  # End of code blocks
            "\n",  # Line breaks
            " ",  # Last resort
        ],
        keep_separator=True,
    )

    chunks = splitter.split_text(doc.page_content)
    doc_hash = hashlib.md5(doc.page_content.encode()).hexdigest()

    return [
        Document(
            page_content=f"# CWE-{doc.metadata['CWE_ID']} (Part {i+1}/{len(chunks)})\n{chunk}",
            metadata={
                **doc.metadata,  # Preserve original metadata
                **{
                    "chunk_id": f"{doc.metadata['CWE_ID']}_{i}",
                    "total_chunks": len(chunks),
                    "chunk_sequence": i,
                    "parent_hash": doc_hash,
                    "chunk_hash": hashlib.md5(chunk.encode()).hexdigest(),
                    "is_code_chunk": "```" in chunk,
                },
            },
        )
        for i, chunk in enumerate(chunks)
    ]


def add_sequential_ids(documents):
    """Adds sequential numeric IDs instead of UUIDs"""
    for i, doc in enumerate(documents, start=1):
        if not hasattr(doc, "metadata"):
            doc.metadata = {}
        doc.metadata["id"] = f"doc_{i}"
    return documents


# Initialize embeddings (same as when saving)
def create_vectorstore(
    docs,
    save_path="tmp/faiss_index",
    collection_name=None,
    vec_store=Vectorstore.FAISS,
    device="cuda",
):
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

    if vec_store == Vectorstore.FAISS:
        # Check if vector store exists and load it
        if os.path.exists(f"{save_path}/index.faiss") and os.path.exists(
            f"{save_path}/index.pkl"
        ):
            print("Loading existing vector store...")
            vectorstore = FAISS.load_local(
                folder_path=save_path,
                embeddings=embeddings,
                allow_dangerous_deserialization=True,  # Required security flag
            )
        else:
            print("Creating new vector store...")
            # Assuming 'all_docs' is your list of documents
            vectorstore = FAISS.from_documents(docs, embeddings)
            vectorstore.save_local(save_path)
            print(f"Vector store saved to {save_path}")
    elif vec_store == Vectorstore.CHROMADB:
        processed_docs = []
        for doc in docs:
            # Create a copy of the original metadata
            new_metadata = doc.metadata.copy()

            # Only modify Child_IDs if it exists
            new_metadata["Child_IDs"] = ",".join(map(str, new_metadata["Child_IDs"]))

            # Create new Document with same content and updated metadata
            processed_docs.append(
                Document(
                    page_content=doc.page_content,
                    metadata=new_metadata,  # All original metadata preserved
                )
            )
        if os.path.exists(save_path):
            print("Loading existing ChromaDB vector store...")
            vectorstore = Chroma(
                persist_directory=save_path,
                embedding_function=embeddings,
                collection_name=collection_name,
            )
        else:
            print("Creating new ChromaDB vector store...")
            vectorstore = Chroma.from_documents(
                processed_docs,
                embedding=embeddings,
                persist_directory=save_path,
                collection_name=collection_name,
            )
            print(f"ChromaDB vector store saved to {save_path}")

    else:
        raise ValueError(f"Unknown vector store type: {vec_store}")

    return vectorstore


def fusion_retrieval(
    vectorstore, bm25, docs, query: str, k: int = 5, vs_filter = None, alpha: float = 0.5
) -> List[Document]:
    """
    Perform fusion retrieval combining keyword-based (BM25) and vector-based search.

    Args:
    vectorstore (VectorStore): The vectorstore containing the documents.
    bm25 (BM25Okapi): Pre-computed BM25 index.
    query (str): The query string.
    k (int): The number of documents to retrieve.
    alpha (float): The weight for vector search scores (1-alpha will be the weight for BM25 scores).

    Returns:
    List[Document]: The top k documents based on the combined scores.
    """

    epsilon = 1e-8

    # Step 2: Perform BM25 search
    bm25_scores = bm25.get_scores(query.split())

    # Step 3: Perform vector search
    vector_results = vectorstore.similarity_search_with_score(
        query, 
        k=len(docs),
        filter=vs_filter
    )

    # Step 4: Normalize scores
    vector_scores = np.array([score for _, score in vector_results])
    vector_scores = 1 - (vector_scores - np.min(vector_scores)) / (
        np.max(vector_scores) - np.min(vector_scores) + epsilon
    )
    vector_results=zip([doc for doc, _ in vector_results], vector_scores)

    bm25_scores = (bm25_scores - np.min(bm25_scores)) / (
        np.max(bm25_scores) - np.min(bm25_scores) + epsilon
    )

    def combine(vec, bm25):
        return alpha * vec + (1 - alpha) * bm25

    doc_bm25scores = zip(docs, bm25_scores)
    score_dict = {doc[0].metadata["id"]: doc for doc in doc_bm25scores}
    score_dict = {
        doc.metadata["id"]: (
            doc,
            combine(score, score_dict[doc.metadata["id"]][1]),
        )
        for doc, score in vector_results
    }
    sorted_doc_scores = list(score_dict.values())
    sorted_doc_scores.sort(key=lambda a : a[1], reverse=True)

    # Step 7: Return top k documents
    return [doc_score[0] for doc_score in sorted_doc_scores[:k]]


def fusion_rag_label(
    data_df, vectorstore, bm25_docs, match_col="gpt_vulnerability", k=1, alpha=0.5
):  # max for k is 20
    ret_df = data_df.copy()

    tokenized_docs = [doc.page_content.split() for doc in bm25_docs]
    bm25_retriever = BM25Okapi(tokenized_docs)

    # display(ret_df[ret_df['cve_primary_cwe'].isna()])
    def label(desc):
        try:
            ret = list(
                set(
                    [
                        x.metadata["CWE_ID"]
                        for x in fusion_retrieval(
                            vectorstore, bm25_retriever, bm25_docs, desc, k, alpha
                        )
                    ]
                )
            )
            # print(len(ret))
            return ret
        except Exception as e:
            print(f"General error processing message: {e}")
            return

    ret_df["rag_candidates"] = ret_df[match_col].progress_map(label)
    ret_df["rag_label"] = ret_df.apply(
        lambda row: (
            row["cve_primary_cwe"]
            if row["cve_primary_cwe"] in row["rag_candidates"]
            else row["rag_candidates"][0]
        ),
        axis=1,
    )

    return ret_df


def bare_rag_label(
    data_df, vectorstore, bm25_docs, match_col="gpt_vulnerability", k=1
):  # max for k is 20
    ret_df = data_df.copy()
    faiss_retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": k,
        }
    )
    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = k
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5],
        # retrievers=[faiss_retriever], weights=[1]
    )

    # display(ret_df[ret_df['cve_primary_cwe'].isna()])
    def label(desc):
        try:
            ret = list(
                set([x.metadata["CWE_ID"] for x in ensemble_retriever.invoke(desc)])
            )
            # print(len(ret))
            return ret
        except Exception as e:
            print(f"General error processing message: {e}")
            return

    ret_df["rag_candidates"] = ret_df[match_col].progress_map(label)
    ret_df["rag_label"] = ret_df.apply(
        lambda row: (
            row["cve_primary_cwe"]
            if row["cve_primary_cwe"] in row["rag_candidates"]
            else row["rag_candidates"][0]
        ),
        axis=1,
    )

    return ret_df


def tree_traversal_retrieval(
    query: str,
    vectorstore,
    hierarchy_cache,
    bm25_docs=None,
    k_list: List = [3, 10, 5, 2, 1],
    top_abstraction=CWE_abstraction.PILLAR,
    alpha = 0.5
) -> List[Document]:
    """Perform tree traversal retrieval."""
    k_dict = {
        "PILLAR": k_list[0],
        "CLASS": k_list[1],
        "BASE": k_list[2],
        "VARIANT": k_list[3],
        "COMPOUND": k_list[4],
    }

    def retrieve_level(level, parent_ids=None, depth=0, max_depth=5) -> List[Document]:
        if parent_ids:
            filter_condition = {"CWE_ID": {"$in": parent_ids}}
            level_bm25_docs = [
                doc for doc in bm25_docs if doc.metadata["CWE_ID"] in parent_ids
            ]
        else:
            level_bm25_docs = [
                doc for doc in bm25_docs if doc.metadata["Abstraction"] == level.value
            ]
            filter_condition = {"Abstraction": {"$eq": level.value}}

        level_query = query
        # print(f"\n[Level {level.name} | Depth {depth}] Using filter: {filter_condition}")
        """
        if(level == CWE_abstraction.PILLAR) :
            level_query = remove_subtitle(query, ["Demonstrative Scenario"])
        """
        '''
        bm25_retriever = BM25Retriever.from_documents(level_bm25_docs)
        bm25_retriever.k = k_dict[level.name]  # Set number of results
        vector_retriever = vectorstore.as_retriever(
            search_kwargs={"k": k_dict[level.name], "filter": filter_condition}
        )
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.5, 0.5],  # Adjust based on your needs
        )

        docs = ensemble_retriever.invoke(level_query)
        '''
        tokenized_docs = [doc.page_content.split() for doc in level_bm25_docs]
        bm25_retriever = BM25Okapi(tokenized_docs)
        docs = fusion_retrieval(
            vectorstore, 
            bm25_retriever, 
            level_bm25_docs, 
            level_query, 
            k_dict[level.name], 
            filter_condition,
            alpha
        )


        """
        docs = vectorstore.similarity_search(
            query=level_query,
            filter=filter_condition,
            k=k_dict[level.name],
        )
        """

        if not docs:
            # print("No docs found, stopping recursion.")
            return []
        if depth > max_depth:
            # print(f"Max depth {max_depth} reached, stopping recursion.")
            return docs

        child_ids = []
        for doc in docs:
            child_ids.extend(
                hierarchy_cache.get(doc.metadata["CWE_ID"], {}).get("children", [])
            )
        child_ids = list(set(child_ids))
        max_level = max([doc.metadata.get("Abstraction", 0) for doc in docs])

        """
        print(
            f"Max level in current docs: {max_level}, number of Child IDs: {len(child_ids)}"
        )
        """

        if not child_ids or max_level == 0:
            # print("No child IDs or max_level=0, stopping recursion.")
            return docs

        child_docs = retrieve_level(
            CWE_abstraction(max_level), parent_ids=child_ids, depth=depth + 1
        )
        return docs + child_docs

    ret = retrieve_level(top_abstraction)

    ret = list({doc.metadata["id"]: doc for doc in ret}.values())

    # print(f"Number of documents retrieved: {len(ret)}")
    return ret


def raptor_rag_label(
    data_df,
    vectorstore,
    match_col="gpt_vulnerability",
    k_list: List = [3, 10, 5, 2, 0],
    bm25_docs=None,
    # max_workers=10,
    top_abstraction=CWE_abstraction.PILLAR,
    alpha = 0.5,
):  # max for k is 20
    ret_df = data_df.copy()

    hierarchy_cache = {}
    all_metadata = vectorstore.get()["metadatas"]
    for meta in all_metadata:
        hierarchy_cache[meta["CWE_ID"]] = {
            "abstraction": meta["Abstraction"],
            "children": meta["Child_IDs"].split(",") if meta.get("Child_IDs") else [],
        }
    # display(ret_df[ret_df['cve_primary_cwe'].isna()])

    def label(desc):
        try:
            ret = [
                x.metadata["CWE_ID"]
                for x in tree_traversal_retrieval(
                    desc,
                    vectorstore,
                    bm25_docs=bm25_docs,
                    k_list=k_list,
                    hierarchy_cache=hierarchy_cache,
                    top_abstraction=top_abstraction,
                    alpha=alpha
                )
            ]
            # print(len(ret))
            return ret
        except Exception as e:
            print(f"General error processing message: {e}")
            return

    ret_df["rag_candidates"] = ret_df[match_col].progress_map(label)
    ret_df["rag_label"] = ret_df.apply(
        lambda row: (
            row["cve_primary_cwe"]
            if row["cve_primary_cwe"] in row["rag_candidates"]
            else row["rag_candidates"][0]
        ),
        axis=1,
    )

    return ret_df


def evaluate_rag(df, col):
    y_test = df.cve_primary_cwe
    y_pred = df[col]

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    #print(len(df[df.cve_primary_cwe == df[col]]) / len(df))

    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)


from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def raptor_rag_label_optimized(
    data_df,
    vectorstore,
    match_col="gpt_vulnerability",
    k_list=[3, 10, 5, 2, 0],
    device="cpu",
    max_workers=8,
):
    # 1. Pre-fetch hierarchy relationships
    hierarchy_cache = {}
    all_metadata = vectorstore.get()["metadatas"]
    for meta in all_metadata:
        hierarchy_cache[meta["CWE_ID"]] = {
            "abstraction": meta["Abstraction"],
            "children": meta["Child_IDs"].split(",") if meta.get("Child_IDs") else [],
        }

    # 2. Thread-safe retrieval function
    def retrieve_single(query):
        try:
            # Modified tree traversal using cache
            results = []
            current_level = CWE_abstraction.PILLAR
            parent_ids = None

            for _ in range(10):  # Max depth
                if parent_ids:
                    filter_ = {"CWE_ID": {"$in": parent_ids}}
                else:
                    filter_ = {"Abstraction": {"$eq": current_level.value}}

                level_results = vectorstore.similarity_search(
                    query=query, k=k_list[current_level.value], filter=filter_
                )
                results.extend(level_results)

                # Get next level children
                parent_ids = []
                for doc in level_results:
                    parent_ids.extend(
                        hierarchy_cache.get(doc.metadata["CWE_ID"], {}).get(
                            "children", []
                        )
                    )

                if not parent_ids:
                    break

                current_level = CWE_abstraction(
                    max(doc.metadata["Abstraction"] for doc in level_results)
                )

            return [doc.metadata["CWE_ID"] for doc in results]
        except Exception as e:
            print(f"Error processing query: {e}")
            return []

    # 3. Parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        queries = data_df[match_col].tolist()
        results = list(
            tqdm(
                executor.map(retrieve_single, queries),
                total=len(queries),
                desc="Processing RAG queries",
            )
        )

    # 4. Create final dataframe
    ret_df = data_df.copy()
    ret_df["rag_candidates"] = results
    ret_df["rag_label"] = ret_df.apply(
        lambda row: (
            row["cve_primary_cwe"]
            if row["cve_primary_cwe"] in row["rag_candidates"]
            else (row["rag_candidates"][0] if row["rag_candidates"] else None)
        ),
        axis=1,
    )
    return ret_df
