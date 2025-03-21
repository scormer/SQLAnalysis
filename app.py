# RagSearch.py

import streamlit as st
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from chromadb.config import Settings

def flatten_results(results):
    """
    If the result for any key is a list containing a single list,
    unwrap it so that you get the inner list directly.
    """
    for key in results:
        if isinstance(results[key], list) and len(results[key]) == 1 and isinstance(results[key][0], list):
            results[key] = results[key][0]
    return results

# Set up the persistent Chroma client and collection
client = chromadb.PersistentClient(settings=Settings(persist_directory="./chroma"))
embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.get_collection(name="codeanalysis", embedding_function=embedding_function)

# --- Streamlit sidebar for filtering options ---
st.sidebar.header("Filter Options")
object_type_filter = st.sidebar.multiselect(
    "Select Object Type", 
    options=["All", "view", "StoredProcedure", "Function"], 
    default=["All"]
)
tables_filter_input = st.sidebar.text_input("Tables involved (comma delimited)")

# --- User configuration options ---
st.sidebar.header("Search Configuration")
max_objects = st.sidebar.number_input("Max number of objects to retrieve", min_value=1, max_value=1000, value=100, step=1)
confidence_threshold = st.sidebar.slider("Confidence Level", 0.0, 1.0, value=0.5, step=0.01)

# --- Main search input ---
query = st.text_input("Enter your search query:")

if st.button("Search"):
    # Build metadata filter for ObjectType if "All" is not selected
    metadata_filter = {}
    if "All" not in object_type_filter:
        # Map user-friendly values to the codes in your metadata field.
        type_mapping = {
            "view": "V",
            "StoredProcedure": "SP",
            "Function": "F"
        }
        mapped_types = [type_mapping[t] for t in object_type_filter if t in type_mapping]
    else:
        mapped_types = ["V", "SP", "F"]
    
    metadata_filter["ObjectType"] = {"$in": mapped_types}
    # Run similarity search in Chroma using the max_objects from user configuration
    # results = collection.query(query_texts=[query], n_results=max_objects, where=metadata_filter)
    results = collection.query(query_texts=[query], n_results=100, where=metadata_filter)
    
    # Flatten the results to remove an extra level of nesting if needed
    results = flatten_results(results)
    
    # Post-process the results for the Tables involved filter
    if tables_filter_input:
        table_list = [t.strip().lower() for t in tables_filter_input.split(",") if t.strip()]
        filtered_ids = []
        filtered_docs = []
        filtered_metadata = []
        filtered_distances = []
        for idx, meta in enumerate(results["metadatas"]):
            underlying_tables = meta.get("UnderlyingTables", "").lower()
            # Only include rows where every specified table is found in UnderlyingTables
            if all(table in underlying_tables for table in table_list):
                filtered_ids.append(results["ids"][idx])
                filtered_docs.append(results["documents"][idx])
                filtered_metadata.append(meta)
                filtered_distances.append(results["distances"][idx])
        results["ids"] = filtered_ids
        results["documents"] = filtered_docs
        results["metadatas"] = filtered_metadata
        results["distances"] = filtered_distances

    if results["distances"]:
        max_distance = max(results["distances"])
        min_distance = min(results["distances"])
        
        final_confidence = [1 - ((d-min_distance) / (max_distance-min_distance+0.01)) for d in results["distances"]]

        final_ids = []
        final_docs = []
        final_metadata = []
        # final_confidence = []

        for idx, c in enumerate(final_confidence):
            if c >= confidence_threshold:    
                final_ids.append(results["ids"][idx])
                final_docs.append(results["documents"][idx])
                final_metadata.append(results["metadatas"][idx])
    results["ids"] = final_ids
    results["documents"] = final_docs
    results["metadatas"] = final_metadata

    # --- Display the search results ---
    num_results = len(results["ids"])
    st.write(f"### Search Results: {num_results} documents retrieved")
    if num_results:
        for i in range(num_results):
            st.write(f"**ID:** {results['ids'][i]}")
            meta = results["metadatas"][i]
            st.write(f"**ObjectType:** {meta.get('ObjectType')}")
            st.write(f"**ObjectName:** {meta.get('ObjectName')}")
            st.write(f"**UnderlyingTables:** {meta.get('UnderlyingTables')}")
            st.write(f"**Description:** {meta.get('Description')}")
            st.write(f"**Confidence:** {final_confidence[i]:.2f}")
            
            st.write("---")
    else:
        st.write("No results found.")
