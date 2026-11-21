import csv

import streamlit as st
from project.rag import ask_question
from project.loader import load_documents

st.set_page_config(
    page_title="Academic Research Assistant",
    page_icon=":books:",
)

st.title("Academic Research Assistant")
st.write("Ask questions about recent academic papers and get answers based on the provided context.")

question = st.text_input("Enter your question:")

year = st.number_input(
    "Minimum publication year:",
    min_value=2000,
    max_value=2026,
    value=2023
)

try:
    indexed_document_count = len(load_documents())
except (OSError, csv.Error, ValueError, KeyError) as exc:
    indexed_document_count = 1
    st.error(f"Unable to load the paper dataset: {exc}")

k = st.number_input(
    "Number of research papers to retrieve",
    min_value=1,
    max_value=max(1, indexed_document_count),
    value=min(5, indexed_document_count),
    step=1,
)

if st.button("Search"):
    if not question.strip():
        st.warning("Please enter a question.")
    elif k < 1 or k > indexed_document_count:
        st.warning("Please select a valid number of research papers.")
    else:
        try:
            with st.spinner("Searching for relevant papers and generating answer..."):
                result = ask_question(
                    question,
                    min_year=int(year),
                    k=int(k),
                )

            st.subheader("Answer:")
            st.markdown(result.answer)

            st.subheader("Retrieved Research Papers")
            st.write(f"Total number of papers returned: {result.returned_count}")

            if result.limited_by_available_matches:
                st.info(
                    f"Only {result.returned_count} matching papers are available "
                    f"for the selected filters."
                )

            for number, paper in enumerate(result.retrieved_papers, start=1):
                metadata = paper.document.metadata
                st.markdown(f"**{number}. {metadata.get('title', 'Unknown title')}**")
                st.write(f"Publication year: {paper.publication_year if paper.publication_year is not None else 'Unknown'}")
                st.write(f"Discipline: {metadata.get('discipline', 'Unknown')}")
                st.write(f"Paper ID: {metadata.get('paper_id', 'Unknown')}")
                st.write(
                    f"Relevance score: {paper.combined_score:.3f} "
                    "(combined semantic and recency ranking)"
                )
        except (ValueError, FileNotFoundError) as exc:
            st.error(str(exc))
        except Exception:
            st.error("The retrieval or answer-generation service failed. Please try again.")