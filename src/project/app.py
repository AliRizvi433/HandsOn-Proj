import streamlit as st
from project.rag import ask_question

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

if st.button("Search"):
    if question.strip():
        with st.spinner("Searching for relevant papers and generating answer..."):
            answer = ask_question(
                question,
                min_year=int(year)
                )
        st.subheader("Answer:")
        st.markdown(answer)
    else:
        st.warning("Please enter a question.")