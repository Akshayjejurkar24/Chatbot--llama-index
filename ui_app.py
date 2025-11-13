from rag import retrieve_similar_documents, query_llm_with_context, system_prompt
import streamlit as st  # You must import Streamlit first

# Streamlit page config
st.set_page_config(page_title="Bajaj Chatbot", page_icon="🤖")
st.title(" Bajaj Financial Report Chatbot")

# Input box
user_query = st.text_input("Ask a question about Bajaj:") 

if st.button("Get Answer"):
    if user_query.strip() == "":
        st.warning("Please enter a question.")
    else:
        # Retrieve documents
        results = retrieve_similar_documents(user_query)
        
        # Query LLM
        answer = query_llm_with_context(user_query, results, system_prompt)
        
        # Display
        st.subheader("Answer")
        st.write(answer)
