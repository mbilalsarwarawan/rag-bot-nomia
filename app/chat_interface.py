import streamlit as st
from api_utils import get_api_response

def display_chat_interface():
    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Query:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Generating response..."):
            selected_file_id=st.session_state.get("selected_document_id", None)
            if selected_file_id:
                response = get_api_response(prompt, st.session_state.session_id, st.session_state.model,
                                            st.session_state.organization_id, st.session_state.workspace_id, selected_file_id)
            else:
                response = get_api_response(prompt, st.session_state.session_id, st.session_state.model,
                                            st.session_state.organization_id, st.session_state.workspace_id)

            if response:
                st.session_state.session_id = response.get('session_id')
                st.session_state.messages.append({"role": "assistant", "content": response['answer']})

                with st.chat_message("assistant"):
                    st.markdown(response['answer'])

                    with st.expander("Details"):
                        st.subheader("Generated Answer")
                        st.code(response['answer'])
                        st.subheader("Model Used")
                        st.code(response['model'])
                        st.subheader("Session ID")
                        st.code(response['session_id'])
            else:
                st.error("Failed to get a response from the API. Please try again.")
