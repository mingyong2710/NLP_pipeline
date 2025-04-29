import streamlit as st

def add_message(message):
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.chat_history.append(message)

def get_history():
    return st.session_state.get('chat_history', [])