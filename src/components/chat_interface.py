import streamlit as st

def render_chat_interface():
    """This function is no longer used in the main app since we've integrated the input directly there"""
    cols = st.columns([4, 1])
    with cols[0]:
        user_input = st.text_input("", placeholder="Type your message here...")
    with cols[1]:
        send_button = st.button("Send")
    
    if send_button and user_input:
        return user_input
    return None