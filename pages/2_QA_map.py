import streamlit as st
import pandas as pd


def write_to_file(text,fil):
    with open(f"{fil}.txt", "w") as f:
        f.write(text)

def read_file_content(fil):
    try:
        with open(f"{fil}.txt", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "File not created yet."

# question 1
st.subheader("Question 1 style:")
current_content1 = read_file_content("style1")
st.text_area("Edit the text below:", current_content1, key="editable_text1")

# Update button
if st.button("Update style",key=1):
    edited_text = st.session_state.editable_text1
    write_to_file(edited_text,"style1")

# question 2
st.subheader("Question 2 style:")
current_content2 = read_file_content("style2")
st.text_area("Edit the text below:", current_content2, key="editable_text2")

# Update button
if st.button("Update style",key=2):
    edited_text = st.session_state.editable_text2
    write_to_file(edited_text,"style2")


# question 3
st.subheader("Question 3 style:")
current_content3 = read_file_content("style3")
st.text_area("Edit the text below:", current_content3, key="editable_text3")

# Update button
if st.button("Update style",key=3):
    edited_text = st.session_state.editable_text3
    write_to_file(edited_text,"style3")

# question 4
st.subheader("Question 4 style:")
current_content4 = read_file_content("style4")
st.text_area("Edit the text below:", current_content4, key="editable_text4")

# Update button
if st.button("Update style",key=4):
    edited_text = st.session_state.editable_text4
    write_to_file(edited_text,"style4")

