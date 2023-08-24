import streamlit as st

def write_to_file(text):
    with open("output.txt", "w") as f:
        f.write(text)

def read_file_content():
    try:
        with open("output.txt", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "File not created yet."

def main():
    st.title("Text File Editor")

    global editable_text
    show_editable_text = st.button("Show Editable Text")

    if show_editable_text:
        editable_text = read_file_content()
    
    if "editable_text" not in globals():
        globals()["editable_text"] = ""

    if show_editable_text:
        edited_text = st.text_area("Edit the text below:", editable_text)

        # Update button
        if st.button("Update File"):
            write_to_file(edited_text)
            st.success("Text updated in the file!")

if __name__ == "__main__":
    main()
