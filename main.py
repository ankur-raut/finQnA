import streamlit as st
from langchain.llms import OpenAI, Cohere
import os
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings,CohereEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)
import re

from langchain.agents import Tool
from langchain.tools import DuckDuckGoSearchRun
# from langchain.utilities import WikipediaAPIWrapper
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from typing import Dict, Union, Any, List

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import tracing_enabled
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.utilities import SerpAPIWrapper

st.set_page_config(layout="wide")

st.title('FinQA')

persist_directory = ""
model = ""
with st.sidebar:
    with st.form('Cohere/OpenAI'):
        mod = st.radio('Choose OpenAI/Cohere', ('OpenAI', 'Cohere'))
        api_key = st.text_input('Enter API key', type="password")
        serpAI_key = st.text_input('Enter SERPAIAPI key', type="password")
        model = st.radio('Choose Company', ('ArtisanAppetite foods', 'BMW','Titan Watches'))
        submitted = st.form_submit_button("Submit")

st.markdown('<a href="/QA_map" target="_self">Go to Edit Style -></a>', unsafe_allow_html=True)

# Check if API key is provided and set up the language model accordingly
if api_key:
    if model == 'ArtisanAppetite foods':
        persist_directory = 'ArtisanAppetite'
    if model == 'BMW':
        persist_directory = 'bmw'
    if model == 'Titan Watches':
        persist_directory = 'titan'
    if(mod=='OpenAI'):
        os.environ["OPENAI_API_KEY"] = api_key
        llm = OpenAI(temperature=0.1, verbose=True)
        embeddings = OpenAIEmbeddings()
        os.environ["SERPAPI_API_KEY"] = serpAI_key
    if(mod=='Cohere'):
        os.environ["Cohere_API_KEY"] = api_key
        llm = Cohere(cohere_api_key=api_key)
        embeddings = CohereEmbeddings(cohere_api_key=api_key)
        os.environ["SERPAPI_API_KEY"] = serpAI_key

def ans(prompt,style,store):
    s=[]

    # First, define custom callback handler implementations
    class MyCustomHandlerOne(BaseCallbackHandler):
        def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
            print(f"on_agent_action {action}")
            s.append(action)

    # Instantiate the handlers
    handler1 = MyCustomHandlerOne()

    # search = DuckDuckGoSearchRun()
    search = SerpAPIWrapper()

    qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=store.as_retriever()
        )
    print("wrokedddd")

    tools = [
        Tool(
            name='Knowledge Base',
            func=qa.run,
            description=(
                'use this tool when answering all queries to get the answers except industry analysis'
                'more information about the topic'
            )
        ),
        # Tool(
        #     name='DuckDuckGo Search',
        #     func= search.run,
        #     description="Useful for when you need to do a search on the internet to find information that another tool can't find. be specific with your input."
        # )
        Tool(
            name='SerpAPI Search',
            func= search.run,
            description="Useful for when you need to do a search on the internet to find information that another tool can't find. be specific with your input."
        )
    ]

    agent_instructions = "Try 'Knowledge Base' tool first, Use the 'DuckDuckGo Search' tool only when you need industry analysis."


    agent_chain = initialize_agent(
        tools = tools,
        llm = llm,
        agent_instructions=agent_instructions,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors="Check your output and make sure it conforms!" #to fix the ouputParser error
    )

    templete = f"""
    You are a finance data analyst. You have to answer the question in detail.
    The answer should be in points explained properly.
    question = {prompt}
    """
    res = agent_chain.run(templete, callbacks=[handler1])

    how_to_describe_tone = """
    1. Tone: The overall attitude or feeling conveyed by the writing.
    2. Pace: The speed at which the events or ideas are presented.
    3. Mood: The emotional atmosphere or ambiance created by the writing.
    4. Voice: The distinctive style or personality of the author shining through the writing.
    5. Diction: The choice and use of words, phrases, and language in the writing.
    6. Style: The manner in which the writing is crafted, including sentence structure and literary devices.
    7. Tension: The level of suspense, anticipation, or conflict present in the writing.
    8. Humor: The presence of comedic elements or the ability to evoke laughter.
    9. Intensity: The degree of emotional or intellectual impact the writing has on the reader.
    10. Imagery: The use of vivid and descriptive language to create mental images.
    11. Rhythm: The pattern or flow of words and phrases, often contributing to the musicality of the writing.
    12. Complexity: The level of intricacy, depth, or sophistication in the ideas or narrative.
    13. Authenticity: The genuineness or credibility of the writing, reflecting the author's personal experiences or expertise.
    14. Irony: The use of words or situations to convey a meaning that is opposite to the literal interpretation.
    15. Sensory details: The inclusion of sensory information (sight, sound, taste, touch, smell) to enhance the reader's experience.
    """

    def get_authors_tone_description(how_to_describe_tone, users_tweets):
        print(users_tweets)
        template = """
            You are an AI Bot that is very good at generating writing in a similar tone as examples.
            Be opinionated and have an active voice.
            Take a strong stance with your response.

            % HOW TO DESCRIBE TONE
            {how_to_describe_tone}

            % START OF EXAMPLES
            {users_tweets}
            % END OF EXAMPLES

            List out the tone qualities of the examples above.
            """

        prompt = PromptTemplate(
            input_variables=["how_to_describe_tone", "users_tweets"],
            template=template,
        )

        final_prompt = prompt.format(how_to_describe_tone=how_to_describe_tone, users_tweets=users_tweets)

        authors_tone_description = llm.predict(final_prompt)

        return authors_tone_description
    
    
    # example = style

    authors_tone_description = get_authors_tone_description(how_to_describe_tone, style)
    print (authors_tone_description)

    print(res)
    def count_paragraphs(text):
        paragraphs = text.split('\n\n')  # Splitting based on double line breaks
        para = len(paragraphs)
        cnt=0
        for i in text:
            if i != ' ':
                cnt = cnt+1
        word = text.split(' ')
        word = len(word)
        line = text.split('.')
        line = len(line)
        return para,cnt, line-1,word
    
    para = count_paragraphs(style.strip())

    prm = f"""
    Please rewrite the given text in number of paragraphs while maintaining a character count as mentioned in the format

    format:
    {authors_tone_description}

    number of Paragraphs = {para[0]}
    number of Characters = {para[1]}
    number of Lines = {para[2]}
    number of words = {para[3]}

    text = {res}

    Do not include lines like 'The tone of the examples above is confident, authoritative, and optimistic.' in your output strictly

    Do not include ;Word Count:, Number of Lines:, Number of Paragraphs:' in your output strictly
    """
    print(prm)
    out = llm(prm)
    print(out)
    cleaned_text = re.sub(r'(Word Count: \d+|Number of Lines: \d+|Number of Paragraphs: \d+)\n', '', out)
    cleaned_text = re.sub(r'Paragraph \d+: ', '', cleaned_text)
    print("before no of")
    print(cleaned_text)
    cleaned_text = re.sub(r'Number of Paragraphs: \d+\n', '', cleaned_text)

    tool_component = []
    for i in s:
        tool_component.append(i.tool)
        print(i.tool)
    tool_component = ' , '.join(map(str, tool_component))
    cleaned_text = re.sub(r'Tone:.+', '', cleaned_text).strip()
    def remove_specific_sentences(paragraph, phrases_to_remove):
        sentences = paragraph.split('. ')
        filtered_sentences = [sentence for sentence in sentences if all(phrase not in sentence for phrase in phrases_to_remove)]
        modified_paragraph = '. '.join(filtered_sentences)
        return modified_paragraph

    phrases_to_remove = ["Word Count", "Number of Paragraphs"]
    modified_paragraph = remove_specific_sentences(cleaned_text, phrases_to_remove)
    return [modified_paragraph,tool_component]

def write_to_file(text,fil):
    with open(f"{fil}.txt", "w") as f:
        f.write(text)

def read_file_content(fil):
    try:
        with open(f"{fil}.txt", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "File not created yet."

col1, col2, col3 = st.columns([1, 2, 1])

# Components in the first column
with col1:
    # prompt = st.text_area('Enter Question', key=1)
    prompt =f'what is {model} background / Profile ?'
    st.write(prompt)
    # prompt = prompt

# Components in the third column
with col3:
    submitted = st.button("Submit", key=11)

if submitted:
    st.title('Answer:')
    # st.write("1 pressed")
    # st.write(template_string)
    current_content1 = read_file_content("style1")
    store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    ans = ans(prompt,current_content1,store)
    # st.write(ans[0])
    # st.write(ans[2])
    ans1 = st.text_area("Answer in Style Mentioned:", value=ans[0])
    st.write("Tool used:")
    st.write(ans[1])

# Create a single line layout
col1, col2, col3 = st.columns([1, 2, 1])
# Components in the first column
with col1:
    # prompt2 = st.text_area('Enter Question', key=2)
    prompt2=f'Provide a summary of the {model} financial position.'
    st.write(prompt2)


# Components in the third column
with col3:
    submitted2 = st.button("Submit", key=21)

if submitted2:
    st.title('Answer:')
    current_content2 = read_file_content("style2")
    store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    ans = ans(prompt2,current_content2,store)
    ans2 = st.text_area("Answer in Style Mentioned:", value=ans[0])
    st.write("Tool used:")
    st.write(ans[1])

# Create a single line layoutt
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    # prompt3 = st.text_area('Enter Question', key=3)
    prompt3=f'what are some of the key risks and potential ways to mitigate in {model}?'
    st.write(prompt3)

# Components in the third column
with col3:
    submitted3 = st.button("Submit", key=31)

if submitted3:
    st.title('Answer:')
    current_content3 = read_file_content("style3")
    store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    ans = ans(prompt3,current_content3,store)
    ans3 = st.text_area("Answer in Style Mentioned:", value=ans[0])
    st.write("Tool used:")
    st.write(ans[1])

# Create a single line layout
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    # prompt4 = st.text_area('Enter Question', key=4)
    prompt4=f'provide an analysis of the the indutry and the competition of {model}?'
    st.write(prompt4)

# Components in the third column
with col3:
    submitted4 = st.button("Submit", key=41)


if submitted4:
    st.title('Answer:')
    current_content4 = read_file_content("style4")
    store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    ans = ans(prompt4,current_content4,store)
    ans4 = st.text_area("Answer in Style Mentioned:", value=ans[0])
    # st.write(style4)
    st.write("Tool used:")
    st.write(ans[1])