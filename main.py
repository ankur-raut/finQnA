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

st.set_page_config(layout="wide")

st.title('FinQA')

persist_directory = ""
model = ""
with st.sidebar:
    with st.form('Cohere/OpenAI'):
        mod = st.radio('Choose OpenAI/Cohere', ('OpenAI', 'Cohere'))
        api_key = st.text_input('Enter API key', type="password")
        model = st.radio('Choose Company', ('ArtisanAppetite foods', 'BMW','Titan Watches'))
        submitted = st.form_submit_button("Submit")


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
    if(mod=='Cohere'):
        os.environ["Cohere_API_KEY"] = api_key
        llm = Cohere(cohere_api_key=api_key)
        embeddings = CohereEmbeddings(cohere_api_key=api_key)

def ans(prompt,style,store):
    s=[]

    # First, define custom callback handler implementations
    class MyCustomHandlerOne(BaseCallbackHandler):
        def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
            print(f"on_agent_action {action}")
            s.append(action)

    # Instantiate the handlers
    handler1 = MyCustomHandlerOne()

    search = DuckDuckGoSearchRun()

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
        Tool(
            name='DuckDuckGo Search',
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

    def get_authors_tone_description(how_to_describe_tone, users_tweets,text):
        template = """
            You are an AI Bot that is very good at generating writing in a similar tone as examples.
            Be opinionated and have an active voice.
            Take a strong stance with your response.
            Do not mention the tone in your output.

            % HOW TO DESCRIBE TONE
            {how_to_describe_tone}

            % START OF EXAMPLES
            {tweet_examples}
            % END OF EXAMPLES

            %TEXT
            {text}

            %YOUR TASK
            Your goal is to write content with the tone that is described below and mimic the tone.
            """

        prompt = PromptTemplate(
            input_variables=["how_to_describe_tone", "tweet_examples","text"],
            template=template,
        )

        final_prompt = prompt.format(how_to_describe_tone=how_to_describe_tone, tweet_examples=users_tweets,text=text)
        
        # print(final_prompt)
        authors_tone_description = llm.predict(final_prompt)

        return authors_tone_description
    
    
    example = style

    authors_tone_description = get_authors_tone_description(how_to_describe_tone, example,res)

    # template = """
    # % INSTRUCTIONS
    # - You are a finance data analyst AI Bot that is very good at mimicking an author writing style.
    # - You have to answer the question in detail.
    # - Your goal is to write content with the tone that is described below.
    # - Do not go outside the tone instructions below
    

    # % Description of the authors tone:
    # {authors_tone_description}

    # % Authors writing samples
    # {example}

    # % YOUR TASK
    # {question}
    # """

    # prompt = PromptTemplate(
    #     input_variables=["authors_tone_description", "example","question"],
    #     template=template,
    # )

    # final_prompt = prompt.format(authors_tone_description=authors_tone_description, example=example,question=prompt)

    # res = agent_chain.run(final_prompt, callbacks=[handler1])


    # prompt_style = f"""
    # You are an editor.
    # You have to convert the provided text into a style that is similar to the style mentioned.
    # The style does not mention any instructions rather it is an example of how the answer is expected to be.
    # You have to copy similar format exactly as the style is in.

    # Do not copy the text in style strictly in any case.

    # text = {res}
    # style = {style}
    # """

    # response = llm(prompt_style)
        # agent_action = s[0]
    tool_component = []
    for i in s:
        tool_component.append(i.tool)
        print(i.tool)
    tool_component = ' , '.join(map(str, tool_component))
    authors_tone_description = re.sub(r'Tone:.+', '', authors_tone_description).strip()
    return [authors_tone_description,tool_component]

# prompt = """
# what are some of the key risks and potential ways to mitigate in ArtisanAppetite foods?
# """

# style = """
# To fulfill growing demand, X company must be able to depend on the quality and stability of suppliers who may face limited resources, regulation restraints, and capacity controls. Currently, X company pays its suppliers an average of 23% above market prices to procure the high-quality Arabica coffee its brand relies upon. Given the limited supply of beans that meet its standards as well as price volatility discussed in the commodity prices section below, the company faces potential shortages, and skyrocketing prices of its core product should its criteria not be met in a given harvest.

# Additionally, X company relationships with suppliers – especially those in the developing world – have long been a target issue for activists of human rights, environmental and labor issues. Activists began pressuring the company to offer fair trade coffee in 2000, and the company faced a major reputational blow in 2006 after campaigns by Oxfam against X company dealings with the Ethiopian government as well as the documentary Black Gold. In an industry with active competition on sustainability issues, the health of the X company brand is reliant on the company’s ability to source ethically traded inputs.
# """

# prompt = st.text_input('Enter Question')
# style = st.text_input('Enter style')


# Create a single line layout
col1, col2, col3 = st.columns([1, 2, 1])

# Components in the first column
with col1:
    # prompt = st.text_area('Enter Question', key=1)
    prompt =f'what is {model} background / Profile ?'
    st.write(prompt)
    # prompt = prompt

# Components in the second column
with col2:
    style = """It is an artificial intelligence research laboratory. 
The company created the ChatGPT Generative AI program, 
which operates on a large language model and takes in a text prompt to generate a human-like response. 
OpenAI systems are powered by Microsoft's Azure-based supercomputing platform.
    """
    style = st.text_area("Enter style", value=style, key=1)
    # st.write(style)

    # style = st.text_area('Enter style', key=1)
    # style = style

# Components in the third column
with col3:
    submitted = st.button("Submit", key=11)

if submitted:
    st.title('Answer:')
    # st.write("1 pressed")
    # st.write(template_string)
    store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    ans = ans(prompt,style,store)
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
    # prompt = prompt2

# Components in the second column
with col2:
    style2 = """Three sources briefed on OpenAI's recent pitch to investors said the organization 
expects $200 million in revenue next year and $1 billion by 2024.

The forecast, first reported by Reuters, represents how some in Silicon Valley 
are betting the underlying technology will go far beyond splashy and sometimes flawed public demos.

OpenAI was most recently valued at $20 billion in a secondary share sale, one of the sources said. 
The startup has already inspired rivals and companies building applications atop its generative 
AI software, which includes the image maker DALL-E 2. OpenAI charges developers licensing its technology 
about a penny or a little more to generate 20,000 words of text, and about 2 cents to create an image 
from a written prompt, according to its website.
    """
    style2 = st.text_area("Enter style", value=style2, key=2)
    # style2 = st.text_area('Enter style', key=2)
    # style = style2

# Components in the third column
with col3:
    submitted2 = st.button("Submit", key=21)

if submitted2:
    st.title('Answer:')
    # st.write("1 pressed")
    # st.write(template_string)
    store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    ans = ans(prompt2,style2,store)
    # st.write(ans[0])
    # st.write(ans[2])
    ans2 = st.text_area("Answer in Style Mentioned:", value=ans[0])
    st.write("Tool used:")
    st.write(ans[1])

# Create a single line layoutt
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    # prompt3 = st.text_area('Enter Question', key=3)
    prompt3=f'what are some of the key risks and potential ways to mitigate in {model}?'
    st.write(prompt3)
    # prompt = prompt3

# Components in the second column
with col2:
    style3 = """They suggest several ways to mitigate these risks. 
    It includes the need for an international authority that can inspect systems, require audits, 
    test for compliance with safety standards, place restrictions on deployment degrees and security levels, etc.
    """
    style3 = st.text_area("Enter style", value=style3, key=3)
    # style3 = st.text_area('Enter style', key=3)
    # style = style3

# Components in the third column
with col3:
    submitted3 = st.button("Submit", key=31)

if submitted3:
    st.title('Answer:')
    # st.write(template_string)
    store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    ans = ans(prompt3,style3,store)
    # st.write(ans[0])
    # st.write(ans[2])
    ans3 = st.text_area("Answer in Style Mentioned:", value=ans[0])
    st.write("Tool used:")
    st.write(ans[1])

# Create a single line layout
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    # prompt4 = st.text_area('Enter Question', key=4)
    prompt4=f'provide an analysis of the the indutry and the competition of {model}?'
    st.write(prompt4)
    # prompt = prompt4

# Components in the second column
with col2:
    style4 = """OpenAI's top competitors include Anthropic, Cohere, and Aleph Alpha. 
    Anthropic provides artificial intelligence (AI) 
    safety and research services specializing in developing general AI systems and language models.
    """
    style4 = st.text_area("Enter style", value=style4, key=4)
    # style4 = st.text_area('Enter style', key=4)
    # style = style4

# Components in the third column
with col3:
    submitted4 = st.button("Submit", key=41)


if submitted4:
    st.title('Answer:')
    # # st.write("4 pressed")
    # template_string = f"""
    # question = {prompt4}
    # You are a finance data analyst. You have to answer the above question in detail.
    # The answer should be in points explained properly.
    # You have to strictly follow the style of answering the question as follows and generate similar answers:
    # style = {style4}

    # Do not copy the content of the style in your answer.
    # Only study the style and follow the grammer accordingly
    # """
    # st.write(template_string)
    store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    ans = ans(prompt4,style4,store)
    # st.write(ans[0])
    # st.write(ans[2])
    ans4 = st.text_area("Answer in Style Mentioned:", value=ans[0])
    st.write("Tool used:")
    st.write(ans[1])



# uploaded_file = st.file_uploader(f"Upload image:", type=['pdf'])
# if uploaded_file is not None: 
#     loader = PyPDFLoader('ArtisanAppetite foods.pdf')
#     # Split pages from pdf
#     pages = loader.load_and_split()
#     # Load documents into vector database aka ChromaDB
#     store = Chroma.from_documents(pages, embeddings, collection_name='annualreport')


# Display submitted text and styled text
# if submitted:
#     st.write("Submitted text:", prompt)
# # if (st.button("Submit")):
#     persist_directory = 'ArtisanAppetite'
#     store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
#     st.write(ans(template_string,store))