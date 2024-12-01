from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from langserve import add_routes
import streamlit as st
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun,DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper,DuckDuckGoSearchAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain.agents import create_openai_tools_agent,initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler

# load_dotenv()
st.sidebar.title("Settings")
groq_api_key=st.sidebar.text_input("enter your groq api key",type='password')
from langchain_groq import ChatGroq


# os.environ['LANGCHAIN_API_KEY']=os.getenv('LANGCHAIN_API_KEY')
# os.environ['LANGCHAIN_TRACING_V2']="true"
# os.environ['LANCHAIN_PROJECT']=os.getenv("LANCHAIN_PROJECT")
# os.environ['HF_TOKEN']=os.getenv('HF_TOKEN')  


api_wrapper_wiki=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=250)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

api_wrapper_arxiv=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=250)
arxiv=ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

search=DuckDuckGoSearchRun(name="Search")


st.title("Langchain- chat with search")

if "messages" not in st.session_state:
    st.session_state['messages']=[
        {"role":"assistant",
         "content":"Hi Im a chatbot"}
    ]
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

if prompt:=st.chat_input(placeholder="what is machine learning?"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    llm=ChatGroq(groq_api_key=groq_api_key,model_name="Llama3-8b-8192",streaming=True)
    tools=[search,arxiv,wiki]
    search_agent=initialize_agent(tools=tools,llm=llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_errors=True)
    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response=search_agent.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({'role':'assistant',"content":response})
        st.write(response)
