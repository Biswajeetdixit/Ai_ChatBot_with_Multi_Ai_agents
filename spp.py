import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler


##Arxiv and wikipedia Tools
arxiv_wrapper=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=250)
wiki=WikipediaQueryRun(api_wrapper=wiki_wrapper)

search=DuckDuckGoSearchRun(name="Search")

st.title("🔎 LangChain - Chat with search")
"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""
### sidebar for setting
st.sidebar.title("setting")
api_key=st.sidebar.text_input("Plese enter your api_key :",type="password")


if "messages" not in st.session_state:
    st.session_state['messages']=[
        {"role":"assisstant","content":"Hi,I am a chat bot who can search the web.How can i help you ?"}

    ]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])



if prompt:=st.chat_input(placeholder="what is Generative AI ?"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)


    llm=ChatGroq(groq_api_key=api_key,model_name="llama-3.2-11b-text-preview",streaming=True)
    tools=[search,arxiv,wiki]

    search_agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_errors=True)



    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response=search_agent.run(st.session_state.messages[-1]['content'],callbacks=[st_cb])
        st.session_state.messages.append({'role':'assistant','content':response})
        st.write(response)


