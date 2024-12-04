import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
import scholarly  # Google Scholar API
from pytube import Search  # YouTube Search
import os
from dotenv import load_dotenv
from groq import Client, APIConnectionError, GroqError

# Load environment variables
load_dotenv()

# Attempt to get GROQ_API_KEY from .env file
groq_api_key = os.getenv("GROQ_API_KEY")

# Streamlit sidebar for manual API Key input as a fallback
st.sidebar.title("Settings")
api_key_input = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# Prioritize manual input if given
api_key = api_key_input if api_key_input else groq_api_key

# Debugging: Show if API key is set (for development purposes only)
if not api_key:
    st.warning("API Key not provided. Please enter it in the sidebar.")

# Initialize the Groq Client with error handling
try:
    client = Client(api_key=api_key)
except (APIConnectionError, GroqError) as e:
    st.error(f"Connection to Groq API failed: {e}")
    st.stop()

# Streamlit app title and description
st.title("🔎 LangChain - Enhanced Chat with Search")
st.write(
    """
    This chatbot integrates multiple sources including ArXiv, Wikipedia, DuckDuckGo, Google Scholar, 
    and YouTube. Enter a query to explore information from these resources.
    """
)

# Initialize ArXiv and Wikipedia wrappers
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

# DuckDuckGo search tool
search_tool = DuckDuckGoSearchRun(name="Search")

# Google Scholar Tool
def search_google_scholar(query):
    search_query = scholarly.search_pubs(query)
    results = []
    for result in search_query:
        results.append((result.bib['title'], result.bib.get('abstract', 'No abstract available')))
        if len(results) >= 3:  # Limiting to 3 results for simplicity
            break
    return results

# YouTube Search Tool
def search_youtube(query):
    yt_search = Search(query)
    results = []
    for video in yt_search.results[:3]:  # Limiting to 3 videos for simplicity
        title = video.title
        url = f"https://www.youtube.com/watch?v={video.video_id}"
        results.append((title, url))
    return results

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot that can search the web. How can I help you?"}
    ]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

# User input for queries
if user_input := st.chat_input(placeholder="Ask me anything, like 'What is machine learning?'"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Initialize the ChatGroq model with the provided API key
    llm = ChatGroq(groq_api_key=api_key, model_name="gemma2-9b-it", streaming=True)
    
    # Define available tools
    tools = [search_tool, arxiv_tool, wiki_tool]

    # Initialize agent with tools
    search_agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True
    )

    # Use Streamlit handler for live response streaming
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

        # Run agent and store/display response
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)

    # Additional tools for specific queries
    if "scholar" in user_input.lower():
        scholar_results = search_google_scholar(user_input)
        for title, abstract in scholar_results:
            st.write(f"**Title**: {title}")
            st.write(f"**Abstract**: {abstract}")

    if "youtube" in user_input.lower():
        youtube_results = search_youtube(user_input)
        for title, url in youtube_results:
            st.write(f"**Title**: {title}")
            st.write(f"[Watch on YouTube]({url})")
