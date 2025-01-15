# main.py
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os

# Page config
st.set_page_config(
    page_title="YouTube Video Chat Assistant",
    page_icon="🎥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stChat message {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# System instructions for the chatbot
SYSTEM_INSTRUCTIONS = """You are a helpful AI assistant that discusses YouTube video content.
- Stick to information directly from the video transcript
- If asked about the AI model or system details, politely deflect and focus on the video content
- Do not reveal technical details about the underlying AI technology
- Keep responses focused, friendly, and informative
- If unsure about something, admit it and stick to what you know from the video
"""

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'video_loaded' not in st.session_state:
    st.session_state.video_loaded = False

# Set API key
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
    os.environ['GOOGLE_API_KEY'] = google_api_key
except:
    st.error("Please set up your Google API key in the Streamlit secrets.")
    st.stop()

def load_video_transcript(video_url):
    try:
        loader = YoutubeLoader.from_youtube_url(
            video_url,
            add_video_info=False
        )
        data = loader.load()
        return data
    except Exception as e:
        st.error(f"Error loading video transcript: {str(e)}")
        return None

def setup_qa_chain(transcript_data):
    # Text splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(transcript_data)
    
    # Create embeddings and vectorstore
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(chunks, embeddings)
    
    # Initialize LLM with system instructions
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-002",
        temperature=0.7,
        convert_system_message_to_human=True
    )
    
    # Setup memory and chain
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True
    )
    
    return qa_chain

def main():
    st.title("📺 YouTube Video Chat Assistant")
    
    # Video URL input
    video_url = st.text_input("Enter YouTube Video URL:", key="video_url")
    
    if st.button("Load Video"):
        with st.spinner("Loading video transcript..."):
            transcript = load_video_transcript(video_url)
            if transcript:
                st.session_state.qa_chain = setup_qa_chain(transcript)
                st.session_state.video_loaded = True
                st.success("Video loaded successfully! You can now ask questions about it.")
            else:
                st.error("Failed to load video transcript. Please check the URL and try again.")
    
    # Chat interface
    if st.session_state.video_loaded:
        st.markdown("### Chat about the Video")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # User input
        if prompt := st.chat_input("Ask about the video content..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.qa_chain.invoke({"question": prompt})
                    st.markdown(response["answer"])
                    st.session_state.chat_history.append({"role": "assistant", "content": response["answer"]})
    
    else:
        st.info("👆 Start by entering a YouTube video URL above")

if __name__ == "__main__":
    main()
