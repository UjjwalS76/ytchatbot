import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="YouTube Video Chat Assistant",
    page_icon="🎥",
    layout="wide"
)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'video_loaded' not in st.session_state:
    st.session_state.video_loaded = False
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

# Set API key from Streamlit secrets
if 'GOOGLE_API_KEY' in st.secrets:
    os.environ['GOOGLE_API_KEY'] = st.secrets['GOOGLE_API_KEY']
    logger.info("API key loaded successfully")
else:
    st.error('Google API key not found in secrets.')
    st.stop()

def validate_youtube_url(url):
    """Validate if the URL is a proper YouTube URL."""
    if not url:
        return False
    if 'youtube.com/watch?v=' not in url and 'youtu.be/' not in url:
        return False
    return True

def load_video_transcript(video_url):
    """Load and return the transcript of a YouTube video."""
    try:
        logger.info(f"Attempting to load transcript for URL: {video_url}")
        
        if not validate_youtube_url(video_url):
            st.error("Please enter a valid YouTube URL")
            return None
            
        # Explicitly allow auto-generated captions and load all captions
        loader = YoutubeLoader.from_youtube_url(
            video_url,
            add_video_info=False,
            include_auto_generated=True,         # <--- Added
            load_all_available_captions=True     # <--- Added
        )
        data = loader.load()
        
        if not data:
            st.error("No transcript found for this video. Make sure the video has closed captions or auto-generated captions.")
            return None
            
        logger.info(f"Successfully loaded transcript with {len(data)} segments")
        return data
        
    except Exception as e:
        logger.error(f"Error loading transcript: {str(e)}")
        st.error(f"Failed to load video transcript: {str(e)}")
        return None

def setup_qa_chain(transcript_data):
    """Set up the question-answering chain with the video transcript."""
    try:
        logger.info("Setting up QA chain...")
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(transcript_data)
        logger.info(f"Split transcript into {len(chunks)} chunks")
        
        # Create embeddings and vectorstore
        logger.info("Creating embeddings...")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        logger.info("Creating vector store...")
        vectorstore = Chroma.from_documents(chunks, embeddings)
        
        # Initialize LLM
        logger.info("Initializing LLM...")
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.7,
        )
        
        # Setup memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create chain
        logger.info("Creating conversation chain...")
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory,
            return_source_documents=False,
            verbose=True
        )
        
        logger.info("QA chain setup complete")
        return qa_chain
        
    except Exception as e:
        logger.error(f"Error in setup_qa_chain: {str(e)}")
        st.error(f"Error setting up the chat system: {str(e)}")
        return None

def main():
    st.title("💬 YouTube Video Chat Assistant")
    
    # Display current status
    if st.session_state.video_loaded:
        st.sidebar.success("Video loaded and ready for chat")
    else:
        st.sidebar.info("No video loaded")
    
    # Sidebar with instructions
    with st.sidebar:
        st.markdown("""
        ### How to use:
        1. Enter a YouTube video URL
        2. Click 'Load Video' to process the transcript
        3. Ask questions about the video content
        
        ### Note:
        - Video must have closed captions available, or use auto-generated captions
        - Only processes English transcripts currently
        """)
    
    # Video URL input
    video_url = st.text_input(
        "Enter YouTube Video URL:",
        key="video_url",
        help="Paste the full YouTube video URL here"
    )
    
    # Load video button
    if video_url:
        if st.button("Load Video", key="load_btn"):
            st.session_state.video_loaded = False  # Reset state
            st.session_state.chat_history = []     # Clear chat history
            
            with st.spinner("Processing video transcript..."):
                logger.info("Starting video processing...")
                
                transcript = load_video_transcript(video_url)
                if transcript:
                    logger.info("Transcript loaded, setting up QA chain...")
                    qa_chain = setup_qa_chain(transcript)
                    
                    if qa_chain:
                        st.session_state.qa_chain = qa_chain
                        st.session_state.video_loaded = True
                        st.success("✅ Video processed successfully! You can now ask questions.")
                        logger.info("Video processing complete")
                    else:
                        st.error("❌ Failed to setup the chat system. Please try again.")
                        logger.error("Failed to create QA chain")
    
    # Chat interface
    if st.session_state.video_loaded and st.session_state.qa_chain:
        st.markdown("### Chat")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about the video..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.write(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.qa_chain({"question": prompt})
                        st.write(response['answer'])
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": response['answer']}
                        )
                    except Exception as e:
                        logger.error(f"Error generating response: {str(e)}")
                        st.error(f"Failed to generate response: {str(e)}")
    
    elif not video_url:
        st.info("👆 Start by entering a YouTube video URL above")

if __name__ == "__main__":
    main()
