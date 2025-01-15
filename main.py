import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from youtube_transcript_api import YouTubeTranscriptApi
import os
import re

def extract_video_id(youtube_url):
    """Extract video ID from YouTube URL."""
    patterns = [
        r'(?:v=|\/videos\/|embed\/|youtu.be\/|\/v\/|\/e\/|watch\?v=|youtube.com\/watch\?v=)([^#\&\?\n]*)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            return match.group(1)
    return None

def get_transcript(video_id):
    """Get transcript directly using youtube_transcript_api."""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([t['text'] for t in transcript_list])
    except Exception as e:
        st.error(f"Error getting transcript: {str(e)}")
        return None

def load_video_transcript(video_url):
    """Load and return the transcript of a YouTube video."""
    try:
        # Extract video ID
        video_id = extract_video_id(video_url)
        if not video_id:
            st.error("Could not extract video ID from URL")
            return None
        
        # Get transcript
        transcript_text = get_transcript(video_id)
        if not transcript_text:
            st.error("Could not get transcript")
            return None
        
        # Create a Document object with the transcript
        from langchain.schema.document import Document
        doc = Document(
            page_content=transcript_text,
            metadata={"source": video_id}
        )
        
        return [doc]
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def setup_qa_chain(transcript_data):
    """Set up the question-answering chain with the video transcript."""
    try:
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(transcript_data)
        
        # Create embeddings and vectorstore
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = Chroma.from_documents(chunks, embeddings)
        
        # Initialize LLM
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
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory,
            return_source_documents=False,
        )
        
        return qa_chain
        
    except Exception as e:
        st.error(f"Error in setup_qa_chain: {str(e)}")
        return None

def main():
    st.title("💬 YouTube Video Chat Assistant")
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'video_loaded' not in st.session_state:
        st.session_state.video_loaded = False
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None

    # Sidebar
    with st.sidebar:
        st.markdown("""
        ### How to use:
        1. Enter a YouTube video URL
        2. Click 'Load Video' to process the transcript
        3. Ask questions about the video content
        """)
        
        # Show API key status
        if 'GOOGLE_API_KEY' in st.secrets:
            st.success("API Key: Configured ✓")
        else:
            st.error("API Key: Missing ✗")
            
    # Video URL input
    video_url = st.text_input("Enter YouTube Video URL:", key="video_url")
    
    # Load video button
    if video_url:
        col1, col2 = st.columns([1, 6])
        with col1:
            process_video = st.button("Load Video", type="primary")
        
        if process_video:
            with st.spinner("Processing video transcript..."):
                # Reset states
                st.session_state.video_loaded = False
                st.session_state.chat_history = []
                
                # Load transcript
                transcript = load_video_transcript(video_url)
                if transcript:
                    qa_chain = setup_qa_chain(transcript)
                    
                    if qa_chain:
                        st.session_state.qa_chain = qa_chain
                        st.session_state.video_loaded = True
                        st.success("✅ Video processed successfully!")
    
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
                        st.error(f"Error generating response: {str(e)}")
    
    elif not video_url:
        st.info("👆 Start by entering a YouTube video URL above")

if __name__ == "__main__":
    if 'GOOGLE_API_KEY' in st.secrets:
        os.environ['GOOGLE_API_KEY'] = st.secrets['GOOGLE_API_KEY']
        main()
    else:
        st.error('Please set GOOGLE_API_KEY in your Streamlit secrets.')
        st.stop()
