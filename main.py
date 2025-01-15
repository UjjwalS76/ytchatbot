import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os

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
else:
    st.error('Google API key not found in secrets.')
    st.stop()

def load_video_transcript(video_url):
    """Load and return the transcript of a YouTube video."""
    try:
        loader = YoutubeLoader.from_youtube_url(
            video_url,
            add_video_info=False
        )
        data = loader.load()
        return data
    except Exception as e:
        st.error(f"Error loading transcript: {str(e)}")
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
            verbose=True
        )
        
        return qa_chain
    except Exception as e:
        st.error(f"Error setting up QA chain: {str(e)}")
        return None

def main():
    st.title("💬 YouTube Video Chat Assistant")
    
    # Sidebar with instructions
    with st.sidebar:
        st.markdown("""
        ### How to use:
        1. Enter a YouTube video URL
        2. Click 'Load Video' to process the transcript
        3. Ask questions about the video content
        """)
    
    # Video URL input
    video_url = st.text_input("Enter YouTube Video URL:", key="video_url")
    
    # Load video button
    if video_url and st.button("Load Video"):
        with st.spinner("Processing video transcript..."):
            transcript = load_video_transcript(video_url)
            if transcript:
                qa_chain = setup_qa_chain(transcript)
                if qa_chain:
                    st.session_state.qa_chain = qa_chain
                    st.session_state.video_loaded = True
                    st.success("Video processed successfully! You can now ask questions.")
                else:
                    st.error("Failed to setup the chat system. Please try again.")
    
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
    main()
