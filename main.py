import streamlit as st
import os
from typing import Optional, List
import re
from youtube_transcript_api import YouTubeTranscriptApi

# LangChain imports
from langchain_community.chat_models import ChatPerplexity
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document

# Streamlit page config
st.set_page_config(
    page_title="YouTube Video Chat Assistant",
    page_icon="🎥",
    layout="wide"
)

# Custom CSS to improve UI
st.markdown("""
<style>
    .stChat {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Perplexity API key
PERPLEXITY_API_KEY = st.secrets.get("PERPLEXITY_API_KEY", os.getenv("PERPLEXITY_API_KEY"))
if not PERPLEXITY_API_KEY:
    st.error("⚠️ Perplexity API key not found!")
    st.stop()
os.environ["PERPLEXITY_API_KEY"] = PERPLEXITY_API_KEY

class YouTubeChatbot:
    """Main class for YouTube video chatbot functionality."""
    
    def __init__(self):
        """Initialize the chatbot with necessary components."""
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.llm = ChatPerplexity(
            model="llama-3.1-sonar-small-128k-online",
            temperature=0.7
        )
        
    @staticmethod
    def extract_video_id(url: str) -> Optional[str]:
        """Extract video ID from various YouTube URL formats."""
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'(?:embed\/)([0-9A-Za-z_-]{11})',
            r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'
        ]
        
        for pattern in patterns:
            if match := re.search(pattern, url):
                return match.group(1)
        return None

    def load_transcript(self, video_url: str) -> Optional[List[Document]]:
        """Load and process YouTube video transcript."""
        try:
            video_id = self.extract_video_id(video_url)
            if not video_id:
                st.error("❌ Invalid YouTube URL")
                return None

            # Get available transcripts
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Try different methods to get English transcript
            transcript = None
            for method in ['find_generated_transcript', 'find_manually_created_transcript']:
                try:
                    transcript = getattr(transcript_list, method)(['en'])
                    break
                except:
                    continue
            
            if not transcript:
                # Try getting any transcript and translating to English
                try:
                    transcript = transcript_list.find_generated_transcript([])
                    transcript = transcript.translate('en')
                except Exception as e:
                    st.error(f"❌ No suitable transcript found: {str(e)}")
                    return None

            # Process transcript text
            transcript_pieces = transcript.fetch()
            transcript_text = ' '.join(t['text'] for t in transcript_pieces)
            
            # Create document
            doc = Document(
                page_content=transcript_text,
                metadata={"source": video_id, "url": video_url}
            )
            
            st.success("✅ Transcript loaded successfully!")
            return [doc]

        except Exception as e:
            st.error(f"❌ Error loading transcript: {str(e)}")
            return None

    def setup_qa_chain(self, documents: List[Document]) -> Optional[ConversationalRetrievalChain]:
        """Set up the question-answering chain."""
        try:
            # Split text into manageable chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(documents)
            
            # Create vector store
            vectorstore = FAISS.from_documents(chunks, self.embeddings)
            
            # Initialize memory and chain
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=vectorstore.as_retriever(),
                memory=memory,
                verbose=False
            )
            
            return qa_chain

        except Exception as e:
            st.error(f"❌ Error setting up QA chain: {str(e)}")
            return None

def main():
    """Main Streamlit application."""
    st.title("🎥 YouTube Video Chat Assistant")
    
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "video_loaded" not in st.session_state:
        st.session_state.video_loaded = False
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = YouTubeChatbot()

    # Sidebar
    with st.sidebar:
        st.markdown("""
        ### 📝 Instructions
        1. Enter a YouTube video URL
        2. Wait for transcript processing
        3. Start chatting about the video!
        
        ### ℹ️ Notes
        - Video must have English captions
        - Works with any public YouTube video
        - Supports natural conversation
        """)

    # Main interface
    video_url = st.text_input("🔗 Enter YouTube URL:", placeholder="https://www.youtube.com/watch?v=...")
    
    if video_url and st.button("🚀 Load Video", type="primary"):
        with st.spinner("📝 Processing video transcript..."):
            # Reset states
            st.session_state.video_loaded = False
            st.session_state.chat_history = []
            
            # Load and process video
            transcript = st.session_state.chatbot.load_transcript(video_url)
            if transcript:
                qa_chain = st.session_state.chatbot.setup_qa_chain(transcript)
                if qa_chain:
                    st.session_state.qa_chain = qa_chain
                    st.session_state.video_loaded = True
                    st.success("🎉 Ready to chat about the video!")

    # Chat interface
    if st.session_state.video_loaded and st.session_state.qa_chain:
        st.markdown("### 💭 Chat")
        
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
                with st.spinner("🤔 Thinking..."):
                    try:
                        response = st.session_state.qa_chain({"question": prompt})
                        st.write(response['answer'])
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": response['answer']}
                        )
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    
    elif not video_url:
        st.info("👆 Start by entering a YouTube video URL above")

if __name__ == "__main__":
    main()
