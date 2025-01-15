import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from youtube_transcript_api import YouTubeTranscriptApi
import os
import re
from pytube import YouTube

def extract_video_id(url):
    """Extract the video ID from various YouTube URL formats."""
    youtube_regex = (
        r'(https?://)?(www\.)?'
        '(youtube|youtu|youtube-nocookie)\.(com|be)/'
        '(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')
    
    youtube_regex_match = re.match(youtube_regex, url)
    
    if youtube_regex_match:
        return youtube_regex_match.group(6)
    return None

def get_video_info(url):
    """Get video title and other info using pytube."""
    try:
        yt = YouTube(url)
        return {
            'title': yt.title,
            'author': yt.author,
            'length': yt.length
        }
    except Exception as e:
        st.error(f"Error getting video info: {str(e)}")
        return None

def load_video_transcript(video_url):
    """Load and return the transcript of a YouTube video."""
    try:
        # First, validate and extract video ID
        video_id = extract_video_id(video_url)
        if not video_id:
            st.error("Invalid YouTube URL format")
            return None

        # Display video info
        st.info("Fetching video information...")
        video_info = get_video_info(video_url)
        if video_info:
            st.write(f"📺 Video: {video_info['title']}")
            st.write(f"👤 Channel: {video_info['author']}")

        # Get available transcripts
        st.info("Fetching available transcripts...")
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Try to get English transcript first
        try:
            transcript = transcript_list.find_generated_transcript(['en'])
        except:
            try:
                transcript = transcript_list.find_manually_created_transcript(['en'])
            except:
                try:
                    # Get any available transcript and translate to English
                    transcript = transcript_list.find_generated_transcript()
                    transcript = transcript.translate('en')
                except Exception as e:
                    st.error(f"No suitable transcript found: {str(e)}")
                    return None

        # Get the actual transcript text
        transcript_pieces = transcript.fetch()
        transcript_text = ' '.join([t['text'] for t in transcript_pieces])
        
        if not transcript_text.strip():
            st.error("Transcript is empty")
            return None
            
        # Create document
        from langchain.schema.document import Document
        doc = Document(
            page_content=transcript_text,
            metadata={
                "source": video_id,
                "title": video_info['title'] if video_info else "Unknown",
                "author": video_info['author'] if video_info else "Unknown"
            }
        )
        
        st.success("Transcript loaded successfully!")
        return [doc]

    except Exception as e:
        st.error(f"Failed to load transcript: {str(e)}")
        st.info("Please ensure the video has captions enabled and is publicly accessible")
        return None

def setup_qa_chain(transcript_data):
    """Set up the question-answering chain with the video transcript."""
    try:
        with st.status("Setting up chat system...") as status:
            status.write("Splitting text into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(transcript_data)
            
            status.write("Creating embeddings...")
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            
            status.write("Building vector database...")
            vectorstore = Chroma.from_documents(chunks, embeddings)
            
            status.write("Initializing AI model...")
            llm = ChatGoogleGenerativeAI(
                model="gemini-pro",
                temperature=0.7,
            )
            
            status.write("Setting up conversation memory...")
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            status.write("Finalizing setup...")
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(),
                memory=memory,
                return_source_documents=False,
            )
            
            status.update(label="✅ Chat system ready!", state="complete")
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
        
        ### Note:
        - Video must have closed captions available
        - Only processes English transcripts currently
        """)
        
        if 'GOOGLE_API_KEY' in st.secrets:
            st.success("API Key: Configured ✓")
        else:
            st.error("API Key: Missing ✗")
    
    # Video URL input
    video_url = st.text_input(
        "Enter YouTube Video URL:",
        help="Paste the full YouTube video URL here"
    )
    
    # Load video button
    if video_url:
        col1, col2 = st.columns([1, 6])
        with col1:
            process_video = st.button("Load Video", type="primary")
        
        if process_video:
            with st.spinner("Loading video..."):
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
