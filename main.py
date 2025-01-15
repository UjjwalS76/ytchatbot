

import streamlit as st
import os
import re
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi

# LangChain + Google Generative AI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document

# Community tools
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Ensure we have the Google API key from secrets
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("No Google API key found in Streamlit secrets!")
    st.stop()

def extract_video_id(url: str):
    """
    Extract the video ID from various YouTube URL formats.
    """
    youtube_regex = (
        r'(https?://)?(www\.)?'
        '(youtube|youtu|youtube-nocookie)\.(com|be)/'
        '(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
    )
    match = re.match(youtube_regex, url)
    if match:
        return match.group(6)
    return None

def get_video_info(url: str):
    """
    Get basic info (title, author, length) from a YouTube video using pytube.
    """
    try:
        yt = YouTube(url)
        return {
            "title": yt.title,
            "author": yt.author,
            "length": yt.length
        }
    except Exception as e:
        st.error(f"Error getting video info: {e}")
        return None

def load_video_transcript(video_url: str):
    """
    Load the transcript from a YouTube video, prioritizing English transcripts.
    Returns a list containing a single LangChain Document with transcript text.
    """
    try:
        # Validate & extract the ID
        video_id = extract_video_id(video_url)
        if not video_id:
            st.error("Invalid YouTube URL format.")
            return None

        # Get & display some info
        info = get_video_info(video_url)
        if info:
            st.write(f"**Title:** {info['title']}")
            st.write(f"**Channel:** {info['author']}")
            st.write(f"**Video Length (s):** {info['length']}")

        # Attempt to retrieve available transcripts
        st.info("Fetching transcripts...")
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Try to get English transcripts (generated or manual)
        transcript = None
        try:
            transcript = transcript_list.find_generated_transcript(['en'])
        except:
            try:
                transcript = transcript_list.find_manually_created_transcript(['en'])
            except:
                # If none found, try any transcript and translate to English
                try:
                    # This might fail if there's no auto-translation
                    transcript = transcript_list.find_generated_transcript([])
                    transcript = transcript.translate('en')
                except Exception as e:
                    st.error(f"No suitable transcript found: {e}")
                    return None

        if not transcript:
            st.error("No transcripts found for this video in English.")
            return None

        # Build the final text
        raw_transcript = transcript.fetch()
        transcript_text = ' '.join([chunk['text'] for chunk in raw_transcript])

        if not transcript_text.strip():
            st.error("Transcript is empty.")
            return None

        # Construct a Document object for LangChain
        doc_meta = {
            "source": video_id,
            "title": info['title'] if info else "Unknown",
            "author": info['author'] if info else "Unknown",
        }
        langchain_doc = Document(page_content=transcript_text, metadata=doc_meta)

        st.success("Transcript loaded successfully!")
        return [langchain_doc]

    except Exception as e:
        st.error(f"Failed to load transcript: {e}")
        st.info("Ensure the video is public and has English (or auto-generated) captions.")
        return None

def setup_qa_chain(transcript_docs):
    """
    Build a conversational retrieval chain from the provided transcript docs.
    """
    try:
        with st.spinner("Setting up the chat system..."):
            st.info("Splitting text into chunks...")
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(transcript_docs)

            st.info("Creating embeddings...")
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

            st.info("Building vector store...")
            vectorstore = Chroma.from_documents(chunks, embeddings)

            st.info("Initializing LLM...")
            llm = ChatGoogleGenerativeAI(
                model="gemini-pro",
                temperature=0.7,
            )

            st.info("Setting up conversation memory...")
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )

            st.info("Finalizing QA chain...")
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(),
                memory=memory,
                return_source_documents=False,
            )

            st.success("✅ Chat system ready!")
            return qa_chain
    except Exception as e:
        st.error(f"Error in setup_qa_chain: {e}")
        return None

def main():
    st.title("💬 YouTube Video Chat Assistant")

    # Initialize session states
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "video_loaded" not in st.session_state:
        st.session_state.video_loaded = False
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None

    # Sidebar with instructions
    with st.sidebar:
        st.markdown("""
        ### Instructions
        1. Enter a public YouTube video URL with English captions.
        2. Click "**Load Video**" to fetch and process its transcript.
        3. Ask questions about the video content in the chat box.

        **Note:** 
        - Only English transcripts are supported currently.
        - Some private or region-restricted videos may fail to load transcripts.
        """)
    
    # Text input for the YouTube URL
    video_url = st.text_input(
        "Enter YouTube Video URL:",
        help="Paste a full YouTube link, e.g. https://www.youtube.com/watch?v=..."
    )

    # Button to process video
    if video_url:
        if st.button("Load Video"):
            st.session_state.video_loaded = False
            st.session_state.chat_history = []

            with st.spinner("Loading and processing transcript..."):
                transcript_docs = load_video_transcript(video_url)
                if transcript_docs:
                    chain = setup_qa_chain(transcript_docs)
                    if chain:
                        st.session_state.qa_chain = chain
                        st.session_state.video_loaded = True
                        st.success("✅ Video processed successfully! Start chatting below.")

    # Chat interface
    if st.session_state.video_loaded and st.session_state.qa_chain:
        st.subheader("Chat about the video")
        
        # Display any existing chat messages
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # User input in the chat
        user_input = st.chat_input("Ask about the video...")
        if user_input:
            # Append user question
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.qa_chain({"question": user_input})
                        answer = response["answer"]
                        st.write(answer)
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": answer}
                        )
                    except Exception as e:
                        st.error(f"Error generating response: {e}")
    
    elif not video_url:
        st.info("Enter a YouTube link above to get started!")

if __name__ == "__main__":
    main()
