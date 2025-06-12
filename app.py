"""
Educational Tutor Agent - Fixed and Optimized Streamlit Application

Run with: streamlit run app.py --server.fileWatcherType none
"""

import streamlit as st
import os
import warnings
from dotenv import load_dotenv
import logging
import requests
import json
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Optimized compatibility setup
os.environ.update({
    "TOKENIZERS_PARALLELISM": "false",
    "OMP_NUM_THREADS": "1"
})
warnings.filterwarnings("ignore")

# Import the real educational agent
try:
    from tutor_agent import setup_educational_agent, setup_web_search_tool
    REAL_AGENT_AVAILABLE = True
    logger.info("✅ Real educational agent available")
except ImportError as e:
    REAL_AGENT_AVAILABLE = False
    logger.warning(f"⚠️ Real agent not available, using mock: {e}")

# Fallback Mock Educational Agent (only used if real agent fails)
class MockEducationalAgent:
    """Fallback mock educational agent that provides basic science answers"""
    
    def __init__(self):
        self.knowledge_base = {
            "osmosis": {
                "answer": "Osmosis is the movement of water molecules through a selectively permeable membrane from an area of high water concentration (low solute concentration) to an area of low water concentration (high solute concentration). This process continues until equilibrium is reached.",
                "sources": ["Biology Textbook - Cell Biology", "ScienceQA - Membrane Transport"]
            },
            "photosynthesis": {
                "answer": "Photosynthesis is the process by which plants convert light energy into chemical energy (glucose). The equation is: 6CO₂ + 6H₂O + light energy → C₆H₁₂O₆ + 6O₂. This occurs in chloroplasts using chlorophyll.",
                "sources": ["Biology Textbook - Plant Biology", "ScienceQA - Energy Processes"]
            },
            "mitosis": {
                "answer": "Mitosis is cell division that produces two identical diploid cells. It consists of prophase, metaphase, anaphase, and telophase. DNA replicates before division, and chromosomes are equally distributed to daughter cells.",
                "sources": ["Biology Textbook - Cell Division", "ScienceQA - Cellular Processes"]
            },
            "respiration": {
                "answer": "Cellular respiration is the process that breaks down glucose to produce ATP energy. The equation is: C₆H₁₂O₆ + 6O₂ → 6CO₂ + 6H₂O + ATP. It occurs in mitochondria through glycolysis, Krebs cycle, and electron transport.",
                "sources": ["Biology Textbook - Metabolism", "ScienceQA - Energy Production"]
            }
        }
    
    def __call__(self, inputs: Dict[str, str]) -> Dict[str, Any]:
        question = inputs.get("question", "").lower()
        
        # Search for relevant topics in the knowledge base
        for topic, data in self.knowledge_base.items():
            if topic in question:
                return {
                    "answer": data["answer"],
                    "source_documents": [
                        {"metadata": {"source": source, "subject": "Biology"}}
                        for source in data["sources"]
                    ]
                }
        
        # Generate a generic science response for unknown topics
        if any(word in question for word in ["science", "biology", "chemistry", "physics"]):
            return {
                "answer": "This is an interesting science question. While I have some information in my knowledge base, you might benefit from additional web research for a comprehensive answer.",
                "source_documents": [{"metadata": {"source": "ScienceQA", "subject": "General Science"}}]
            }
        
        return {
            "answer": "I don't have specific information about this topic in my knowledge base. Let me search the web for current information.",
            "source_documents": []
        }

class FallbackWebSearchTool:
    """Fallback web search tool using free APIs"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key  # Not used but kept for compatibility
        self.search_engines = {
            "duckduckgo": "https://api.duckduckgo.com/",
            "wikipedia": "https://en.wikipedia.org/api/rest_v1/"
        }
    
    def _run(self, query: str) -> str:
        """Run web search and return formatted results"""
        try:
            # Use DuckDuckGo Instant Answer API (no key required)
            results = self._search_duckduckgo(query)
            if results:
                return results
            
            # Fallback to Wikipedia search
            results = self._search_wikipedia(query)
            if results:
                return results
            
            return "No web search results found for this query."
            
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return f"Web search temporarily unavailable: {str(e)}"
    
    def _search_duckduckgo(self, query: str) -> str:
        """Search using DuckDuckGo Instant Answer API"""
        try:
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1"
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                # Format the response
                result_parts = []
                
                if data.get("Abstract"):
                    result_parts.append(f"**{data.get('Heading', 'Answer')}**\n{data['Abstract']}")
                    if data.get("AbstractURL"):
                        result_parts.append(f"Source: {data['AbstractURL']}")
                
                if data.get("Definition"):
                    result_parts.append(f"**Definition**\n{data['Definition']}")
                    if data.get("DefinitionURL"):
                        result_parts.append(f"Source: {data['DefinitionURL']}")
                
                # Add related topics
                if data.get("RelatedTopics"):
                    for topic in data["RelatedTopics"][:2]:  # Limit to 2 topics
                        if isinstance(topic, dict) and topic.get("Text"):
                            result_parts.append(f"**Related Information**\n{topic['Text']}")
                            if topic.get("FirstURL"):
                                result_parts.append(f"Source: {topic['FirstURL']}")
                
                if result_parts:
                    return "\n\n".join(result_parts)
            
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
        
        return ""
    
    def _search_wikipedia(self, query: str) -> str:
        """Search Wikipedia for information"""
        try:
            # Search for pages
            search_url = "https://en.wikipedia.org/api/rest_v1/page/summary/"
            # Clean query for Wikipedia
            clean_query = query.replace(" ", "_")
            
            response = requests.get(f"{search_url}{clean_query}", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("extract"):
                    result = f"**{data.get('title', 'Wikipedia Result')}**\n{data['extract']}"
                    if data.get("content_urls", {}).get("desktop", {}).get("page"):
                        result += f"\n\nSource: {data['content_urls']['desktop']['page']}"
                    return result
            
        except Exception as e:
            logger.error(f"Wikipedia search error: {e}")
        
        return ""

# This function is now imported from tutor_agent.py, but we keep this as a fallback
def setup_fallback_agent():
    """Initialize the fallback educational agent"""
    return MockEducationalAgent()

# Page Configuration
st.set_page_config(
    page_title="Educational Tutor Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Optimized CSS with better mobile responsiveness
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    .main .block-container {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        max-width: 1200px;
    }
    
    .main-title {
        text-align: center;
        color: #1565c0;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(21, 101, 192, 0.3);
    }
    
    .subtitle {
        text-align: center;
        color: #424242;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 500;
        line-height: 1.4;
    }
    
    .answer-card {
        background: #ffffff;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #1a1a1a;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        border: 2px solid #e0e0e0;
    }
    
    .kb-answer { 
        border-color: #1565c0; 
        background: linear-gradient(135deg, #f8f9ff 0%, #ffffff 100%);
    }
    
    .web-result { 
        border-color: #d32f2f; 
        background: linear-gradient(135deg, #fff8f8 0%, #ffffff 100%);
    }
    
    .warning-card { 
        background: linear-gradient(135deg, #fff8e1 0%, #ffffff 100%);
        border-color: #f57c00; 
        color: #bf360c; 
    }
    
    .error-card {
        background: linear-gradient(135deg, #ffebee 0%, #ffffff 100%);
        border-color: #d32f2f;
        color: #c62828;
    }
    
    .answer-content {
        background: #f8f9fa;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 4px solid #1565c0;
        margin: 1rem 0;
        color: #212529;
        line-height: 1.7;
        font-size: 1rem;
    }
    
    .sources-section {
        background: #e8f4f8;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 4px solid #26a69a;
        margin: 1rem 0;
        color: #004d40;
    }
    
    .source-item {
        padding: 0.6rem 0;
        border-bottom: 1px solid #b2dfdb;
        color: #00695c;
        font-size: 0.95rem;
    }
    
    .source-item:last-child { border-bottom: none; }
    
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #333;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.6rem 1.2rem;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: 500;
        margin: 0.5rem 0;
    }
    
    .status-enabled {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .status-disabled {
        background: linear-gradient(135deg, #f8d7da 0%, #f1b0b7 100%);
        color: #721c24;
        border: 1px solid #f1b0b7;
    }
    
    .source-link {
        color: #667eea;
        text-decoration: none;
        font-weight: 500;
        transition: all 0.3s ease;
        word-break: break-word;
    }
    
    .source-link:hover { 
        color: #764ba2;
        text-decoration: underline;
    }
    
    .error-message {
        background: #ffebee;
        color: #c62828;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #f44336;
        margin: 1rem 0;
    }
    
    .success-message {
        background: #e8f5e8;
        color: #2e7d32;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2rem;
        }
        
        .subtitle {
            font-size: 1rem;
        }
        
        .main .block-container {
            padding: 1rem;
            margin: 0.5rem;
        }
        
        .answer-card {
            padding: 1rem;
        }
        
        .section-header {
            font-size: 1.2rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Title and Description
st.markdown('<h1 class="main-title">🎓 Educational Tutor Agent</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Ask any science question and get precise, reliable answers from our knowledge base and trusted web sources!</p>', unsafe_allow_html=True)

# Session state initialization with error handling
def init_session_state():
    """Initialize all session state variables with proper defaults."""
    defaults = {
        "messages": [],
        "qa_chain": None,
        "exa_tool": None,
        "initialized": False,
        "error_state": False,
        "initialization_attempted": False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Initialize session state
init_session_state()

# Improved sidebar with better error handling
def setup_sidebar():
    """Setup sidebar with web search configuration."""
    with st.sidebar:
        st.markdown("## 🌐 Web Search Configuration")
        
        # Initialize enhanced web search tool
        if not st.session_state.exa_tool:
            if REAL_AGENT_AVAILABLE:
                try:
                    st.session_state.exa_tool = setup_web_search_tool()
                    if st.session_state.exa_tool:
                        # Check if Tavily is available
                        tavily_available = st.session_state.exa_tool.tavily_client is not None
                        
                        if tavily_available:
                            st.markdown('<div class="status-indicator status-enabled">🌐 Enhanced Web Search Active (Tavily)</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="status-indicator status-enabled">🌐 Basic Web Search Active (Free APIs)</div>', unsafe_allow_html=True)
                    else:
                        st.session_state.exa_tool = FallbackWebSearchTool()
                        st.markdown('<div class="status-indicator status-enabled">🌐 Basic Web Search Active (Free APIs)</div>', unsafe_allow_html=True)
                except Exception as e:
                    logger.warning(f"Failed to initialize enhanced search tool: {e}")
                    st.session_state.exa_tool = FallbackWebSearchTool()
                    st.markdown('<div class="status-indicator status-enabled">🌐 Basic Web Search Active (Fallback)</div>', unsafe_allow_html=True)
            else:
                try:
                    st.session_state.exa_tool = FallbackWebSearchTool()
                    st.markdown('<div class="status-indicator status-enabled">🌐 Basic Web Search Active (Free APIs)</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"❌ Failed to initialize web search: {str(e)}")
                    st.session_state.exa_tool = None
        else:
            # Display current search capabilities
            if hasattr(st.session_state.exa_tool, 'tavily_client') and st.session_state.exa_tool.tavily_client:
                st.markdown('<div class="status-indicator status-enabled">🌐 Enhanced Web Search Active (Tavily)</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-indicator status-enabled">🌐 Basic Web Search Active</div>', unsafe_allow_html=True)
        
        # API key input for enhanced search
        st.markdown("### 🔑 Tavily API Key (Optional)")
        
        tavily_key_input = st.text_input(
            "Tavily API Key:", 
            type="password", 
            placeholder="tvly-...",
            help="Get your API key from https://tavily.com for enhanced search results"
        )
        
        # Update search tool if key is provided
        if tavily_key_input:
            if len(tavily_key_input) > 10:
                try:
                    if REAL_AGENT_AVAILABLE:
                        # Re-initialize with new key
                        os.environ["TAVILY_API_KEY"] = tavily_key_input
                        
                        st.session_state.exa_tool = setup_web_search_tool()
                        if st.session_state.exa_tool and st.session_state.exa_tool.tavily_client:
                            st.success("✅ Tavily search enabled!")
                        else:
                            st.error("❌ Failed to initialize Tavily search")
                    else:
                        st.warning("⚠️ Real agent not available - enhanced search not supported")
                except Exception as e:
                    st.error(f"❌ Failed to initialize enhanced search: {str(e)}")
            else:
                st.warning("⚠️ Please enter a valid Tavily API key (minimum 10 characters)")
        
        st.markdown("---")
        
        # About section
        st.markdown("## ℹ️ About")
        st.markdown("""
        This AI tutor combines:
        - **Knowledge Base**: Pre-trained science knowledge
        - **Web Search**: Real-time information from trusted sources
        - **Smart Routing**: Automatically searches web for incomplete answers
        """)
        
        st.markdown("---")
        
        # Controls
        st.markdown("## 🎛️ Controls")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        
        with col2:
            if st.button("🔄 Reset Agent", use_container_width=True):
                # Reset all states
                for key in ["qa_chain", "exa_tool", "initialized", "initialization_attempted"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        
        # Display current status
        st.markdown("### 📊 System Status")
        kb_status = "✅ Ready" if st.session_state.get("qa_chain") else "❌ Not Ready"
        web_status = "✅ Available" if st.session_state.get("exa_tool") else "❌ Unavailable"
        
        st.markdown(f"**Knowledge Base:** {kb_status}")
        st.markdown(f"**Web Search:** {web_status}")

# Setup sidebar
setup_sidebar()

# Cached agent initialization with better error handling
@st.cache_resource
def get_qa_chain():
    """Initialize and cache the QA chain with proper error handling."""
    try:
        logger.info("Initializing QA chain...")
        
        if REAL_AGENT_AVAILABLE:
            logger.info("🤖 Using real educational agent with retrieval QA...")
            qa_chain = setup_educational_agent()
        else:
            logger.warning("⚠️ Using fallback mock agent...")
            qa_chain = setup_fallback_agent()
            
        logger.info("QA chain initialized successfully")
        return qa_chain
    except Exception as e:
        logger.error(f"Failed to initialize QA chain: {e}")
        logger.info("🔄 Falling back to mock agent...")
        try:
            qa_chain = setup_fallback_agent()
            logger.info("✅ Fallback agent initialized")
            return qa_chain
        except Exception as fallback_error:
            logger.error(f"Even fallback failed: {fallback_error}")
            raise

# Initialize agent with comprehensive error handling
def initialize_agent():
    """Initialize the educational agent with proper error handling."""
    if st.session_state.initialized:
        return True
    
    if st.session_state.initialization_attempted and st.session_state.error_state:
        return False
    
    st.session_state.initialization_attempted = True
    
    try:
        with st.spinner("🚀 Initializing AI systems... This may take a moment."):
            # Add progress indicator
            progress_bar = st.progress(0)
            st.write("Loading knowledge base...")
            progress_bar.progress(33)
            
            # Initialize QA chain
            st.session_state.qa_chain = get_qa_chain()
            progress_bar.progress(66)
            
            st.write("Finalizing setup...")
            progress_bar.progress(100)
            
            st.session_state.initialized = True
            st.session_state.error_state = False
            
            # Clear progress indicators
            progress_bar.empty()
            
            st.markdown('<div class="success-message">🎓 <strong>Educational Tutor Ready!</strong> Ask any science question to get started.</div>', unsafe_allow_html=True)
            return True
            
    except Exception as e:
        st.session_state.error_state = True
        error_msg = f"❌ **Initialization Failed:** {str(e)}"
        st.markdown(f'<div class="error-message">{error_msg}</div>', unsafe_allow_html=True)
        
        # Provide troubleshooting steps
        with st.expander("🔧 Troubleshooting Steps"):
            st.markdown("""
            1. **Check Dependencies**: Ensure all required packages are installed
            2. **Verify Files**: Make sure all dependencies are available
            3. **Restart Application**: Try refreshing the page or restarting Streamlit
            4. **Check Logs**: Look for detailed error messages in the console
            5. **Contact Support**: If issues persist, please report the error
            """)
        
        return False

# Initialize the agent
agent_ready = initialize_agent()

if not agent_ready:
    st.stop()

# Enhanced quality check function with better logic
def is_poor_answer(answer: str, question: str) -> tuple:
    """
    Check if answer quality is poor and return reason.
    Returns (is_poor, reason)
    """
    if not answer or not answer.strip():
        return True, "Empty answer"
    
    answer_lower = answer.lower().strip()
    question_lower = question.lower().strip()
    
    # Basic quality checks
    if len(answer.split()) < 8:
        return True, "Too short"
    
    # Check for non-answers
    non_answers = [
        "doesn't directly answer",
        "don't have specific information",
        "i don't know",
        "i'm not sure",
        "unclear",
        "unknown"
    ]
    
    if any(phrase in answer_lower for phrase in non_answers):
        return True, "Non-answer detected"
    
    # Check for generic responses
    if answer_lower.strip() in ["a product", "unknown", "unclear", "yes", "no"]:
        return True, "Too generic"
    
    return False, "Good quality"

# Improved web result formatter
def format_web_results(web_results: str):
    """Parse and format web results with better structure."""
    if not web_results or not web_results.strip():
        st.warning("No web results to display.")
        return
    
    try:
        sections = [s.strip() for s in web_results.split('\n\n') if s.strip()]
        
        if not sections:
            st.warning("No valid web results found.")
            return
        
        for i, section in enumerate(sections):
            if not section:
                continue
            
            lines = [line.strip() for line in section.split('\n') if line.strip()]
            if len(lines) < 1:
                continue
            
            # Extract title (first line, remove markdown formatting)
            title = lines[0].replace('**', '').replace('*', '').strip()
            if title.startswith('#'):
                title = title.lstrip('#').strip()
            
            # Extract content and source
            content_lines = []
            source_url = ""
            
            for line in lines[1:]:
                if line.startswith('Source:'):
                    source_url = line.replace('Source:', '').strip()
                elif line.startswith('URL:'):
                    source_url = line.replace('URL:', '').strip()
                elif not line.startswith('http'):
                    content_lines.append(line)
                else:
                    source_url = line.strip()
            
            content = ' '.join(content_lines).strip()
            
            if content or (not content and len(lines) == 1):  # Display if there's content or just a title
                display_content = content if content else title
                st.markdown(f'<div class="answer-content"><strong>📄 {title}</strong><br><br>{display_content}</div>', unsafe_allow_html=True)
                
                if source_url:
                    # Clean up the URL
                    if not source_url.startswith('http'):
                        source_url = 'https://' + source_url
                    
                    st.markdown(f'<div class="sources-section"><strong>🔗 Source:</strong> <a href="{source_url}" target="_blank" class="source-link">{source_url}</a></div>', unsafe_allow_html=True)
                
                if i < len(sections) - 1:  # Add separator except for last item
                    st.markdown("---")
    
    except Exception as e:
        st.error(f"Error formatting web results: {str(e)}")
        # Fallback: display raw results
        st.markdown(f'<div class="answer-content">{web_results}</div>', unsafe_allow_html=True)

# Display chat history
def display_chat_history():
    """Display all previous chat messages."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                # For assistant messages, preserve the HTML formatting
                st.markdown(message["content"], unsafe_allow_html=True)
            else:
                st.markdown(message["content"])

# Display chat history
display_chat_history()

# Main chat interface
if question := st.chat_input("Ask a science question...", key="main_chat_input"):
    # Validate question
    if not question.strip():
        st.warning("Please enter a valid question.")
        st.stop()
    
    if len(question.strip()) < 3:
        st.warning("Please enter a more detailed question.")
        st.stop()
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": question})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(question)
    
    # Generate assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        with st.spinner("🤔 Analyzing your question..."):
            try:
                # Initialize response components
                kb_answer = ""
                kb_sources = []
                web_results = ""
                is_poor = True
                quality_reason = "No response"
                
                # Get knowledge base response
                try:
                    if REAL_AGENT_AVAILABLE and hasattr(st.session_state.qa_chain, 'invoke'):
                        # Real LangChain agent (returns LangChain response)
                        response = st.session_state.qa_chain.invoke({"question": question})
                        kb_answer = response.get("answer", "I couldn't generate an answer from the knowledge base.")
                        kb_sources = response.get("source_documents", [])
                    elif hasattr(st.session_state.qa_chain, '__call__'):
                        # Try direct call (works for both real and mock agents)
                        response = st.session_state.qa_chain({"question": question})
                        kb_answer = response.get("answer", "I couldn't generate an answer from the knowledge base.")
                        if "source_documents" in response:
                            # Check if sources are in the right format
                            sources = response.get("source_documents", [])
                            if sources and isinstance(sources[0], dict) and "metadata" in sources[0]:
                                kb_sources = sources  # Already in correct format
                            else:
                                # Convert Document objects to expected format
                                kb_sources = []
                                for doc in sources:
                                    if hasattr(doc, 'metadata'):
                                        kb_sources.append({"metadata": doc.metadata})
                                    elif isinstance(doc, dict):
                                        kb_sources.append({"metadata": doc.get("metadata", {})})
                                    else:
                                        kb_sources.append({"metadata": {}})
                        else:
                            kb_sources = []
                    else:
                        # Fallback error
                        raise ValueError("QA chain is not callable")
                    
                    # Check answer quality
                    is_poor, quality_reason = is_poor_answer(kb_answer, question)
                    
                except Exception as kb_error:
                    logger.error(f"Knowledge base error: {kb_error}")
                    kb_answer = f"Knowledge base temporarily unavailable: {str(kb_error)}"
                    is_poor = True
                    quality_reason = "KB Error"
                
                # Decide which answer to show
                final_answer = ""
                answer_source = ""
                use_web_search = False
                
                # If knowledge base answer is good, use it
                if not is_poor and kb_answer and "temporarily unavailable" not in kb_answer:
                    final_answer = kb_answer
                    answer_source = "📚 Knowledge Base"
                    use_web_search = False
                else:
                    # Knowledge base answer is poor or unavailable, try web search
                    if st.session_state.exa_tool:
                        with st.spinner("🌐 Searching web for information..."):
                            try:
                                web_results = st.session_state.exa_tool._run(question)
                                if web_results and "not available" not in web_results.lower() and "error" not in web_results.lower():
                                    final_answer = web_results
                                    answer_source = "🌐 Web Search"
                                    use_web_search = True
                                else:
                                    # Both failed, show KB answer with warning
                                    final_answer = kb_answer if kb_answer else "I couldn't find a good answer to your question."
                                    answer_source = f"⚠️ Limited Information ({quality_reason})"
                                    use_web_search = False
                            except Exception as e:
                                logger.error(f"Web search error: {e}")
                                # Web search failed, show KB answer with warning
                                final_answer = kb_answer if kb_answer else "I couldn't find a good answer to your question."
                                answer_source = f"⚠️ Limited Information ({quality_reason})"
                                use_web_search = False
                    else:
                        # No web search available, show KB answer with warning
                        final_answer = kb_answer if kb_answer else "I couldn't find a good answer to your question."
                        answer_source = f"⚠️ Limited Information ({quality_reason})"
                        use_web_search = False
                
                # Build single response HTML
                if use_web_search:
                    # Web search response - use custom formatting
                    response_html = f'''
                    <div class="answer-card web-result">
                        <div class="section-header">{answer_source}</div>
                    '''
                    response_placeholder.markdown(response_html, unsafe_allow_html=True)
                    format_web_results(final_answer)
                    response_html += '</div>'
                else:
                    # Knowledge base or error response
                    card_class = "kb-answer" if not is_poor else "warning-card"
                    response_html = f'''
                    <div class="answer-card {card_class}">
                        <div class="section-header">{answer_source}</div>
                        <div class="answer-content">{final_answer}</div>
                    '''
                    
                    # Add sources for good KB answers
                    if not is_poor and kb_sources:
                        response_html += '<div class="sources-section"><strong>📖 Sources:</strong><br>'
                        for doc in kb_sources[:3]:  # Show up to 3 sources
                            metadata = doc.get("metadata", {}) if isinstance(doc, dict) else {}
                            subject = metadata.get('subject', 'Science')
                            source_type = metadata.get('source', 'ScienceQA')
                            response_html += f'<div class="source-item">• {source_type} - {subject}</div>'
                        response_html += '</div>'
                    
                    response_html += '</div>'
                    response_placeholder.markdown(response_html, unsafe_allow_html=True)
                
                # Store conversation (simplified version for storage)
                st.session_state.messages.append({"role": "assistant", "content": response_html})
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error while processing your question: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                logger.error(f"Chat error: {e}")

# Footer with additional information
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem; padding: 1rem;">
    <p>🎓 Educational Tutor Agent | Powered by AI and Real-time Web Search</p>
    <p>For best results, ask specific science questions about topics like biology, chemistry, physics, or earth science.</p>
</div>
""", unsafe_allow_html=True)