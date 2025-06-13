"""
Educational Tutor Agent - Fixed and Optimized Core Logic
"""

import os
import warnings
import logging
from typing import Optional, List, Dict, Any
from functools import lru_cache

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Compatibility setup (once)
os.environ.update({
    "TOKENIZERS_PARALLELISM": "false",
    "OMP_NUM_THREADS": "1",
    "CUDA_VISIBLE_DEVICES": "0" if os.getenv("CUDA_VISIBLE_DEVICES") else ""
})
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Core imports with error handling
try:
    from langchain.schema import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    
    # Use new HuggingFace imports
    try:
        from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
    except ImportError:
        # Fallback to old imports
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_community.llms import HuggingFacePipeline
        except ImportError:
            from langchain.embeddings import HuggingFaceEmbeddings
            from langchain.llms import HuggingFacePipeline
    
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferMemory
    from langchain.tools import BaseTool
    from langchain.callbacks.manager import CallbackManagerForToolRun
    from langchain.prompts import PromptTemplate
    
    import torch
    from datasets import load_dataset
    from transformers import pipeline
    
    logger.info("✅ Core dependencies loaded successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import core dependencies: {e}")
    raise ImportError(f"Missing required dependencies: {e}")

# Web search imports
try:
    from web_search import create_web_search_tool
    WEB_SEARCH_AVAILABLE = True
    logger.info("✅ Enhanced web search available")
except ImportError:
    WEB_SEARCH_AVAILABLE = False
    logger.warning("⚠️ Web search module not available")

# Optimized configuration class
class Config:
    """Configuration settings for the educational agent."""
    
    # Model settings - using more reliable models
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = "google/flan-t5-base"  # Changed from small for better performance
    
    # Processing settings
    CHUNK_SIZE = 1000  # Increased for better context
    CHUNK_OVERLAP = 200  # Increased overlap
    TOP_K_RETRIEVAL = 4  # Increased for better coverage
    MAX_DATASET_SIZE = 500  # Increased for better knowledge base
    
    # Generation settings - improved for better answers
    MAX_NEW_TOKENS = 256  # Increased for more detailed answers
    MIN_LENGTH = 30  # Increased minimum
    TEMPERATURE = 0.1  # Slightly increased for some creativity
    DO_SAMPLE = True  # Enable sampling for better diversity
    NUM_BEAMS = 2  # Add beam search for better quality
    
    # API settings
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    
    # Device settings
    FORCE_CPU = os.getenv("FORCE_CPU", "false").lower() == "true"

config = Config()

# Configure torch with better error handling
try:
    torch.set_num_threads(1)
    if hasattr(torch, 'set_grad_enabled'):
        torch.set_grad_enabled(False)
    logger.info("✅ PyTorch configured")
except Exception as e:
    logger.warning(f"⚠️ PyTorch configuration warning: {e}")

@lru_cache(maxsize=1)
def get_device() -> str:
    """Get device configuration with caching and better logic."""
    if config.FORCE_CPU:
        logger.info("🖥️ Forcing CPU usage")
        return 'cpu'
    
    if torch.cuda.is_available():
        logger.info("🚀 CUDA available - using GPU")
        return 'cuda'
    else:
        logger.info("🖥️ CUDA not available - using CPU")
        return 'cpu'

def load_scienceqa_dataset(max_size: int = None) -> Any:
    """Load ScienceQA dataset with better error handling and cloud compatibility."""
    if max_size is None:
        max_size = config.MAX_DATASET_SIZE
    
    try:
        logger.info(f"🔄 Loading ScienceQA dataset (max {max_size} examples)...")
        
        # Check if running in cloud environment
        is_cloud = (os.getenv("STREAMLIT_SHARING_MODE") or 
                   "streamlit.app" in os.getenv("HOSTNAME", "") or
                   os.getenv("STREAMLIT_CLOUD"))
        
        if is_cloud:
            # Cloud-optimized loading - use smaller dataset and more conservative approach
            logger.info("☁️ Cloud environment detected - using optimized loading")
            max_size = min(max_size, 100)  # Limit to 100 examples in cloud
        
        # Try different loading strategies with cloud compatibility
        try:
            # First try with streaming=False for cloud compatibility
            dataset = load_dataset(
                "derek-thomas/ScienceQA", 
                split=f"train[:{max_size}]",
                download_mode="reuse_dataset_if_exists",
                streaming=False,
                verification_mode="no_checks"  # Skip verification for faster loading
            )
        except Exception as e1:
            logger.warning(f"Standard loading failed: {e1}")
            try:
                # Try with different download mode
                dataset = load_dataset(
                    "derek-thomas/ScienceQA", 
                    split=f"train[:{max_size}]",
                    download_mode="reuse_dataset_if_exists",
                    streaming=False
                )
            except Exception as e2:
                logger.warning(f"Reuse loading failed: {e2}")
                try:
                    # Try loading full dataset and selecting subset
                    full_dataset = load_dataset(
                        "derek-thomas/ScienceQA", 
                        split="train",
                        streaming=False,
                        verification_mode="no_checks"
                    )
                    dataset = full_dataset.select(range(min(max_size, len(full_dataset))))
                except Exception as e3:
                    logger.warning(f"Full dataset loading failed: {e3}")
                    # Final fallback with minimal parameters
                    try:
                        dataset = load_dataset("derek-thomas/ScienceQA", split="train")
                        dataset = dataset.select(range(min(max_size, len(dataset))))
                    except Exception as e4:
                        logger.error(f"All loading methods failed: {e4}")
                        raise e4
        
        logger.info(f"✅ Loaded {len(dataset)} examples from ScienceQA")
        return dataset
        
    except Exception as e:
        logger.error(f"❌ Failed to load ScienceQA dataset: {e}")
        
        # In cloud environments, try one more time with very minimal settings
        if is_cloud:
            try:
                logger.info("☁️ Attempting cloud-specific fallback loading...")
                from datasets import Dataset
                # Try to load just a few examples for cloud deployment
                dataset = load_dataset("derek-thomas/ScienceQA", split="train[:50]")
                logger.info(f"✅ Cloud fallback loaded {len(dataset)} examples")
                return dataset
            except Exception as cloud_error:
                logger.warning(f"Cloud fallback also failed: {cloud_error}")
        
        # Try clearing cache as last resort
        try:
            logger.info("🧹 Attempting to clear dataset cache...")
            import shutil
            from datasets import config as datasets_config
            cache_dir = datasets_config.HF_DATASETS_CACHE
            if os.path.exists(cache_dir):
                logger.info(f"Clearing cache directory: {cache_dir}")
                shutil.rmtree(cache_dir, ignore_errors=True)
                # Try loading again after cache clear
                dataset = load_dataset("derek-thomas/ScienceQA", split=f"train[:{max_size}]")
                logger.info(f"✅ Loaded {len(dataset)} examples after cache clear")
                return dataset
        except Exception as cache_error:
            logger.warning(f"Cache clearing failed: {cache_error}")
        
        # Create a minimal fallback dataset
        logger.info("Creating fallback dataset...")
        return create_fallback_dataset()

def create_fallback_dataset() -> List[Dict]:
    """Create a minimal fallback dataset for testing."""
    fallback_data = [
        {
            "question": "What is photosynthesis?",
            "choices": ["Process plants use to make food", "Animal breathing", "Water cycle", "Rock formation"],
            "answer": 0,
            "subject": "Biology",
            "lecture": "Photosynthesis is the process by which plants convert light energy into chemical energy (glucose) using carbon dioxide and water."
        },
        {
            "question": "What is osmosis?",
            "choices": ["Movement of water through membrane", "Cell division", "Chemical reaction", "Energy production"],
            "answer": 0,
            "subject": "Biology", 
            "lecture": "Osmosis is the movement of water molecules through a semipermeable membrane from an area of lower solute concentration to higher solute concentration."
        },
        {
            "question": "What is the water cycle?",
            "choices": ["Evaporation and precipitation", "Plant growth", "Animal migration", "Rock weathering"],
            "answer": 0,
            "subject": "Earth Science",
            "lecture": "The water cycle involves evaporation of water, condensation into clouds, and precipitation back to Earth."
        }
    ]
    logger.info(f"✅ Created fallback dataset with {len(fallback_data)} examples")
    return fallback_data

def prepare_documents(data) -> List[Document]:
    """Convert dataset to documents with improved processing."""
    documents = []
    
    try:
        # Handle both dataset objects and lists
        items = data if isinstance(data, list) else list(data)
        
        for i, item in enumerate(items):
            try:
                # More robust data extraction
                question = str(item.get("question", "")).strip()
                if not question:
                    continue
                
                choices = item.get("choices", [])
                answer_idx = item.get("answer", 0)
                
                # Safe answer extraction with better error handling
                if isinstance(choices, list) and len(choices) > 0:
                    if isinstance(answer_idx, int) and 0 <= answer_idx < len(choices):
                        answer = str(choices[answer_idx]).strip()
                    else:
                        answer = str(choices[0]).strip()  # Default to first choice
                else:
                    answer = "Answer not available"
                
                subject = str(item.get("subject", "Science")).strip()
                
                # Combine multiple explanation fields
                explanation_fields = ["solution", "lecture", "hint", "explanation"]
                explanations = []
                for field in explanation_fields:
                    if field in item and item[field]:
                        explanations.append(str(item[field]).strip())
                
                explanation = " ".join(explanations) if explanations else "No explanation available."
                
                # Create more comprehensive content
                content_parts = [
                    f"Question: {question}",
                    f"Answer: {answer}",
                    f"Subject: {subject}",
                    f"Explanation: {explanation}"
                ]
                
                # Add choices if available
                if choices and len(choices) > 1:
                    choices_text = ", ".join([f"({chr(65+i)}) {choice}" for i, choice in enumerate(choices)])
                    content_parts.insert(2, f"Choices: {choices_text}")
                
                content = "\n".join(content_parts)
                
                metadata = {
                    "question": question,
                    "answer": answer,
                    "subject": subject,
                    "source": "ScienceQA",
                    "doc_id": i
                }
                
                documents.append(Document(page_content=content, metadata=metadata))
                
            except Exception as e:
                logger.warning(f"⚠️ Skipping item {i} due to error: {e}")
                continue
        
        logger.info(f"📄 Prepared {len(documents)} documents")
        return documents
        
    except Exception as e:
        logger.error(f"❌ Error preparing documents: {e}")
        return []

@lru_cache(maxsize=1)
def initialize_embeddings():
    """Initialize embeddings with better error handling."""
    try:
        logger.info(f"🔧 Initializing embeddings ({config.EMBEDDING_MODEL})...")
        
        # Configure embeddings with corrected parameters
        model_kwargs = {
            'device': get_device(),
            'trust_remote_code': False
        }
        
        # Fixed: Remove unsupported parameters
        embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs=model_kwargs
        )
        
        # Test the embeddings
        test_embedding = embeddings.embed_query("test")
        if len(test_embedding) == 0:
            raise ValueError("Embeddings test failed")
        
        logger.info("✅ Embeddings initialized and tested successfully")
        return embeddings
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize embeddings: {e}")
        raise RuntimeError(f"Embedding initialization failed: {e}")

def create_vector_store(documents: List[Document], embeddings):
    """Create FAISS vector store with better error handling."""
    if not documents:
        raise ValueError("No documents provided for vector store creation")
    
    try:
        logger.info("✂️ Splitting documents...")
        
        # Optimized text splitter with better parameters
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True
        )
        
        split_docs = text_splitter.split_documents(documents)
        logger.info(f"📄 Split into {len(split_docs)} chunks")
        
        if not split_docs:
            raise ValueError("No chunks created from documents")
        
        logger.info("🗄️ Creating vector database...")
        
        # Create vector store with error handling
        try:
            vectorstore = FAISS.from_documents(split_docs, embeddings)
        except Exception as e:
            logger.error(f"FAISS creation failed: {e}")
            # Try with a smaller batch
            logger.info("Trying batch processing...")
            batch_size = min(50, len(split_docs))
            vectorstore = FAISS.from_documents(split_docs[:batch_size], embeddings)
            
            # Add remaining documents in batches
            for i in range(batch_size, len(split_docs), batch_size):
                batch = split_docs[i:i + batch_size]
                batch_vs = FAISS.from_documents(batch, embeddings)
                vectorstore.merge_from(batch_vs)
        
        logger.info("✅ Vector database created successfully")
        return vectorstore
        
    except Exception as e:
        logger.error(f"❌ Failed to create vector store: {e}")
        raise RuntimeError(f"Vector store creation failed: {e}")

@lru_cache(maxsize=1)
def initialize_llm():
    """Initialize LLM with better configuration and error handling."""
    try:
        logger.info(f"🤖 Initializing LLM ({config.LLM_MODEL})...")
        
        # Simplified pipeline configuration to avoid compatibility issues
        device_id = 0 if get_device() == 'cuda' else -1
        
        try:
            # Try with basic configuration first
            generator = pipeline(
                "text2text-generation",
                model=config.LLM_MODEL,
                max_new_tokens=config.MAX_NEW_TOKENS,
                temperature=config.TEMPERATURE,
                do_sample=config.DO_SAMPLE,
                device=device_id,
                truncation=True
            )
        except Exception as e:
            logger.warning(f"Pipeline creation failed with advanced config: {e}")
            # Fallback with minimal config
            generator = pipeline(
                "text2text-generation",
                model=config.LLM_MODEL,
                max_length=200,
                device=device_id if get_device() == 'cuda' else -1
            )
        
        llm = HuggingFacePipeline(pipeline=generator)
        
        # Test the LLM
        test_result = llm("Test question: What is water made of?")
        if not test_result:
            raise ValueError("LLM test failed")
        
        logger.info("✅ LLM initialized and tested successfully")
        return llm
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize LLM: {e}")
        raise RuntimeError(f"LLM initialization failed: {e}")

def setup_web_search_tool():
    """Setup enhanced web search tool with Tavily support."""
    if WEB_SEARCH_AVAILABLE:
        try:
            search_tool = create_web_search_tool(
                tavily_api_key=config.TAVILY_API_KEY
            )
            logger.info("🔍 Enhanced web search tool initialized!")
            return search_tool
        except Exception as e:
            logger.error(f"❌ Failed to setup web search tool: {e}")
            return None
    else:
        logger.warning("⚠️ Web search not available")
        return None

# Improved prompt template
PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "question", "web_search_results"],
    template="""You are an educational tutor specializing in science. Your goal is to provide clear, accurate, and helpful answers to student questions. You have access to a knowledge base and, if necessary, web search results.

Context Information from Knowledge Base:
{context}

Web Search Results (if available):
{web_search_results}

Student Question: {question}

Instructions:
- First, try to answer the question using the 'Context Information from Knowledge Base'.
- If the knowledge base context is insufficient or outdated, use the 'Web Search Results' to supplement your answer.
- If both sources are insufficient, state that you don't have enough information but still provide a general scientific explanation if possible.
- Provide a clear, detailed explanation, breaking down complex concepts.
- Use scientific terminology appropriately but explain complex terms in simple language.
- Focus on educational value, accuracy, and completeness.
- Always cite your sources, indicating whether information came from the 'Knowledge Base' or 'Web Search Results'.

Answer:"""
)

def setup_educational_agent():
    """Setup the complete educational agent with comprehensive error handling."""
    try:
        logger.info("🚀 Setting up Educational Tutor Agent...")
        
        # Step 1: Load dataset
        logger.info("📚 Step 1: Loading knowledge base...")
        dataset = load_scienceqa_dataset()
        
        # Step 2: Prepare documents
        logger.info("📄 Step 2: Processing documents...")
        documents = prepare_documents(dataset)
        
        if not documents:
            raise ValueError("No documents were prepared from the dataset")
        
        # Step 3: Initialize embeddings
        logger.info("🔧 Step 3: Initializing embeddings...")
        embeddings = initialize_embeddings()
        
        # Step 4: Create vector store
        logger.info("🗄️ Step 4: Creating knowledge base...")
        vectorstore = create_vector_store(documents, embeddings)
        
        # Step 5: Initialize LLM
        logger.info("🤖 Step 5: Initializing language model...")
        llm = initialize_llm()
        
        # Step 6: Create retriever
        logger.info("🔍 Step 6: Setting up retrieval system...")
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": config.TOP_K_RETRIEVAL}
        )
        
        # Step 7: Setup memory
        logger.info("🧠 Step 7: Configuring memory...")
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="question",
            output_key="answer",
            return_messages=True,
            max_token_limit=1000  # Limit memory size
        )
        
        # Step 8: Create QA chain
        logger.info("🔗 Step 8: Creating question-answering chain...")
        
        # Create a custom chain that incorporates web search
        class CustomRetrievalQA(ConversationalRetrievalChain):
            web_search_tool: Optional[Any] = None

            def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
                question = inputs["question"]
                chat_history = self.memory.load_memory_variables({})[self.memory.memory_key]

                # Retrieve documents from vectorstore
                docs = self.retriever.get_relevant_documents(question)
                context = "\n\n".join([doc.page_content for doc in docs])

                web_search_results = "No web search performed." # Default value
                if self.web_search_tool:
                    logger.info(f"Performing web search for: {question}")
                    try:
                        web_results = self.web_search_tool._run(question)
                        web_search_results = f"Web Search Results:\n{web_results}"
                    except Exception as e:
                        logger.warning(f"Web search failed: {e}")
                        web_search_results = f"Web search failed: {e}"

                # Prepare inputs for the LLM
                llm_inputs = {
                    "question": question,
                    "context": context,
                    "web_search_results": web_search_results,
                    "chat_history": chat_history # Pass chat history to the prompt
                }

                # Generate answer using the LLM
                answer = self.combine_docs_chain.run(llm_inputs)

                # Save to memory
                self.memory.save_context({"question": question}, {"answer": answer})

                return {"answer": answer, "source_documents": docs}

        qa_chain = CustomRetrievalQA(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": PROMPT_TEMPLATE},
            verbose=False,
            max_tokens_limit=2000,
            get_chat_history=lambda h: h,
            web_search_tool=setup_web_search_tool() # Pass the web search tool here
        )
        
        logger.info("✅ Educational Tutor Agent setup complete!")
        return qa_chain
        
    except Exception as e:
        logger.error(f"❌ Failed to setup Educational Tutor Agent: {e}")
        raise RuntimeError(f"Agent setup failed: {e}")

# Main execution guard
if __name__ == "__main__":
    try:
        logger.info("🧪 Testing Educational Tutor Agent...")
        agent = setup_educational_agent()
        
        # Test query
        test_query = "What is photosynthesis?"
        result = agent({"question": test_query})
        print(f"\nTest Query: {test_query}")
        print(f"Answer: {result['answer']}")
        print("✅ Agent test successful!")
        
    except Exception as e:
        logger.error(f"❌ Agent test failed: {e}")
        print(f"Error: {e}")
