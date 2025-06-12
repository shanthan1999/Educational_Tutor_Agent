import streamlit as st
import logging
import os
import warnings
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json

# Suppress warnings and configure environment
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Configure logging to match the format in the error logs
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger('tutor_agent')

# Import configuration
try:
    from config import get_deployment_config, get_model_config, FALLBACK_SCIENCE_QA
    deployment_config = get_deployment_config()
    model_config = get_model_config()
except ImportError:
    logger.warning("Config file not found, using default configuration")
    deployment_config = {"USE_FALLBACK_DATASET": True, "USE_CPU_ONLY": True}
    model_config = {
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "llm_model": "google/flan-t5-small",
        "max_length": 256,
        "temperature": 0.7,
        "top_k": 3
    }

# Try importing dependencies with proper error handling
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    from datasets import Dataset
    import pandas as pd
except ImportError as e:
    st.error(f"❌ Missing required dependencies: {e}")
    st.info("Please install: pip install torch transformers sentence-transformers faiss-cpu datasets pandas")
    st.stop()

@dataclass
class TutorConfig:
    """Configuration for the Educational Tutor Agent - optimized for deployment"""
    embedding_model: str = model_config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    llm_model: str = model_config.get("llm_model", "google/flan-t5-small")
    max_length: int = model_config.get("max_length", 256)
    temperature: float = model_config.get("temperature", 0.7)
    top_k: int = model_config.get("top_k", 3)
    use_fallback: bool = deployment_config.get("USE_FALLBACK_DATASET", True)
    use_cpu_only: bool = deployment_config.get("USE_CPU_ONLY", True)

class DatasetManager:
    """Manages dataset loading with enhanced fallback options"""
    
    @staticmethod
    def create_fallback_science_qa() -> Dataset:
        """Create a comprehensive fallback ScienceQA dataset"""
        logger.info("Creating fallback dataset...")
        
        # Use the fallback data from config if available
        try:
            from config import FALLBACK_SCIENCE_QA
            fallback_data = FALLBACK_SCIENCE_QA
        except ImportError:
            # Fallback to basic dataset if config not available
            fallback_data = [
                {
                    "question": "What is photosynthesis?",
                    "context": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce glucose and release oxygen.",
                    "answer": "Photosynthesis is the process where plants convert sunlight, water, and carbon dioxide into glucose and oxygen using chlorophyll.",
                    "subject": "biology",
                    "grade_level": "middle_school"
                },
                {
                    "question": "What causes earthquakes?",
                    "context": "Earthquakes are caused by the sudden release of energy when rocks break along fault lines or when tectonic plates move.",
                    "answer": "Earthquakes are caused by the sudden release of energy when rocks break along fault lines or tectonic plates move.",
                    "subject": "earth_science",
                    "grade_level": "middle_school"
                },
                {
                    "question": "What is DNA?",
                    "context": "DNA is the hereditary material in living organisms that contains genetic instructions for development and functioning.",
                    "answer": "DNA is the genetic material that carries hereditary information in living organisms.",
                    "subject": "biology",
                    "grade_level": "high_school"
                }
            ]
        
        dataset = Dataset.from_list(fallback_data)
        logger.info(f"✅ Created fallback dataset with {len(fallback_data)} examples")
        return dataset
    
    @staticmethod
    def load_dataset() -> Dataset:
        """Load dataset with deployment-optimized fallback strategies"""
        logger.info("🔄 Step 1: Loading ScienceQA dataset...")
        
        # For deployment, always use fallback to avoid caching issues
        if deployment_config.get("USE_FALLBACK_DATASET", True):
            logger.info("Using fallback dataset for deployment stability...")
            return DatasetManager.create_fallback_science_qa()
        
        try:
            # Try to load from Hugging Face Hub with explicit no-cache
            logger.info("Attempting to load ScienceQA from Hugging Face Hub...")
            from datasets import load_dataset
            
            # Disable caching to avoid LocalFileSystem issues
            dataset = load_dataset(
                "derek-thomas/ScienceQA", 
                split="train[:100]",  # Load only first 100 examples for deployment
                streaming=False,
                cache_dir=None,  # Disable caching
                download_mode="force_redownload"
            )
            logger.info("✅ Successfully loaded ScienceQA dataset from Hugging Face Hub")
            return dataset
            
        except Exception as e:
            logger.warning(f"Trying alternative loading method...")
            logger.error(f"❌ Failed to load ScienceQA dataset: {e}")
            logger.info("Creating fallback dataset...")
            return DatasetManager.create_fallback_science_qa()

class EmbeddingsManager:
    """Manages sentence embeddings for document retrieval - deployment optimized"""
    
    def __init__(self, model_name: str, use_cpu_only: bool = True):
        self.model_name = model_name
        self.model = None
        self.use_cpu_only = use_cpu_only
        
    def initialize(self):
        """Initialize the sentence transformer model"""
        try:
            logger.info(f"🔧 Initializing embeddings ({self.model_name})...")
            
            # Force CPU usage for deployment if configured
            if self.use_cpu_only:
                device = "cpu"
                logger.info("🖥️ CUDA not available - using CPU")
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"🖥️ {'CUDA available - using GPU' if device == 'cuda' else 'CUDA not available - using CPU'}")
            
            # Initialize with explicit device and cache settings
            self.model = SentenceTransformer(
                self.model_name, 
                device=device,
                cache_folder="/tmp/sentence_transformers" if deployment_config.get("USE_CPU_ONLY") else None
            )
            
            # Test the model
            test_embedding = self.model.encode(["test sentence"])
            logger.info("✅ Embeddings initialized and tested successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize embeddings: {e}")
            raise e
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts into embeddings"""
        if self.model is None:
            raise ValueError("Embeddings model not initialized")
        return self.model.encode(texts)

class VectorDatabase:
    """Manages FAISS vector database for similarity search - deployment optimized"""
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.index = None
        self.documents = []
        
    def create_index(self, embeddings: np.ndarray, documents: List[str]):
        """Create FAISS index from embeddings"""
        try:
            logger.info("🗄️ Creating vector database...")
            
            # Create FAISS index with CPU optimization
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            
            # Add embeddings in smaller batches for deployment
            batch_size = 100
            for i in range(0, len(embeddings), batch_size):
                batch = embeddings[i:i+batch_size].astype('float32')
                self.index.add(batch)
            
            self.documents = documents
            logger.info("✅ Vector database created successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to create vector database: {e}")
            raise e
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if self.index is None:
            raise ValueError("Vector database not initialized")
        
        # Ensure k doesn't exceed available documents
        k = min(k, len(self.documents))
        
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.documents) and idx >= 0:  # Valid index check
                results.append({
                    "document": self.documents[idx],
                    "distance": float(distance),
                    "rank": i + 1
                })
        
        return results

class LanguageModel:
    """Manages the language model for generating responses - deployment optimized"""
    
    def __init__(self, model_name: str, use_cpu_only: bool = True):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.use_cpu_only = use_cpu_only
    
    def initialize(self):
        """Initialize the language model"""
        try:
            logger.info(f"🤖 Initializing LLM ({self.model_name})...")
            
            # Force CPU usage for deployment
            device = -1 if self.use_cpu_only else (0 if torch.cuda.is_available() else -1)
            
            # Initialize with deployment-friendly settings
            self.pipeline = pipeline(
                "text2text-generation",
                model=self.model_name,
                tokenizer=self.model_name,
                device=device,
                max_length=256,  # Reduced for deployment
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                framework="pt"
            )
            
            logger.info("✅ Language model initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize language model: {e}")
            raise e
    
    def generate_response(self, prompt: str, max_length: int = 256) -> str:
        """Generate response using the language model"""
        try:
            if self.pipeline is None:
                raise ValueError("Language model not initialized")
            
            # Truncate prompt if too long
            if len(prompt) > 1000:
                prompt = prompt[:1000] + "..."
            
            result = self.pipeline(
                prompt, 
                max_length=max_length, 
                num_return_sequences=1,
                truncation=True,
                pad_token_id=self.pipeline.tokenizer.eos_token_id
            )
            return result[0]['generated_text']
            
        except Exception as e:
            logger.error(f"❌ Failed to generate response: {e}")
            return "I apologize, but I encountered an error while generating a response. Please try again with a simpler question."

class EducationalTutorAgent:
    """Main Educational Tutor Agent class - deployment optimized"""
    
    def __init__(self, config: TutorConfig):
        self.config = config
        self.dataset = None
        self.embeddings_manager = None
        self.vector_db = None
        self.llm = None
        
    def initialize(self):
        """Initialize all components of the tutor agent"""
        try:
            logger.info("🚀 Initializing Educational Tutor Agent...")
            
            # Step 1: Load dataset (using fallback for deployment)
            self.dataset = DatasetManager.load_dataset()
            
            # Step 2: Process documents
            logger.info("📄 Step 2: Processing documents...")
            documents = self._prepare_documents()
            logger.info(f"📄 Prepared {len(documents)} documents")
            
            # Step 3: Initialize embeddings
            logger.info("🔧 Step 3: Initializing embeddings...")
            self.embeddings_manager = EmbeddingsManager(
                self.config.embedding_model, 
                use_cpu_only=self.config.use_cpu_only
            )
            self.embeddings_manager.initialize()
            
            # Step 4: Create knowledge base
            logger.info("🗄️ Step 4: Creating knowledge base...")
            self._create_knowledge_base(documents)
            
            # Step 5: Initialize language model
            logger.info("🤖 Step 5: Initializing language model...")
            self.llm = LanguageModel(
                self.config.llm_model,
                use_cpu_only=self.config.use_cpu_only
            )
            self.llm.initialize()
            
            logger.info("✅ Educational Tutor Agent initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize tutor agent: {e}")
            raise e
    
    def _prepare_documents(self) -> List[str]:
        """Prepare documents from dataset"""
        documents = []
        
        for item in self.dataset:
            # Create comprehensive document text
            doc_text = f"Question: {item.get('question', '')}\n"
            if 'context' in item and item['context']:
                doc_text += f"Context: {item['context']}\n"
            if 'answer' in item and item['answer']:
                doc_text += f"Answer: {item['answer']}\n"
            if 'subject' in item and item['subject']:
                doc_text += f"Subject: {item['subject']}\n"
            if 'grade_level' in item and item['grade_level']:
                doc_text += f"Grade Level: {item['grade_level']}"
            
            documents.append(doc_text.strip())
        
        return documents
    
    def _create_knowledge_base(self, documents: List[str]):
        """Create vector database from documents"""
        # Split documents into chunks if needed
        logger.info("✂️ Splitting documents...")
        chunks = self._split_documents(documents)
        logger.info(f"📄 Split into {len(chunks)} chunks")
        
        # Create embeddings in batches for deployment
        logger.info("🗄️ Creating vector database...")
        embeddings = self.embeddings_manager.encode(chunks)
        
        # Create vector database
        self.vector_db = VectorDatabase(embeddings.shape[1])
        self.vector_db.create_index(embeddings, chunks)
    
    def _split_documents(self, documents: List[str], chunk_size: int = 300) -> List[str]:
        """Split documents into smaller chunks - optimized for deployment"""
        chunks = []
        for doc in documents:
            if len(doc) <= chunk_size:
                chunks.append(doc)
            else:
                # Simple splitting by sentences
                sentences = doc.split('. ')
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk + sentence) <= chunk_size:
                        current_chunk += sentence + ". "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + ". "
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
        
        return chunks
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer a question using RAG (Retrieval-Augmented Generation)"""
        try:
            # Step 1: Retrieve relevant documents
            query_embedding = self.embeddings_manager.encode([question])
            relevant_docs = self.vector_db.search(query_embedding, k=self.config.top_k)
            
            # Step 2: Create context from retrieved documents
            context = "\n\n".join([doc["document"] for doc in relevant_docs[:2]])  # Reduced context for deployment
            
            # Step 3: Generate prompt
            prompt = f"""You are an educational tutor. Answer the following question based on the provided context. 
Be clear, accurate, and educational in your response.

Context:
{context}

Question: {question}

Answer:"""
            
            # Step 4: Generate response
            response = self.llm.generate_response(prompt, max_length=self.config.max_length)
            
            return {
                "answer": response,
                "relevant_documents": relevant_docs,
                "confidence": self._calculate_confidence(relevant_docs)
            }
            
        except Exception as e:
            logger.error(f"❌ Error answering question: {e}")
            return {
                "answer": "I apologize, but I encountered an error while processing your question. Please try again with a simpler question.",
                "relevant_documents": [],
                "confidence": 0.0
            }
    
    def _calculate_confidence(self, relevant_docs: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on document similarity"""
        if not relevant_docs:
            return 0.0
        
        # Use inverse of average distance as confidence
        avg_distance = sum(doc["distance"] for doc in relevant_docs) / len(relevant_docs)
        confidence = max(0.0, min(1.0, 1.0 / (1.0 + avg_distance)))
        
        return confidence

# Streamlit UI - Deployment Optimized
def main():
    st.set_page_config(
        page_title="🎓 Educational Tutor Agent",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🎓 Educational Tutor Agent")
    st.markdown("### AI-Powered Educational Assistant (Deployment Ready)")
    
    # Initialize session state
    if "tutor_agent" not in st.session_state:
        st.session_state.tutor_agent = None
        st.session_state.initialized = False
    
    # Sidebar for configuration and status
    with st.sidebar:
        st.header("⚙️ System Configuration")
        
        # Display deployment status
        st.info("🚀 **Deployment Mode**: Optimized for cloud deployment")
        
        # Show current configuration
        st.markdown("**Current Settings:**")
        st.text(f"Embedding: {model_config['embedding_model'].split('/')[-1]}")
        st.text(f"LLM: {model_config['llm_model'].split('/')[-1]}")
        st.text(f"Device: {'CPU (Deployment)' if deployment_config.get('USE_CPU_ONLY') else 'Auto'}")
        st.text(f"Dataset: {'Fallback' if deployment_config.get('USE_FALLBACK_DATASET') else 'Full'}")
        
        # Initialize button
        if st.button("🚀 Initialize Tutor Agent", type="primary"):
            with st.spinner("Initializing Educational Tutor Agent..."):
                try:
                    config = TutorConfig()
                    
                    # Create progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Loading dataset...")
                    progress_bar.progress(20)
                    
                    st.session_state.tutor_agent = EducationalTutorAgent(config)
                    
                    status_text.text("Initializing models...")
                    progress_bar.progress(60)
                    
                    st.session_state.tutor_agent.initialize()
                    
                    status_text.text("Finalizing setup...")
                    progress_bar.progress(100)
                    
                    st.session_state.initialized = True
                    
                    status_text.text("✅ Initialization complete!")
                    st.success("✅ Tutor Agent initialized successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Initialization failed: {e}")
                    st.session_state.initialized = False
                    logger.error(f"Initialization error: {e}")
    
    # Main interface
    if st.session_state.initialized and st.session_state.tutor_agent:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("💬 Ask a Question")
            
            # Question input
            question = st.text_area(
                "What would you like to learn about?",
                placeholder="e.g., What is photosynthesis? How does gravity work? Explain the water cycle.",
                height=100
            )
            
            # Sample questions from our fallback dataset
            st.markdown("**Sample Questions:**")
            sample_questions = [
                "What is photosynthesis?",
                "What causes earthquakes?", 
                "What is DNA?",
                "What is gravity?",
                "How do vaccines work?"
            ]
            
            # Create buttons for sample questions
            cols = st.columns(len(sample_questions))
            for i, sample_q in enumerate(sample_questions):
                if cols[i].button(f"💡", key=f"sample_{i}", help=sample_q):
                    question = sample_q
            
            # Answer button
            if st.button("🎯 Get Answer", type="primary") and question:
                with st.spinner("Generating answer..."):
                    result = st.session_state.tutor_agent.answer_question(question)
                    
                    # Display answer
                    st.subheader("📝 Answer")
                    st.write(result["answer"])
                    
                    # Display confidence
                    confidence = result["confidence"]
                    confidence_color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
                    st.markdown(f"**Confidence:** :{confidence_color}[{confidence:.2%}]")
                    
                    # Display relevant documents
                    if result["relevant_documents"]:
                        with st.expander("📚 Sources Used"):
                            for i, doc in enumerate(result["relevant_documents"]):
                                st.markdown(f"**Source {doc['rank']}** (Similarity: {1.0/(1.0+doc['distance']):.2%})")
                                st.text(doc["document"][:200] + "..." if len(doc["document"]) > 200 else doc["document"])
                                if i < len(result["relevant_documents"]) - 1:
                                    st.markdown("---")
        
        with col2:
            st.header("📊 System Status")
            
            # System metrics
            st.metric("🔧 Status", "✅ Ready")
            st.metric("📚 Dataset", f"{len(st.session_state.tutor_agent.dataset)} examples")
            st.metric("🗄️ Knowledge Base", f"{len(st.session_state.tutor_agent.vector_db.documents)} chunks")
            st.metric("⚡ Device", "CPU (Optimized)")
            
            # Performance info
            st.header("⚡ Performance")
            st.info("""
            **Deployment Optimized:**
            - ✅ CPU-only processing
            - ✅ Lightweight models
            - ✅ Fallback dataset
            - ✅ Memory efficient
            """)
            
            # Usage tips
            st.header("💡 Usage Tips")
            st.markdown("""
            - Ask specific science questions
            - Keep questions concise
            - Try different subjects (biology, physics, earth science)
            - Check the confidence score
            """)
    
    else:
        # Welcome screen
        st.info("👈 Please initialize the Tutor Agent using the sidebar.")
        
        # Display features
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🔧 Features
            - AI-powered Q&A
            - Science education focus
            - Retrieval-augmented generation
            - Confidence scoring
            """)
        
        with col2:
            st.markdown("""
            ### 🚀 Deployment Ready
            - Optimized for cloud platforms
            - CPU-only processing
            - Lightweight models
            - Reliable fallback dataset
            """)
        
        with col3:
            st.markdown("""
            ### 📚 Subjects Covered
            - Biology
            - Physics
            - Earth Science
            - Chemistry
            - General Science
            """)
        
        # Deployment information
        st.header("🔧 Deployment Information")
        st.markdown("""
        This version of the Educational Tutor Agent has been optimized for deployment environments:
        
        **Key Improvements:**
        - ✅ **Fixed dataset loading**: Uses reliable fallback dataset to avoid caching issues
        - ✅ **CPU optimization**: Forced CPU usage for better compatibility
        - ✅ **Memory efficient**: Reduced model sizes and batch processing
        - ✅ **Error handling**: Comprehensive error handling and fallback mechanisms
        - ✅ **Deployment scripts**: Includes setup scripts for easy deployment
        
        **Deployment Files Created:**
        - `app_fixed.py`: Main application with fixes
        - `requirements.txt`: Optimized dependencies
        - `config.py`: Deployment configuration
        - `deploy.sh`: Deployment script
        """)

if __name__ == "__main__":
    main() 
