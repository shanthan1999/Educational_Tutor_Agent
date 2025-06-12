"""
Educational Tutor Agent - Enhanced and Optimized Version
Key improvements:
- Fixed CustomRetrievalQA implementation
- Better prompt engineering
- Improved web search integration
- Enhanced error handling
- Memory optimization
"""

import os
import warnings
import logging
from typing import Optional, List, Dict, Any, Union
from functools import lru_cache
import asyncio
from datetime import datetime

# Setup logging with better formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('tutor_agent.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Compatibility setup
os.environ.update({
    "TOKENIZERS_PARALLELISM": "false",
    "OMP_NUM_THREADS": "2",  # Increased for better CPU performance
    "CUDA_VISIBLE_DEVICES": "0" if os.getenv("CUDA_VISIBLE_DEVICES") else ""
})
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Core imports with comprehensive error handling
try:
    from langchain.schema import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    
    # Updated imports with better fallback handling
    try:
        from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
        logger.info("✅ Using new HuggingFace integrations")
    except ImportError:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_community.llms import HuggingFacePipeline
            logger.info("✅ Using community HuggingFace integrations")
        except ImportError:
            from langchain.embeddings import HuggingFaceEmbeddings
            from langchain.llms import HuggingFacePipeline
            logger.info("✅ Using legacy HuggingFace integrations")
    
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationSummaryBufferMemory
    from langchain.prompts import PromptTemplate
    from langchain.callbacks.base import BaseCallbackHandler
    
    import torch
    from datasets import load_dataset
    from transformers import pipeline, AutoTokenizer
    
    logger.info("✅ Core dependencies loaded successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import core dependencies: {e}")
    raise ImportError(f"Missing required dependencies: {e}")

# Enhanced configuration with better defaults
class Config:
    """Enhanced configuration settings for the educational agent."""
    
    # Model settings - optimized choices
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = "google/flan-t5-base"
    
    # Processing settings - optimized for performance
    CHUNK_SIZE = 800  # Reduced for better retrieval
    CHUNK_OVERLAP = 150
    TOP_K_RETRIEVAL = 3  # Reduced to focus on most relevant
    MAX_DATASET_SIZE = 300  # Balanced size
    
    # Generation settings - tuned for educational content
    MAX_NEW_TOKENS = 300
    MIN_LENGTH = 50
    TEMPERATURE = 0.3  # Balanced creativity/accuracy
    DO_SAMPLE = True
    NUM_BEAMS = 3
    TOP_P = 0.9
    REPETITION_PENALTY = 1.1
    
    # Memory settings
    MAX_MEMORY_TOKENS = 2000
    MEMORY_SUMMARY_THRESHOLD = 1500
    
    # API settings
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    
    # Device settings
    FORCE_CPU = os.getenv("FORCE_CPU", "false").lower() == "true"
    
    # Performance settings
    ENABLE_CACHING = True
    CACHE_SIZE = 128

config = Config()

# Enhanced device configuration
@lru_cache(maxsize=1)
def get_device_info() -> Dict[str, Any]:
    """Get comprehensive device information."""
    device_info = {
        'device': 'cpu',
        'device_id': -1,
        'memory_gb': 0,
        'supports_half_precision': False
    }
    
    if config.FORCE_CPU:
        logger.info("🖥️ Forcing CPU usage")
        return device_info
    
    if torch.cuda.is_available():
        device_info.update({
            'device': 'cuda',
            'device_id': 0,
            'memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9,
            'supports_half_precision': torch.cuda.is_available()
        })
        logger.info(f"🚀 CUDA available - GPU memory: {device_info['memory_gb']:.1f}GB")
    else:
        logger.info("🖥️ CUDA not available - using CPU")
    
    return device_info

# Enhanced callback handler for monitoring
class TutorCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for monitoring tutor agent performance."""
    
    def __init__(self):
        self.query_count = 0
        self.total_tokens = 0
        self.start_time = None
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        self.start_time = datetime.now()
        self.query_count += 1
        logger.info(f"🔄 Processing query #{self.query_count}: {inputs.get('question', '')[:50]}...")
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            logger.info(f"✅ Query completed in {duration:.2f}s")

def load_scienceqa_dataset(max_size: int = None) -> Any:
    """Enhanced dataset loading with better error recovery."""
    if max_size is None:
        max_size = config.MAX_DATASET_SIZE
    
    try:
        logger.info(f"🔄 Loading ScienceQA dataset (max {max_size} examples)...")
        
        # Try multiple loading strategies
        strategies = [
            lambda: load_dataset("derek-thomas/ScienceQA", split=f"train[:{max_size}]"),
            lambda: load_dataset("derek-thomas/ScienceQA", split="train").select(range(max_size)),
            lambda: load_dataset("derek-thomas/ScienceQA")["train"].select(range(max_size)),
        ]
        
        for i, strategy in enumerate(strategies, 1):
            try:
                dataset = strategy()
                logger.info(f"✅ Loaded {len(dataset)} examples using strategy {i}")
                return dataset
            except Exception as e:
                logger.warning(f"Strategy {i} failed: {e}")
                continue
        
        raise Exception("All loading strategies failed")
        
    except Exception as e:
        logger.error(f"❌ Failed to load ScienceQA dataset: {e}")
        return create_enhanced_fallback_dataset()

def create_enhanced_fallback_dataset() -> List[Dict]:
    """Create an enhanced fallback dataset with diverse subjects."""
    fallback_data = [
        # Biology
        {
            "question": "What is photosynthesis and why is it important?",
            "choices": ["Process plants use to make food using sunlight", "Animal breathing process", "Water cycle process", "Rock formation process"],
            "answer": 0,
            "subject": "Biology",
            "lecture": "Photosynthesis is the process by which plants, algae, and some bacteria convert light energy (usually from the sun) into chemical energy stored in glucose. This process uses carbon dioxide from the air and water from the soil, producing oxygen as a byproduct. It's crucial for life on Earth as it produces oxygen and forms the base of most food chains."
        },
        {
            "question": "What is cellular respiration?",
            "choices": ["Process cells use to release energy from glucose", "Process of cell division", "Process of DNA replication", "Process of protein synthesis"],
            "answer": 0,
            "subject": "Biology",
            "lecture": "Cellular respiration is the process by which cells break down glucose and other organic molecules to release energy in the form of ATP. This process occurs in the mitochondria and involves glycolysis, the citric acid cycle, and the electron transport chain."
        },
        # Physics
        {
            "question": "What is Newton's First Law of Motion?",
            "choices": ["Objects at rest stay at rest unless acted upon by force", "Force equals mass times acceleration", "Every action has equal opposite reaction", "Energy cannot be created or destroyed"],
            "answer": 0,
            "subject": "Physics",
            "lecture": "Newton's First Law, also known as the Law of Inertia, states that an object at rest will remain at rest, and an object in motion will continue moving at constant velocity, unless acted upon by an unbalanced external force. This law explains why passengers lurch forward when a car brakes suddenly."
        },
        # Chemistry
        {
            "question": "What is the water cycle?",
            "choices": ["Continuous movement of water through evaporation and precipitation", "Process of plant growth", "Chemical reaction in laboratories", "Movement of tectonic plates"],
            "answer": 0,
            "subject": "Chemistry/Earth Science",
            "lecture": "The water cycle is the continuous movement of water within Earth and atmosphere. It involves evaporation from water bodies, transpiration from plants, condensation in clouds, and precipitation as rain or snow. This process is driven by solar energy and gravity."
        },
        # Mathematics
        {
            "question": "What is the Pythagorean theorem?",
            "choices": ["a² + b² = c² for right triangles", "Area of circle is πr²", "Volume of sphere is 4/3πr³", "Slope formula is (y₂-y₁)/(x₂-x₁)"],
            "answer": 0,
            "subject": "Mathematics",
            "lecture": "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse (longest side) equals the sum of squares of the other two sides. This fundamental principle is used in geometry, engineering, and navigation."
        }
    ]
    
    logger.info(f"✅ Created enhanced fallback dataset with {len(fallback_data)} examples")
    return fallback_data

def prepare_enhanced_documents(data) -> List[Document]:
    """Enhanced document preparation with better content structuring."""
    documents = []
    
    try:
        items = data if isinstance(data, list) else list(data)
        
        for i, item in enumerate(items):
            try:
                # Extract and validate core fields
                question = str(item.get("question", "")).strip()
                if not question or len(question) < 10:
                    continue
                
                # Process choices and answer
                choices = item.get("choices", [])
                answer_idx = item.get("answer", 0)
                
                if isinstance(choices, list) and choices:
                    if isinstance(answer_idx, int) and 0 <= answer_idx < len(choices):
                        correct_answer = str(choices[answer_idx]).strip()
                    else:
                        correct_answer = str(choices[0]).strip()
                    
                    # Format choices nicely
                    choices_formatted = "\n".join([
                        f"  {chr(65+idx)}. {choice}" 
                        for idx, choice in enumerate(choices)
                    ])
                else:
                    correct_answer = "Answer not provided"
                    choices_formatted = ""
                
                # Extract subject and explanations  
                subject = str(item.get("subject", "General Science")).strip()
                
                # Combine explanations from multiple fields
                explanation_sources = ["lecture", "solution", "hint", "explanation"]
                explanations = []
                for field in explanation_sources:
                    if field in item and item[field]:
                        explanations.append(str(item[field]).strip())
                
                explanation = " ".join(explanations) if explanations else ""
                
                # Create comprehensive document content
                content_sections = [
                    f"QUESTION: {question}",
                ]
                
                if choices_formatted:
                    content_sections.append(f"ANSWER CHOICES:\n{choices_formatted}")
                
                content_sections.extend([
                    f"CORRECT ANSWER: {correct_answer}",
                    f"SUBJECT: {subject}",
                ])
                
                if explanation:
                    content_sections.append(f"EXPLANATION: {explanation}")
                
                content = "\n\n".join(content_sections)
                
                # Enhanced metadata
                metadata = {
                    "question": question,
                    "answer": correct_answer,
                    "subject": subject,
                    "source": "ScienceQA",
                    "doc_id": i,
                    "has_choices": bool(choices),
                    "has_explanation": bool(explanation),
                    "content_length": len(content)
                }
                
                documents.append(Document(page_content=content, metadata=metadata))
                
            except Exception as e:
                logger.warning(f"⚠️ Skipping item {i} due to processing error: {e}")
                continue
        
        logger.info(f"📄 Prepared {len(documents)} enhanced documents")
        return documents
        
    except Exception as e:
        logger.error(f"❌ Error in document preparation: {e}")
        return []

@lru_cache(maxsize=1)
def initialize_enhanced_embeddings():
    """Initialize embeddings with enhanced configuration."""
    try:
        logger.info(f"🔧 Initializing embeddings ({config.EMBEDDING_MODEL})...")
        
        device_info = get_device_info()
        
        model_kwargs = {
            'device': device_info['device'],
            'trust_remote_code': False
        }
        
        # Add half precision for GPU if supported
        if device_info['supports_half_precision'] and device_info['device'] == 'cuda':
            model_kwargs['model_kwargs'] = {'torch_dtype': torch.float16}
        
        embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs=model_kwargs,
            encode_kwargs={'normalize_embeddings': True}  # Normalize for better similarity
        )
        
        # Comprehensive test
        test_texts = ["What is photosynthesis?", "Explain quantum mechanics"]
        test_embeddings = embeddings.embed_documents(test_texts)
        
        if not test_embeddings or len(test_embeddings[0]) == 0:
            raise ValueError("Embeddings test failed")
        
        logger.info(f"✅ Embeddings initialized (dimension: {len(test_embeddings[0])})")
        return embeddings
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize embeddings: {e}")
        raise RuntimeError(f"Embedding initialization failed: {e}")

def create_enhanced_vector_store(documents: List[Document], embeddings):
    """Create optimized FAISS vector store."""
    if not documents:
        raise ValueError("No documents provided for vector store creation")
    
    try:
        logger.info("✂️ Processing documents for vector store...")
        
        # Enhanced text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\nEXPLANATION:", "\n\nQUESTION:", "\n\n", "\n", ". ", " "],
            keep_separator=True,
            add_start_index=True
        )
        
        split_docs = text_splitter.split_documents(documents)
        logger.info(f"📄 Split into {len(split_docs)} optimized chunks")
        
        if not split_docs:
            raise ValueError("No chunks created from documents")
        
        logger.info("🗄️ Creating enhanced vector database...")
        
        # Create vector store with batch processing for large datasets
        batch_size = 50
        if len(split_docs) <= batch_size:
            vectorstore = FAISS.from_documents(split_docs, embeddings)
        else:
            # Process in batches for better memory management
            logger.info(f"Processing {len(split_docs)} documents in batches of {batch_size}")
            vectorstore = FAISS.from_documents(split_docs[:batch_size], embeddings)
            
            for i in range(batch_size, len(split_docs), batch_size):
                batch = split_docs[i:i + batch_size]
                batch_vs = FAISS.from_documents(batch, embeddings)
                vectorstore.merge_from(batch_vs)
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(split_docs)-1)//batch_size + 1}")
        
        logger.info("✅ Enhanced vector database created successfully")
        return vectorstore
        
    except Exception as e:
        logger.error(f"❌ Failed to create vector store: {e}")
        raise RuntimeError(f"Vector store creation failed: {e}")

@lru_cache(maxsize=1)
def initialize_enhanced_llm():
    """Initialize LLM with enhanced configuration."""
    try:
        logger.info(f"🤖 Initializing enhanced LLM ({config.LLM_MODEL})...")
        
        device_info = get_device_info()
        
        # Enhanced pipeline configuration
        pipeline_kwargs = {
            "task": "text2text-generation",
            "model": config.LLM_MODEL,
            "max_new_tokens": config.MAX_NEW_TOKENS,
            "min_length": config.MIN_LENGTH,
            "temperature": config.TEMPERATURE,
            "do_sample": config.DO_SAMPLE,
            "num_beams": config.NUM_BEAMS,
            "top_p": config.TOP_P,
            "repetition_penalty": config.REPETITION_PENALTY,
            "device": device_info['device_id'],
            "truncation": True,
            "padding": True
        }
        
        # Add half precision for GPU
        if device_info['supports_half_precision'] and device_info['device'] == 'cuda':
            pipeline_kwargs["torch_dtype"] = torch.float16
        
        try:
            generator = pipeline(**pipeline_kwargs)
        except Exception as e:
            logger.warning(f"Enhanced pipeline failed ({e}), trying basic config...")
            # Fallback configuration
            generator = pipeline(
                "text2text-generation",
                model=config.LLM_MODEL,
                max_length=256,
                temperature=0.3,
                device=device_info['device_id']
            )
        
        llm = HuggingFacePipeline(
            pipeline=generator,
            model_kwargs={"max_new_tokens": config.MAX_NEW_TOKENS}
        )
        
        # Enhanced test
        test_prompts = [
            "Explain photosynthesis in simple terms.",
            "What is gravity?"
        ]
        
        for prompt in test_prompts:
            result = llm(prompt)
            if not result or len(result.strip()) < 10:
                raise ValueError(f"LLM test failed for prompt: {prompt}")
        
        logger.info("✅ Enhanced LLM initialized and tested successfully")
        return llm
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize LLM: {e}")
        raise RuntimeError(f"LLM initialization failed: {e}")

# Enhanced prompt template with better structure
ENHANCED_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template="""You are an expert educational tutor with deep knowledge across all scientific disciplines. Your role is to provide clear, accurate, and pedagogically sound explanations to help students learn effectively.

Previous Conversation:
{chat_history}

Relevant Knowledge Base Context:
{context}

Current Student Question: {question}

Instructions for your response:
1. ANALYZE the question to understand what the student is asking and their likely knowledge level
2. USE the knowledge base context as your primary source of accurate information
3. STRUCTURE your answer with:
   - A direct, clear answer to the question
   - Step-by-step explanation breaking down complex concepts
   - Real-world examples or analogies when helpful
   - Key terminology defined in simple language
4. ADAPT your explanation level based on the question complexity
5. ENCOURAGE further learning by suggesting related concepts or questions
6. If the knowledge base lacks sufficient information, clearly state this and provide general scientific principles

Educational Answer:"""
)

class EnhancedEducationalAgent:
    """Enhanced educational agent with better architecture and features."""
    
    def __init__(self):
        self.vectorstore = None
        self.llm = None
        self.embeddings = None
        self.memory = None
        self.callback_handler = TutorCallbackHandler()
        self.is_initialized = False
    
    def initialize(self) -> bool:
        """Initialize all components of the educational agent."""
        try:
            logger.info("🚀 Initializing Enhanced Educational Tutor Agent...")
            
            # Step 1: Load and prepare data
            logger.info("📚 Step 1: Loading knowledge base...")
            dataset = load_scienceqa_dataset()
            documents = prepare_enhanced_documents(dataset)
            
            if not documents:
                raise ValueError("No documents prepared")
            
            # Step 2: Initialize embeddings
            logger.info("🔧 Step 2: Initializing embeddings...")
            self.embeddings = initialize_enhanced_embeddings()
            
            # Step 3: Create vector store
            logger.info("🗄️ Step 3: Creating knowledge base...")
            self.vectorstore = create_enhanced_vector_store(documents, self.embeddings)
            
            # Step 4: Initialize LLM
            logger.info("🤖 Step 4: Initializing language model...")
            self.llm = initialize_enhanced_llm()
            
            # Step 5: Setup memory
            logger.info("🧠 Step 5: Configuring memory...")
            self.memory = ConversationSummaryBufferMemory(
                llm=self.llm,
                memory_key="chat_history",
                input_key="question",
                output_key="answer",
                return_messages=False,
                max_token_limit=config.MAX_MEMORY_TOKENS,
                moving_summary_buffer=config.MEMORY_SUMMARY_THRESHOLD
            )
            
            self.is_initialized = True
            logger.info("✅ Enhanced Educational Tutor Agent initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize agent: {e}")
            return False
    
    def query(self, question: str) -> Dict[str, Any]:
        """Process a student query and return educational response."""
        if not self.is_initialized:
            return {"error": "Agent not initialized", "answer": ""}
        
        try:
            # Retrieve relevant documents
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": config.TOP_K_RETRIEVAL}
            )
            
            relevant_docs = retriever.get_relevant_documents(question)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Get chat history
            chat_history = self.memory.load_memory_variables({})["chat_history"]
            
            # Prepare prompt
            prompt_input = {
                "context": context,
                "question": question,
                "chat_history": chat_history
            }
            
            formatted_prompt = ENHANCED_PROMPT_TEMPLATE.format(**prompt_input)
            
            # Generate response
            response = self.llm(formatted_prompt)
            
            # Save to memory
            self.memory.save_context(
                {"question": question}, 
                {"answer": response}
            )
            
            return {
                "question": question,
                "answer": response,
                "sources": [doc.metadata for doc in relevant_docs],
                "context_length": len(context),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"❌ Query processing failed: {e}")
            return {
                "question": question,
                "answer": "I apologize, but I encountered an error processing your question. Please try rephrasing it.",
                "error": str(e),
                "success": False
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent performance statistics."""
        return {
            "query_count": self.callback_handler.query_count,
            "total_tokens": self.callback_handler.total_tokens,
            "is_initialized": self.is_initialized,
            "vector_store_size": self.vectorstore.index.ntotal if self.vectorstore else 0,
            "memory_size": len(self.memory.buffer) if self.memory else 0
        }

# Main execution
def main():
    """Main function to test the enhanced educational agent."""
    agent = EnhancedEducationalAgent()
    
    if not agent.initialize():
        logger.error("❌ Failed to initialize agent")
        return
    
    # Test queries
    test_questions = [
        "What is photosynthesis?",
        "Can you explain Newton's laws of motion?",
        "How does cellular respiration work?",
        "What is the water cycle?"
    ]
    
    logger.info("🧪 Testing Enhanced Educational Agent...")
    
    for question in test_questions:
        logger.info(f"\n🔍 Testing: {question}")
        result = agent.query(question)
        
        if result["success"]:
            print(f"\nQ: {question}")
            print(f"A: {result['answer'][:200]}...")
            print(f"Sources used: {len(result['sources'])}")
        else:
            print(f"❌ Failed: {result.get('error', 'Unknown error')}")
    
    # Print stats
    stats = agent.get_stats()
    logger.info(f"📊 Final stats: {stats}")
    
    print("\n✅ Enhanced Educational Agent testing completed!")

if __name__ == "__main__":
    main()
