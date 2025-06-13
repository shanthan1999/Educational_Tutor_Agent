"""
Enhanced Web Search Module with Tavily Integration
"""

import os
import logging
from typing import Optional, Dict, Any, List
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun

# Environment variable loading - compatible with Streamlit Cloud
def load_env_safely():
    """Load environment variables safely for both local and cloud environments."""
    try:
        # Try dotenv for local development
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        # dotenv not available in cloud environments - that's fine
        pass
    except Exception:
        # Any other error loading dotenv - continue without it
        pass

load_env_safely()

logger = logging.getLogger(__name__)

def get_tavily_api_key() -> Optional[str]:
    """Get Tavily API key from multiple sources with cloud compatibility."""
    # Try multiple environment variable sources
    api_key = None
    
    # Method 1: Direct os.environ (works in Streamlit Cloud)
    api_key = os.environ.get("TAVILY_API_KEY")
    if api_key and api_key.strip() and not api_key.startswith("your_"):
        return api_key.strip()
    
    # Method 2: os.getenv (fallback)
    api_key = os.getenv("TAVILY_API_KEY")
    if api_key and api_key.strip() and not api_key.startswith("your_"):
        return api_key.strip()
    
    # Method 3: Check Streamlit secrets (if available)
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and 'TAVILY_API_KEY' in st.secrets:
            api_key = st.secrets['TAVILY_API_KEY']
            if api_key and api_key.strip() and not api_key.startswith("your_"):
                return api_key.strip()
    except Exception:
        pass
    
    return None

class EnhancedWebSearchTool(BaseTool):
    """Enhanced web search tool with Tavily as primary search provider."""
    
    name: str = "web_search"
    description: str = "Search the web for current information and educational content using Tavily API."
    
    def __init__(self, tavily_api_key: Optional[str] = None):
        super().__init__()
        
        # Get API key with cloud compatibility
        final_api_key = tavily_api_key or get_tavily_api_key()
        
        # Store API key as instance variable (not Pydantic field)
        object.__setattr__(self, 'tavily_api_key', final_api_key)
        
        # Initialize search client
        object.__setattr__(self, 'tavily_client', None)
        
        # Log API key status for debugging
        if final_api_key:
            logger.info(f"🔑 Tavily API key found: {final_api_key[:10]}...")
        else:
            logger.warning("🔑 No Tavily API key found")
        
        self._setup_clients()
    
    def _setup_clients(self):
        """Initialize Tavily search client."""
        # Setup Tavily
        if self.tavily_api_key:
            try:
                from tavily import TavilyClient
                object.__setattr__(self, 'tavily_client', TavilyClient(api_key=self.tavily_api_key))
                logger.info("✅ Tavily search client initialized successfully")
                
                # Test the client with a simple call
                try:
                    test_response = self.tavily_client.search(query="test", max_results=1)
                    if test_response:
                        logger.info("✅ Tavily API test successful")
                    else:
                        logger.warning("⚠️ Tavily API test returned empty response")
                except Exception as test_error:
                    logger.error(f"❌ Tavily API test failed: {test_error}")
                    # Don't fail initialization, just log the error
                    
            except ImportError:
                logger.warning("⚠️ Tavily not available - install with: pip install tavily-python")
                object.__setattr__(self, 'tavily_client', None)
            except Exception as e:
                logger.error(f"❌ Tavily initialization failed: {e}")
                object.__setattr__(self, 'tavily_client', None)
        else:
            logger.info("ℹ️ No Tavily API key provided - using free search fallback")
        
        if not self.tavily_client:
            logger.info("🔄 Using free search fallback (DuckDuckGo + Wikipedia)")
    
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Execute web search using Tavily or free fallback."""
        try:
            # Try Tavily first
            if self.tavily_client:
                result = self._search_with_tavily(query)
                if result:
                    return result
            
            # Fallback to free search
            return self._free_search_fallback(query)
            
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return f"Web search encountered an error: {str(e)}"
    
    def _search_with_tavily(self, query: str) -> Optional[str]:
        """Search using Tavily API."""
        try:
            # Perform search with Tavily
            response = self.tavily_client.search(
                query=query,
                search_depth="basic",
                include_images=False,
                include_answer=True,
                max_results=3
            )
            
            if not response or 'results' not in response:
                return None
            
            # Format results
            formatted_results = []
            
            # Include answer if available
            if response.get('answer'):
                formatted_results.append(f"**Direct Answer**\n{response['answer']}\n")
            
            # Include search results
            for i, result in enumerate(response['results'][:3], 1):
                title = result.get('title', 'No Title')
                content = result.get('content', 'No content available')
                url = result.get('url', '')
                
                formatted_result = f"**Result {i}: {title}**\n{content}"
                if url:
                    formatted_result += f"\nSource: {url}"
                
                formatted_results.append(formatted_result)
            
            if formatted_results:
                return "\n\n".join(formatted_results)
            
        except Exception as e:
            logger.error(f"Tavily search error: {e}")
        
        return None
    
    def _free_search_fallback(self, query: str) -> str:
        """Fallback search using free APIs."""
        try:
            import requests
            
            # Try DuckDuckGo Instant Answer API
            duck_result = self._search_duckduckgo(query)
            if duck_result:
                return duck_result
            
            # Try Wikipedia
            wiki_result = self._search_wikipedia(query)
            if wiki_result:
                return wiki_result
            
            return f"Limited search results available for: {query}\n\nConsider adding a Tavily API key for enhanced web search capabilities."
            
        except Exception as e:
            logger.error(f"Free search error: {e}")
            return f"Web search temporarily unavailable. Error: {str(e)}"
    
    def _search_duckduckgo(self, query: str) -> Optional[str]:
        """Search using DuckDuckGo Instant Answer API."""
        try:
            import requests
            
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
                
                result_parts = []
                
                if data.get("Abstract"):
                    result_parts.append(f"**{data.get('Heading', 'Answer')}**\n{data['Abstract']}")
                    if data.get("AbstractURL"):
                        result_parts.append(f"Source: {data['AbstractURL']}")
                
                if data.get("Definition"):
                    result_parts.append(f"**Definition**\n{data['Definition']}")
                
                if result_parts:
                    return "\n\n".join(result_parts)
            
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
        
        return None
    
    def _search_wikipedia(self, query: str) -> Optional[str]:
        """Search Wikipedia for information."""
        try:
            import requests
            
            search_url = "https://en.wikipedia.org/api/rest_v1/page/summary/"
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
        
        return None

def create_web_search_tool(tavily_api_key: Optional[str] = None) -> EnhancedWebSearchTool:
    """Factory function to create web search tool."""
    return EnhancedWebSearchTool(tavily_api_key=tavily_api_key) 
