"""
Reasoning and Summarization Module
Uses LLM (GPT-4o/Claude) for intelligent analysis and insights
"""
import logging
from typing import Dict, Any, List, Optional
import json

logger = logging.getLogger(__name__)


class ReasoningEngine:
    """LLM-powered reasoning and summarization"""
    
    def __init__(self, config):
        self.config = config
        self.model_config = config.models
        self.client = None
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize LLM client"""
        try:
            if self.model_config.reasoning_model.startswith("gpt"):
                from openai import OpenAI
                
                if not self.config.openai_api_key:
                    logger.warning("OpenAI API key not configured")
                    return
                
                self.client = OpenAI(api_key=self.config.openai_api_key)
                self.client_type = "openai"
                logger.info(f"OpenAI client initialized with model: {self.model_config.reasoning_model}")
                
            elif self.model_config.reasoning_model.startswith("claude"):
                from anthropic import Anthropic
                
                if not self.config.anthropic_api_key:
                    logger.warning("Anthropic API key not configured")
                    return
                
                self.client = Anthropic(api_key=self.config.anthropic_api_key)
                self.client_type = "anthropic"
                logger.info(f"Anthropic client initialized with model: {self.model_config.reasoning_model}")
            
        except Exception as e:
            logger.warning(f"LLM client initialization failed: {e}")
            logger.info("Reasoning features will be limited")
            self.client = None
    
    def analyze_document(self, structured_doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive document analysis
        
        Returns:
            Dict containing analysis results
        """
        try:
            if self.client is None:
                return self._fallback_analysis(structured_doc)
            
            # Prepare document content for analysis
            content = self._prepare_content(structured_doc)
            
            # Create analysis prompt
            prompt = self._create_analysis_prompt(content)
            
            # Get LLM response
            response = self._query_llm(prompt)
            
            # Parse response
            analysis = self._parse_analysis_response(response)
            
            return {
                "summary": analysis.get("summary", ""),
                "key_insights": analysis.get("key_insights", []),
                "action_items": analysis.get("action_items", []),
                "risk_highlights": analysis.get("risk_highlights", []),
                "routing_suggestions": analysis.get("routing_suggestions", {}),
                "departmental_relevance": analysis.get("departmental_relevance", {}),
                "metadata": {
                    "model": self.model_config.reasoning_model,
                    "confidence": analysis.get("confidence", 0.8)
                },
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Document analysis failed: {e}")
            return {
                "summary": "",
                "success": False,
                "error": str(e)
            }
    
    def _prepare_content(self, structured_doc: Dict[str, Any]) -> str:
        """Prepare document content for LLM analysis"""
        content_parts = []
        
        # Add document info
        doc_info = structured_doc.get("document_info", {})
        content_parts.append(f"Document: {doc_info.get('file_name', 'Unknown')}")
        content_parts.append(f"Type: {doc_info.get('file_type', 'Unknown')}")
        content_parts.append("")
        
        # Add main text content
        full_text = structured_doc.get("content", {}).get("full_text", "")
        if full_text:
            content_parts.append("Content:")
            content_parts.append(full_text[:5000])  # Limit to first 5000 chars
            if len(full_text) > 5000:
                content_parts.append("\n[Content truncated...]")
            content_parts.append("")
        
        # Add structured elements summary
        elements = structured_doc.get("content", {}).get("structured_elements", [])
        if elements:
            content_parts.append(f"Document contains {len(elements)} structured elements:")
            
            element_counts = {}
            for elem in elements:
                elem_type = elem.get("type", "unknown")
                element_counts[elem_type] = element_counts.get(elem_type, 0) + 1
            
            for elem_type, count in element_counts.items():
                content_parts.append(f"- {elem_type}: {count}")
            content_parts.append("")
        
        # Add layout info
        layout = structured_doc.get("layout", {})
        if layout.get("total_regions", 0) > 0:
            content_parts.append(f"Layout: {layout['total_regions']} regions detected")
            content_parts.append("")
        
        return "\n".join(content_parts)
    
    def _create_analysis_prompt(self, content: str) -> str:
        """Create comprehensive analysis prompt"""
        prompt = f"""Analyze the following document and provide comprehensive insights:

{content}

Please provide a detailed analysis in the following JSON format:
{{
  "summary": "A concise 2-3 sentence summary of the document",
  "key_insights": ["Insight 1", "Insight 2", "Insight 3"],
  "action_items": ["Action item 1", "Action item 2"],
  "risk_highlights": ["Risk 1", "Risk 2"],
  "routing_suggestions": {{
    "primary_department": "Department name",
    "secondary_departments": ["Dept 1", "Dept 2"],
    "priority": "high/medium/low",
    "urgency": "immediate/routine"
  }},
  "departmental_relevance": {{
    "Finance": 0.8,
    "Legal": 0.6,
    "Operations": 0.9
  }},
  "confidence": 0.85
}}

Respond with only the JSON object, no additional text."""
        
        return prompt
    
    def _query_llm(self, prompt: str) -> str:
        """Query the LLM with the prompt"""
        try:
            if self.client_type == "openai":
                response = self.client.chat.completions.create(
                    model=self.model_config.reasoning_model,
                    messages=[
                        {"role": "system", "content": "You are an expert document analyst providing structured insights."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=2000
                )
                return response.choices[0].message.content
                
            elif self.client_type == "anthropic":
                response = self.client.messages.create(
                    model=self.model_config.reasoning_model,
                    max_tokens=2000,
                    temperature=0.3,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.content[0].text
            
        except Exception as e:
            logger.error(f"LLM query failed: {e}")
            raise
    
    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured format"""
        try:
            # Try to extract JSON from response
            # LLM might include markdown code blocks
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            
            # Parse JSON
            analysis = json.loads(response)
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            # Return basic structure if parsing fails
            return {
                "summary": response[:500] if response else "",
                "key_insights": [],
                "action_items": [],
                "risk_highlights": [],
                "routing_suggestions": {},
                "departmental_relevance": {},
                "confidence": 0.5
            }
    
    def _fallback_analysis(self, structured_doc: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback analysis when LLM is not available"""
        doc_info = structured_doc.get("document_info", {})
        content = structured_doc.get("content", {})
        
        # Basic summary
        full_text = content.get("full_text", "")
        word_count = len(full_text.split())
        
        summary = f"Document '{doc_info.get('file_name', 'Unknown')}' contains approximately {word_count} words. "
        
        elements = content.get("structured_elements", [])
        if elements:
            element_types = set(e.get("type", "unknown") for e in elements)
            summary += f"Includes {len(elements)} structured elements: {', '.join(element_types)}."
        
        return {
            "summary": summary,
            "key_insights": ["Detailed analysis requires LLM configuration"],
            "action_items": [],
            "risk_highlights": [],
            "routing_suggestions": {
                "primary_department": "Unknown",
                "priority": "medium"
            },
            "departmental_relevance": {},
            "metadata": {
                "model": "fallback",
                "confidence": 0.3
            },
            "success": True,
            "fallback": True
        }
    
    def generate_summary(self, text: str, max_length: int = 200) -> str:
        """
        Generate concise summary of text
        
        Args:
            text: Input text to summarize
            max_length: Maximum length of summary in words
            
        Returns:
            Summary string
        """
        try:
            if self.client is None:
                # Simple fallback: return first few sentences
                sentences = text.split(". ")
                return ". ".join(sentences[:3]) + "."
            
            prompt = f"Summarize the following text in no more than {max_length} words:\n\n{text[:2000]}"
            
            response = self._query_llm(prompt)
            return response.strip()
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return text[:500] + "..." if len(text) > 500 else text
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text
        
        Returns:
            Dict mapping entity types to lists of entities
        """
        try:
            if self.client is None:
                return {"entities": []}
            
            prompt = f"""Extract named entities from this text and categorize them:

{text[:2000]}

Return JSON format:
{{
  "people": ["Name 1", "Name 2"],
  "organizations": ["Org 1", "Org 2"],
  "locations": ["Location 1", "Location 2"],
  "dates": ["Date 1", "Date 2"],
  "amounts": ["Amount 1", "Amount 2"]
}}"""
            
            response = self._query_llm(prompt)
            entities = json.loads(response)
            return entities
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return {}
