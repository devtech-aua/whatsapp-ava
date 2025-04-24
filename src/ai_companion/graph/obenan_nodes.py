"""
Specialized agent nodes for the Obenan Multi-Agent Framework.
Each node represents a specialized agent with distinct responsibilities.
"""

import json
from uuid import uuid4
import os

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from ai_companion.graph.state import AICompanionState
from ai_companion.graph.utils.chains import get_character_response_chain
from ai_companion.graph.utils.helpers import (
    get_chat_model,
    get_text_to_image_module,
    get_text_to_speech_module,
)
from ai_companion.modules.memory.long_term.memory_manager import get_memory_manager


def format_messages(messages):
    """Format messages for router prompts."""
    formatted = []
    for msg in messages:
        role = "User" if msg.type == "human" else "Assistant"
        formatted.append(f"{role}: {msg.content}")
    return "\n".join(formatted)


def get_advanced_llm(model_name=None):
    """Get an advanced LLM model for semantic routing."""
    # For now, reuse the existing chat model function
    # In a real implementation, this could use specific models
    return get_chat_model(model_name)


async def get_semantic_router_chain():
    """Create a chain that uses advanced LLMs to extract intent and route to agents."""
    llm = get_advanced_llm()
    
    prompt = PromptTemplate.from_template(
        """You are OmniPulse (OBI-000), the central orchestration agent for a business management system.
        Your role is to analyze user messages and determine:
        1. The specialized agent that should handle this request
        2. The specific business intent and operation needed
        3. Structured parameters to pass to the agent
        
        Here are the available specialized agents:
        
        ObiProfile (OBI-001): Manages business profiles across platforms
        - Operations: update_profile_info, update_business_hours, update_images, get_profile_details
        
        ObiContent (OBI-002): Generates and optimizes business content
        - Operations: create_post, optimize_content, schedule_content, analyze_content_performance
        
        ObiTalk (OBI-003): Handles customer interactions and reviews
        - Operations: respond_to_review, analyze_sentiment, generate_response, manage_faq
        
        ObiMetrics (OBI-004): Analyzes business performance data
        - Operations: get_performance_report, analyze_trends, compare_metrics, forecast_performance
        
        ObiWatch (OBI-005): Monitors competitors
        - Operations: analyze_competitor, compare_offerings, identify_opportunities, track_competitor_changes
        
        ObiLocal (OBI-006): Provides local market intelligence
        - Operations: analyze_local_market, identify_trends, get_local_insights, area_performance
        
        ObiGuard (OBI-007): Ensures content compliance
        - Operations: check_compliance, moderate_content, flag_issues, suggest_corrections
        
        ObiSync (OBI-008): Manages external platform integration
        - Operations: sync_platform, check_sync_status, resolve_sync_issues, add_new_platform
        
        ObiVision (OBI-009): Analyzes visual content and documents
        - Operations: analyze_image, extract_text, validate_document, enhance_image
        
        ObiPlatform (OBI-010): Provides core platform services
        - Operations: manage_users, configure_settings, handle_notifications, process_requests
        
        User message history:
        {formatted_messages}
        
        Extract the semantic intent from this conversation and determine which agent should handle it.
        Respond with a structured JSON object:
        
        {{
            "agent": "agent_code",  // e.g., "OBI-001" for ObiProfile
            "operation": "operation_name",
            "parameters": {{
                // Structured parameters needed for the operation
            }},
            "human_readable_analysis": "Brief explanation of why this routing decision was made"
        }}
        """
    )
    
    # Create a chain that produces structured output
    return prompt | llm | JsonOutputParser()


def get_agent_specific_extractor(agent_code):
    """Get parameter extractor specific to an agent's domain."""
    
    extractors = {
        "OBI-001": PromptTemplate.from_template(
            """Extract profile-specific details from the conversation.
            Focus on business identity information like:
            - Name, logo, and branding elements
            - Business description and taglines
            - Hours of operation
            - Address and service area
            - Contact details
            - Categories and attributes
            
            Conversation:
            {messages}
            
            Operation: {operation}
            
            Return a detailed JSON structure with all profile parameters found.
            """
        ),
        "OBI-002": PromptTemplate.from_template(
            """Extract content-specific details from the conversation.
            Focus on content creation parameters like:
            - Content type (post, article, update)
            - Target platform
            - Key messages or themes
            - Tone and style preferences
            - Target audience
            - Call to action
            - Media components
            - Scheduling details
            
            Conversation:
            {messages}
            
            Operation: {operation}
            
            Return a detailed JSON structure with all content parameters found.
            """
        ),
        "OBI-003": PromptTemplate.from_template(
            """Extract customer interaction details from the conversation.
            Focus on:
            - Review content or sentiment
            - Customer questions
            - Response requirements
            - Service issues or feedback
            - Communication preferences
            
            Conversation:
            {messages}
            
            Operation: {operation}
            
            Return a detailed JSON structure with all customer interaction parameters found.
            """
        ),
        "OBI-004": PromptTemplate.from_template(
            """Extract metrics and analytics details from the conversation.
            Focus on:
            - Time periods for analysis
            - Specific metrics requested
            - Comparison benchmarks
            - Format preferences
            - Business goals
            
            Conversation:
            {messages}
            
            Operation: {operation}
            
            Return a detailed JSON structure with all metrics parameters found.
            """
        ),
        "OBI-005": PromptTemplate.from_template(
            """Extract competitor analysis details from the conversation.
            Focus on:
            - Specific competitors mentioned
            - Comparison aspects
            - Market segments
            - Competitive advantages/disadvantages
            - Strategic questions
            
            Conversation:
            {messages}
            
            Operation: {operation}
            
            Return a detailed JSON structure with all competitor analysis parameters found.
            """
        ),
        "OBI-006": PromptTemplate.from_template(
            """Extract local market intelligence details from the conversation.
            Focus on:
            - Geographical areas
            - Local demographics
            - Market trends
            - Seasonal factors
            - Local events or influences
            
            Conversation:
            {messages}
            
            Operation: {operation}
            
            Return a detailed JSON structure with all local market parameters found.
            """
        ),
        "OBI-007": PromptTemplate.from_template(
            """Extract content compliance details from the conversation.
            Focus on:
            - Content to be checked
            - Industry regulations
            - Compliance standards
            - Risk areas
            - Previous compliance issues
            
            Conversation:
            {messages}
            
            Operation: {operation}
            
            Return a detailed JSON structure with all compliance parameters found.
            """
        ),
        "OBI-008": PromptTemplate.from_template(
            """Extract platform integration details from the conversation.
            Focus on:
            - External platforms mentioned
            - Integration issues
            - Sync requirements
            - API or connection details
            - Data mapping needs
            
            Conversation:
            {messages}
            
            Operation: {operation}
            
            Return a detailed JSON structure with all integration parameters found.
            """
        ),
        "OBI-009": PromptTemplate.from_template(
            """Extract visual content analysis details from the conversation.
            Focus on:
            - Image or document references
            - Analysis requirements
            - Text extraction needs
            - Visual enhancement requests
            - Document validation requirements
            
            Conversation:
            {messages}
            
            Operation: {operation}
            
            Return a detailed JSON structure with all visual analysis parameters found.
            """
        ),
        "OBI-010": PromptTemplate.from_template(
            """Extract platform service details from the conversation.
            Focus on:
            - User management requests
            - Configuration changes
            - Notification preferences
            - System settings
            - Access control needs
            
            Conversation:
            {messages}
            
            Operation: {operation}
            
            Return a detailed JSON structure with all platform service parameters found.
            """
        ),
    }
    
    llm = get_advanced_llm()
    return extractors.get(agent_code, extractors["OBI-003"]) | llm | JsonOutputParser()


def agent_code_to_workflow(agent_code):
    """Convert agent code to workflow name."""
    mapping = {
        "OBI-000": "omnipulse",
        "OBI-001": "profile",
        "OBI-002": "content",
        "OBI-003": "talk",
        "OBI-004": "metrics",
        "OBI-005": "watch",
        "OBI-006": "local",
        "OBI-007": "guard",
        "OBI-008": "sync",
        "OBI-009": "vision",
        "OBI-010": "platform"
    }
    return mapping.get(agent_code, "talk")


async def omnipulse_router_node(state: AICompanionState):
    """
    Central orchestration agent (OBI-000) that determines which specialized agent 
    should handle the user's request.
    """
    # Stage 1: Determine which agent should handle the request
    base_router = await get_semantic_router_chain()
    routing_decision = await base_router.ainvoke({"formatted_messages": format_messages(state["messages"][-5:])})
    
    # Stage 2: Extract detailed parameters specific to the selected agent
    agent_code = routing_decision["agent"]
    operation = routing_decision["operation"]
    
    # Get agent-specific parameter extractor
    parameter_extractor = get_agent_specific_extractor(agent_code)
    
    # Extract detailed parameters for the specific agent/operation
    detailed_parameters = await parameter_extractor.ainvoke({
        "messages": format_messages(state["messages"][-5:]),
        "operation": operation,
        "base_parameters": routing_decision["parameters"]
    })
    
    return {
        "workflow": agent_code_to_workflow(agent_code),
        "operation": operation,
        "request_details": detailed_parameters,
        "routing_analysis": routing_decision["human_readable_analysis"]
    }


# Implementation of specialized agents (OBI-001 through OBI-010)

async def obiprofile_node(state: AICompanionState, config: RunnableConfig):
    """OBI-001: Manages business profile information across platforms."""
    memory_context = state.get("memory_context", "")
    request_details = state.get("request_details", {})
    operation = state.get("operation", "get_profile_details")
    
    # Create specialized prompt for profile management
    profile_prompt = PromptTemplate.from_template(
        """You are ObiProfile (OBI-001), the specialized agent for managing business profiles.
        
        Current operation: {operation}
        Request details: {request_details}
        Memory context: {memory_context}
        
        Generate a response that addresses the profile management request.
        Be specific, professional, and helpful in managing the business profile information.
        """
    )
    
    # Get response from LLM
    llm = get_advanced_llm()
    response = await (profile_prompt | llm).ainvoke({
        "operation": operation,
        "request_details": json.dumps(request_details, indent=2),
        "memory_context": memory_context,
    })
    
    return {"messages": AIMessage(content=response.content)}


async def obicontent_node(state: AICompanionState, config: RunnableConfig):
    """OBI-002: Generates and optimizes content for business profiles and posts."""
    memory_context = state.get("memory_context", "")
    request_details = state.get("request_details", {})
    operation = state.get("operation", "create_post")
    
    # Create specialized prompt for content generation
    content_prompt = PromptTemplate.from_template(
        """You are ObiContent (OBI-002), the specialized agent for content creation and optimization.
        
        Current operation: {operation}
        Request details: {request_details}
        Memory context: {memory_context}
        
        Generate a response that addresses the content creation/optimization request.
        Be creative, engaging, and strategically focused on business goals.
        """
    )
    
    # Get response from LLM
    llm = get_advanced_llm()
    response = await (content_prompt | llm).ainvoke({
        "operation": operation,
        "request_details": json.dumps(request_details, indent=2),
        "memory_context": memory_context,
    })
    
    return {"messages": AIMessage(content=response.content)}


async def obitalk_node(state: AICompanionState, config: RunnableConfig):
    """OBI-003: Handles review management and customer interactions."""
    memory_context = state.get("memory_context", "")
    request_details = state.get("request_details", {})
    operation = state.get("operation", "generate_response")
    
    # Create specialized prompt for customer interactions
    talk_prompt = PromptTemplate.from_template(
        """You are ObiTalk (OBI-003), the specialized agent for customer interactions and review management.
        
        Current operation: {operation}
        Request details: {request_details}
        Memory context: {memory_context}
        
        Generate a response that addresses the customer interaction request.
        Be empathetic, solution-oriented, and aligned with the business voice.
        """
    )
    
    # Get response from LLM
    llm = get_advanced_llm()
    response = await (talk_prompt | llm).ainvoke({
        "operation": operation,
        "request_details": json.dumps(request_details, indent=2),
        "memory_context": memory_context,
    })
    
    return {"messages": AIMessage(content=response.content)}


async def obimetrics_node(state: AICompanionState, config: RunnableConfig):
    """OBI-004: Analyzes performance data and generates insights."""
    memory_context = state.get("memory_context", "")
    request_details = state.get("request_details", {})
    operation = state.get("operation", "get_performance_report")
    
    # Create specialized prompt for analytics
    metrics_prompt = PromptTemplate.from_template(
        """You are ObiMetrics (OBI-004), the specialized agent for performance analytics and insights.
        
        Current operation: {operation}
        Request details: {request_details}
        Memory context: {memory_context}
        
        Generate a response that addresses the metrics and analytics request.
        Be data-driven, insightful, and focused on actionable business intelligence.
        """
    )
    
    # Get response from LLM
    llm = get_advanced_llm()
    response = await (metrics_prompt | llm).ainvoke({
        "operation": operation,
        "request_details": json.dumps(request_details, indent=2),
        "memory_context": memory_context,
    })
    
    return {"messages": AIMessage(content=response.content)}


async def obiwatch_node(state: AICompanionState, config: RunnableConfig):
    """OBI-005: Monitors competitors and identifies opportunities."""
    memory_context = state.get("memory_context", "")
    request_details = state.get("request_details", {})
    operation = state.get("operation", "analyze_competitor")
    
    # Create specialized prompt for competitor analysis
    watch_prompt = PromptTemplate.from_template(
        """You are ObiWatch (OBI-005), the specialized agent for competitor monitoring and analysis.
        
        Current operation: {operation}
        Request details: {request_details}
        Memory context: {memory_context}
        
        Generate a response that addresses the competitor analysis request.
        Be strategic, detailed, and focused on competitive advantage opportunities.
        """
    )
    
    # Get response from LLM
    llm = get_advanced_llm()
    response = await (watch_prompt | llm).ainvoke({
        "operation": operation,
        "request_details": json.dumps(request_details, indent=2),
        "memory_context": memory_context,
    })
    
    return {"messages": AIMessage(content=response.content)}


async def obilocal_node(state: AICompanionState, config: RunnableConfig):
    """OBI-006: Provides hyper-local market intelligence."""
    memory_context = state.get("memory_context", "")
    request_details = state.get("request_details", {})
    operation = state.get("operation", "analyze_local_market")
    
    # Create specialized prompt for local intelligence
    local_prompt = PromptTemplate.from_template(
        """You are ObiLocal (OBI-006), the specialized agent for hyper-local market intelligence.
        
        Current operation: {operation}
        Request details: {request_details}
        Memory context: {memory_context}
        
        Generate a response that addresses the local market intelligence request.
        Be geographically specific, trend-aware, and focused on local opportunities.
        """
    )
    
    # Get response from LLM
    llm = get_advanced_llm()
    response = await (local_prompt | llm).ainvoke({
        "operation": operation,
        "request_details": json.dumps(request_details, indent=2),
        "memory_context": memory_context,
    })
    
    return {"messages": AIMessage(content=response.content)}


async def obiguard_node(state: AICompanionState, config: RunnableConfig):
    """OBI-007: Ensures content compliance and moderation."""
    memory_context = state.get("memory_context", "")
    request_details = state.get("request_details", {})
    operation = state.get("operation", "check_compliance")
    
    # Create specialized prompt for compliance
    guard_prompt = PromptTemplate.from_template(
        """You are ObiGuard (OBI-007), the specialized agent for content compliance and moderation.
        
        Current operation: {operation}
        Request details: {request_details}
        Memory context: {memory_context}
        
        Generate a response that addresses the compliance and moderation request.
        Be precise, regulatory-aware, and focused on risk mitigation.
        """
    )
    
    # Get response from LLM
    llm = get_advanced_llm()
    response = await (guard_prompt | llm).ainvoke({
        "operation": operation,
        "request_details": json.dumps(request_details, indent=2),
        "memory_context": memory_context,
    })
    
    return {"messages": AIMessage(content=response.content)}


async def obisync_node(state: AICompanionState, config: RunnableConfig):
    """OBI-008: Manages integration with external platforms."""
    memory_context = state.get("memory_context", "")
    request_details = state.get("request_details", {})
    operation = state.get("operation", "sync_platform")
    
    # Create specialized prompt for platform integration
    sync_prompt = PromptTemplate.from_template(
        """You are ObiSync (OBI-008), the specialized agent for external platform integration.
        
        Current operation: {operation}
        Request details: {request_details}
        Memory context: {memory_context}
        
        Generate a response that addresses the platform integration request.
        Be technical, solution-oriented, and focused on seamless data flow.
        """
    )
    
    # Get response from LLM
    llm = get_advanced_llm()
    response = await (sync_prompt | llm).ainvoke({
        "operation": operation,
        "request_details": json.dumps(request_details, indent=2),
        "memory_context": memory_context,
    })
    
    return {"messages": AIMessage(content=response.content)}


async def obivision_node(state: AICompanionState, config: RunnableConfig):
    """OBI-009: Analyzes visual content and documents."""
    memory_context = state.get("memory_context", "")
    request_details = state.get("request_details", {})
    operation = state.get("operation", "analyze_image")
    
    # Create specialized prompt for visual analysis
    vision_prompt = PromptTemplate.from_template(
        """You are ObiVision (OBI-009), the specialized agent for visual content and document analysis.
        
        Current operation: {operation}
        Request details: {request_details}
        Memory context: {memory_context}
        
        Generate a response that addresses the visual content analysis request.
        Be detail-oriented, perceptive, and focused on extracting meaningful insights.
        """
    )
    
    # Get response from LLM
    llm = get_advanced_llm()
    response = await (vision_prompt | llm).ainvoke({
        "operation": operation,
        "request_details": json.dumps(request_details, indent=2),
        "memory_context": memory_context,
    })
    
    return {"messages": AIMessage(content=response.content)}


async def obiplatform_node(state: AICompanionState, config: RunnableConfig):
    """OBI-010: Provides core platform services and user interfaces."""
    memory_context = state.get("memory_context", "")
    request_details = state.get("request_details", {})
    operation = state.get("operation", "manage_users")
    
    # Create specialized prompt for platform services
    platform_prompt = PromptTemplate.from_template(
        """You are ObiPlatform (OBI-010), the specialized agent for core platform services.
        
        Current operation: {operation}
        Request details: {request_details}
        Memory context: {memory_context}
        
        Generate a response that addresses the platform service request.
        Be system-oriented, efficient, and focused on user experience.
        """
    )
    
    # Get response from LLM
    llm = get_advanced_llm()
    response = await (platform_prompt | llm).ainvoke({
        "operation": operation,
        "request_details": json.dumps(request_details, indent=2),
        "memory_context": memory_context,
    })
    
    return {"messages": AIMessage(content=response.content)}
