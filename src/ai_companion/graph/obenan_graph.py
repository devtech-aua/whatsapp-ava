"""
Graph definition for the Obenan Multi-Agent Framework.
This graph orchestrates the 11 specialized agents with OmniPulse as the central coordinator.
"""

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from ai_companion.graph.edges import should_summarize_conversation
from ai_companion.graph.nodes import (
    memory_extraction_node,
    memory_injection_node,
    summarize_conversation_node,
)
from ai_companion.graph.obenan_nodes import (
    omnipulse_router_node,
    obiprofile_node,
    obicontent_node,
    obitalk_node,
    obimetrics_node,
    obiwatch_node,
    obilocal_node,
    obiguard_node,
    obisync_node,
    obivision_node,
    obiplatform_node,
)
from ai_companion.graph.state import AICompanionState


def router_select_agent(state: AICompanionState) -> str:
    """Select which specialized agent should handle the request."""
    workflow = state.get("workflow")
    
    agent_mapping = {
        "profile": "obiprofile_node",
        "content": "obicontent_node",
        "talk": "obitalk_node",
        "metrics": "obimetrics_node",
        "watch": "obiwatch_node",
        "local": "obilocal_node",
        "guard": "obiguard_node",
        "sync": "obisync_node",
        "vision": "obivision_node",
        "platform": "obiplatform_node",
    }
    
    return agent_mapping.get(workflow, "obitalk_node")  # Default to talk agent


@lru_cache(maxsize=1)
def create_obenan_workflow_graph():
    """
    Create a workflow graph for the Obenan Multi-Agent Framework.
    This graph includes all 11 specialized agents coordinated by OmniPulse.
    """
    graph_builder = StateGraph(AICompanionState)

    # Add all nodes - one for each agent in the Obenan framework
    graph_builder.add_node("omnipulse_router_node", omnipulse_router_node)  # OBI-000
    graph_builder.add_node("obiprofile_node", obiprofile_node)  # OBI-001
    graph_builder.add_node("obicontent_node", obicontent_node)  # OBI-002
    graph_builder.add_node("obitalk_node", obitalk_node)  # OBI-003
    graph_builder.add_node("obimetrics_node", obimetrics_node)  # OBI-004
    graph_builder.add_node("obiwatch_node", obiwatch_node)  # OBI-005
    graph_builder.add_node("obilocal_node", obilocal_node)  # OBI-006
    graph_builder.add_node("obiguard_node", obiguard_node)  # OBI-007
    graph_builder.add_node("obisync_node", obisync_node)  # OBI-008
    graph_builder.add_node("obivision_node", obivision_node)  # OBI-009
    graph_builder.add_node("obiplatform_node", obiplatform_node)  # OBI-010
    
    # Memory nodes remain
    graph_builder.add_node("memory_extraction_node", memory_extraction_node)
    graph_builder.add_node("memory_injection_node", memory_injection_node)
    graph_builder.add_node("summarize_conversation_node", summarize_conversation_node)

    # Define the flow
    # First extract memories from user message
    graph_builder.add_edge(START, "memory_extraction_node")
    
    # Then route to omnipulse for orchestration
    graph_builder.add_edge("memory_extraction_node", "omnipulse_router_node")
    
    # Inject memories before processing
    graph_builder.add_edge("omnipulse_router_node", "memory_injection_node")
    
    # Route to appropriate specialized agent based on omnipulse decision
    graph_builder.add_conditional_edges(
        "memory_injection_node", 
        router_select_agent
    )
    
    # All agents route to summarization check
    for agent in [
        "obiprofile_node", 
        "obicontent_node", 
        "obitalk_node", 
        "obimetrics_node", 
        "obiwatch_node", 
        "obilocal_node", 
        "obiguard_node", 
        "obisync_node", 
        "obivision_node", 
        "obiplatform_node"
    ]:
        graph_builder.add_conditional_edges(agent, should_summarize_conversation)
    
    graph_builder.add_edge("summarize_conversation_node", END)

    return graph_builder


# Return the graph builder, not the compiled graph
obenan_graph = create_obenan_workflow_graph()
