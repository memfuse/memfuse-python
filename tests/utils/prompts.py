from typing import List, Dict
from memfuse.prompts.prompt_manager import PromptManager
from collections import namedtuple


Prompt = namedtuple("Prompt", ["system", "user"])


def create_prompt(
        question: str, 
        choices: List[str], 
        context_items: List[Dict]) -> Prompt:
    """Create a prompt for the LLM with rich context including who said what and when.
    
    Args:
        question: The question to answer
        choices: List of possible answers
        context_items: List of context items with content, role, timestamp and metadata
        
    Returns:
        Formatted prompt string with structured conversation history
    """
    formatted_choices = "\n".join([f"{i}. {choice}" for i, choice in enumerate(choices)])
    main_topic = question.lower().replace('hey, remember that time we talked about', '').strip()
    if not main_topic:
        main_topic = question
        
    # Format the context as a structured conversation with metadata
    formatted_context = ""
    
    # Sort context items by creation timestamp if available
    try:
        sorted_items = sorted(context_items, key=lambda x: x.get('created_at', ''))
    except Exception:
        # If sorting fails, use the original order
        sorted_items = context_items
    
    # Group messages by session to maintain conversation coherence
    sessions = {}
    for item in sorted_items:
        # Skip items without content
        if not item.get('content'):
            continue
            
        # Get session information
        session_id = item.get('metadata', {}).get('session_id', 'unknown_session')
        session_name = item.get('metadata', {}).get('session_name', 'Unknown Session')
        
        if session_id not in sessions:
            sessions[session_id] = {
                'name': session_name,
                'messages': []
            }
        
        # Add message to the session
        sessions[session_id]['messages'].append(item)
    
    # Now build the formatted context
    for session_id, session_data in sessions.items():
        session_name = session_data['name']
        formatted_context += f"\n--- CONVERSATION: {session_name} ---\n"
        
        for item in session_data['messages']:
            content = item.get('content', '')
            created_at = item.get('created_at', '')
            
            # Since content now contains pre-formatted conversation chunks with embedded roles,
            # we just add the content directly without role prefixes
            if content:
                # Add timestamp information if available (as a comment/header for the chunk)
                if created_at:
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        timestamp_str = f"[Conversation from {dt.strftime('%Y-%m-%d %H:%M:%S')}]\n"
                        formatted_context += timestamp_str
                    except Exception:
                        formatted_context += f"[Conversation from {created_at}]\n"
                
                formatted_context += f"{content}\n\n"
    
    # If no sessions were found or all were empty, use a fallback approach
    if not formatted_context.strip():
        formatted_context = "No structured conversation history available. Here's what I know:\n\n"
        for item in context_items:
            content = item.get('content', '')
            if content:
                formatted_context += f"- {content}\n\n"

    system_prompt = PromptManager.get_prompt("question_answering_system")
    user_prompt = PromptManager.get_prompt(
        "question_answering_user", 
        formatted_context=formatted_context, 
        question=question, 
        formatted_choices=formatted_choices
    )

    return Prompt(system_prompt, user_prompt)