from typing import List, Dict, Any


class PromptContext:
    """
    A composable context for constructing OpenAI-compatible messages,
    with slots for system instruction, long-term memory, short-term memory, and recent chat history.
    """
    def __init__(
        self,
        query_messages: List[Dict[str, Any]],
        retrieved_memories: List[Dict[str, Any]] = None,
        retrieved_chat_history: List[Dict[str, Any]] = None,
        max_chat_history: int = 10,
    ):
        self.query_messages = query_messages
        self.retrieved_memories = retrieved_memories if retrieved_memories is not None else []
        self.retrieved_chat_history = retrieved_chat_history if retrieved_chat_history is not None else []
        self.max_chat_history = max_chat_history

    @property
    def system_instruction(self) -> Dict[str, Any]:
        # Check if the first message in query_messages is a system message
        if self.query_messages and self.query_messages[0].get("role") == "system":
            return self.query_messages[0]
        # Otherwise return the default system message
        return {"role": "system", "content": "You are a helpful assistant."}
    
    @property
    def system_content(self) -> str:
        """Return just the content of the system message for providers that need it separately"""
        return self.system_instruction.get("content", "You are a helpful assistant.")
    
    @property
    def user_query(self) -> List[Dict[str, Any]]:
        """
        Returns a list of OpenAI message dictionaries representing the user query.
        If the first message is a system message, it is excluded; otherwise, returns all messages as is.
        """
        if self.query_messages and self.query_messages[0].get("role") == "system":
            return self.query_messages[1:]
        return self.query_messages

    @property
    def long_term_memory(self) -> List[Dict[str, Any]]:
        """
        Returns a list of long-term memory items.
        Includes memories with cross_session scope OR memories with no scope (None/missing).
        """
        return [
            item for item in self.retrieved_memories
            if item.get("metadata", {}).get("scope") in ["cross_session", None] or 
               "scope" not in item.get("metadata", {})
        ]

    @property
    def short_term_memory(self) -> List[Dict[str, Any]]:
        """
        Returns a list of short-term memory items.
        Only includes memories specifically marked as in_session scope.
        """
        return [
            item for item in self.retrieved_memories
            if item.get("metadata", {}).get("scope") == "in_session"
        ]

    @property
    def chat_history(self) -> List[Dict[str, Any]]:
        if not self.retrieved_chat_history:
            return []
        return self.retrieved_chat_history[-self.max_chat_history:]

    def compose_for_openai(self) -> List[Dict[str, Any]]:
        """Compose the final message list for OpenAI API."""
        messages: List[Dict[str, str]] = []

        # 1) System instruction
        messages.append(self.system_instruction)

        # 2) Long-term memory with prompt-engineering wrapper
        if self.long_term_memory:
            lt_snippets_list = []
            for item in self.long_term_memory:
                content = item.get("content", "N/A")
                mem_type = (item.get("memory_type") or item.get("type") or "unknown").upper()

                prefix = f"[{mem_type}]"
                
                # Format the snippet including type and role (if applicable)
                lt_snippets_list.append(f"{prefix}: {content}")

            # Join the formatted snippets
            lt_snippets = "\n- ".join(lt_snippets_list)
            
            # Construct the system message content
            lt_content = (
                "You have access to the following long-term memory snippets. "
                "These might include past messages, knowledge, or summaries. "
                "Use these insights to inform your responses but do not repeat them verbatim unless necessary.\n\n"
                f"- {lt_snippets}"
            )
            messages.append({"role": "system", "content": lt_content})

        # 3) Short-term memory with wrapper
        if self.short_term_memory:
            st_snippets_list = []
            for item in self.short_term_memory:
                content = item.get("content", "N/A")
                mem_type = (item.get("memory_type") or item.get("type") or "unknown").upper()

                prefix = f"[{mem_type}]"

                # Format the snippet including type and role (if applicable)
                st_snippets_list.append(f"{prefix}: {content}")

            # Join the formatted snippets
            st_snippets = "\n- ".join(st_snippets_list)

            st_content = (
                "Here are the most relevant recent notes from the current session context. "
                "Reference them as needed:\n\n"
                f"- {st_snippets}"
            )
            messages.append({"role": "system", "content": st_content})

        # 4) Recent chat history: truncate to the last N turns
        recent_history = self.chat_history[-self.max_chat_history:]
        messages.extend(recent_history)

        # 5) User query
        messages.extend(self.user_query)

        return messages
    
    def compose_for_anthropic(self) -> tuple[str, List[Dict[str, Any]]]:
        """
        Compose system message and messages list for Anthropic API.
        
        Returns:
            tuple: (system_prompt, messages)
                - system_prompt: String containing the system instructions
                - messages: List of message dictionaries in Anthropic format
        """
        system_parts = []
        
        # 1) Base system instruction
        system_parts.append(self.system_content)
        
        # 2) Long-term memory
        if self.long_term_memory:
            lt_snippets_list = []
            for item in self.long_term_memory:
                content = item.get("content", "N/A")
                mem_type = (item.get("memory_type") or item.get("type") or "unknown").upper()

                prefix = f"[{mem_type}]"
                
                lt_snippets_list.append(f"{prefix}: {content}")
            
            lt_snippets = "\n- ".join(lt_snippets_list)
            
            lt_content = (
                "\nYou have access to the following long-term memory snippets. "
                "These might include past messages, knowledge, or summaries. "
                "Use these insights to inform your responses but do not repeat them verbatim unless necessary.\n\n"
                f"- {lt_snippets}"
            )
            system_parts.append(lt_content)
        
        # 3) Short-term memory
        if self.short_term_memory:
            st_snippets_list = []
            for item in self.short_term_memory:
                content = item.get("content", "N/A")
                mem_type = (item.get("memory_type") or item.get("type") or "unknown").upper()

                prefix = f"[{mem_type}]"

                st_snippets_list.append(f"{prefix}: {content}")

            st_snippets = "\n- ".join(st_snippets_list)

            st_content = (
                "\nHere are the most relevant recent notes from the current session context. "
                "Reference them as needed:\n\n"
                f"- {st_snippets}"
            )
            system_parts.append(st_content)
        
        # Combine all system parts
        system_prompt = "\n".join(system_parts)
        
        # Prepare messages in Anthropic format
        anthropic_messages = []
        
        # 4) Add chat history
        recent_history = self.chat_history[-self.max_chat_history:]
        for msg in recent_history:
            role = msg.get("role")
            content = msg.get("content", "")
            
            if role in ["user", "assistant"]:
                anthropic_messages.append({
                    "role": role,
                    "content": content
                })
        
        # 5) Add user query (excluding any system messages)
        for msg in self.user_query:
            role = msg.get("role")
            content = msg.get("content", "")
            
            if role in ["user", "assistant"]:
                anthropic_messages.append({
                    "role": role,
                    "content": content
                })
        
        return system_prompt, anthropic_messages

    def compose_for_gemini(self) -> List[Dict[str, Any]]:
        """
        Compose the final message list for Gemini API.
        Gemini only accepts 'user' and 'model' roles, so system content needs to be
        embedded into the user messages rather than as separate system messages.
        """
        messages: List[Dict[str, str]] = []
        system_parts = []

        # 1) Collect all system content parts
        system_parts.append(self.system_content)

        # 2) Long-term memory
        if self.long_term_memory:
            lt_snippets_list = []
            for item in self.long_term_memory:
                content = item.get("content", "N/A")
                mem_type = (item.get("memory_type") or item.get("type") or "unknown").upper()

                prefix = f"[{mem_type}]"
                
                lt_snippets_list.append(f"{prefix}: {content}")

            lt_snippets = "\n- ".join(lt_snippets_list)
            
            lt_content = (
                "\nYou have access to the following long-term memory snippets. "
                "These might include past messages, knowledge, or summaries. "
                "Use these insights to inform your responses but do not repeat them verbatim unless necessary.\n\n"
                f"- {lt_snippets}"
            )
            system_parts.append(lt_content)

        # 3) Short-term memory
        if self.short_term_memory:
            st_snippets_list = []
            for item in self.short_term_memory:
                content = item.get("content", "N/A")
                mem_type = (item.get("memory_type") or item.get("type") or "unknown").upper()

                prefix = f"[{mem_type}]"

                st_snippets_list.append(f"{prefix}: {content}")

            st_snippets = "\n- ".join(st_snippets_list)

            st_content = (
                "\nHere are the most relevant recent notes from the current session context. "
                "Reference them as needed:\n\n"
                f"- {st_snippets}"
            )
            system_parts.append(st_content)

        # 4) Recent chat history: truncate to the last N turns
        recent_history = self.chat_history[-self.max_chat_history:]
        
        # 5) Add chat history and current query
        all_conversation_messages = recent_history + self.user_query
        
        # 6) Process all messages and embed system content in the first user message only
        combined_system_content = "\n".join(system_parts) if system_parts else ""
        system_content_added = False
        
        for msg in all_conversation_messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            # Only include user and assistant messages (assistant will be converted to model by adapter)
            if role in ["user", "assistant"]:
                if role == "user" and not system_content_added and combined_system_content:
                    # Add system content to the first user message only
                    enhanced_content = f"{combined_system_content}\n\nUser: {content}"
                    messages.append({
                        "role": "user",
                        "content": enhanced_content
                    })
                    system_content_added = True
                else:
                    # Add message as-is
                    messages.append({
                        "role": role,
                        "content": content
                    })

        return messages
