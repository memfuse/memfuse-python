<a id="readme-top"></a>

[![GitHub license](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/Percena/MemFuse/blob/readme/LICENSE)

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://memfuse.vercel.app/">
    <img src="https://raw.githubusercontent.com/memfuse/memfuse-python/refs/heads/main/assets/logo.png" alt="MemFuse Logo"
         style="max-width: 90%; height: auto; display: block; margin: 0 auto; padding-left: 16px; padding-right: 16px;">
  </a>
  <br />

  <p align="center">
    <strong>MemFuse Python SDK</strong>
    <br />
    The official Python client for MemFuse, the open-source memory layer for LLMs.
    <br />
    <a href="https://memfuse.vercel.app/"><strong>Explore the Docs »</strong></a>
    <br />
    <br />
    <a href="https://memfuse.vercel.app/">View Demo</a>
    &middot;
    <a href="https://github.com/memfuse/memfuse-python/issues">Report Bug</a>
    &middot;
    <a href="https://github.com/memfuse/memfuse-python/issues">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-memfuse">About MemFuse</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#quick-start">Quick Start</a></li>
    <li><a href="#examples">Examples</a></li>
    <li><a href="#documentation">Documentation</a></li>
    <li><a href="#community--support">Community & Support</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

## About MemFuse

Large-language-model apps are stateless out of the box.
Once the context window rolls over, yesterday's chat, the user's name, or that crucial fact vanishes.

**MemFuse** plugs a persistent, query-able memory layer between your LLM and a storage backend so agents can:

- remember user preferences across sessions
- recall facts & events thousands of turns later
- trim token spend instead of resending the whole chat history
- learn continuously and self-improve over time

This repository contains the Python SDK for interacting with a MemFuse server. For more information about the MemFuse server and its features, please visit the [main MemFuse repository](https://github.com/memfuse/memfuse).

## Installation

First, ensure you have a MemFuse server running. To set up the MemFuse server locally:

1.  Clone the [main MemFuse repository](https://github.com/memfuse/memfuse):
    ```bash
    git clone https://github.com/memfuse/memfuse.git
    cd memfuse
    ```
2.  Once in the `memfuse` directory, install its dependencies and run the server using one of the following methods:

    **Using pip:**

    ```bash
    pip install -e .
    python -m memfuse_core
    ```

    **Or using Poetry:**

    ```bash
    poetry install
    poetry run memfuse-core
    ```

Then, install the MemFuse Python SDK:

```bash
pip install memfuse
```

## Quick Start

Here's a basic example of how to use the MemFuse Python SDK with OpenAI:

```python
from memfuse.llm import OpenAI
from memfuse import MemFuse
import os


memfuse_client = MemFuse(
  # api_key=os.getenv("MEMFUSE_API_KEY")
  # base_url=os.getenv("MEMFUSE_BASE_URL"),
)

memory = memfuse_client.init(
  user="alice",
  # agent="agent_default",
  # session=<randomly-generated-uuid>
)

# Initialize your LLM client with the memory scope
llm_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),  # Your OpenAI API key
    memory=memory
)

# Make a chat completion request
response = llm_client.chat.completions.create(
    model="gpt-4o", # Or any model supported by your LLM provider
    messages=[{"role": "user", "content": "I'm planning a trip to Mars. What is the gravity there?"}]
)

print(f"Response: {response.choices[0].message.content}")
# Example Output: Response: Mars has a gravity of about 3.721 m/s², which is about 38% of Earth's gravity.
```

<!-- Ask a follow-up question. MemFuse automatically recalls relevant context. -->

Now, ask a follow-up question. MemFuse will automatically recall relevant context from the previous turn:

```python
# Ask a follow-up question. MemFuse automatically recalls relevant context.
followup_response = llm_client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What are some challenges of living on that planet?"}]
)

print(f"Follow-up: {followup_response.choices[0].message.content}")
# Example Output: Follow-up: Some challenges of living on Mars include its thin atmosphere, extreme temperatures, high radiation levels, and the lack of liquid water on the surface.
```

MemFuse will automatically manage recalling relevant information and storing new memories from the conversation within the specified `memory` scope.

## Examples

You can find more detailed examples in the [examples/](examples/) directory of this repository, showcasing:

- Basic and asynchronous operations
- Continued conversations
- Integrations with Gradio for chatbots (including streaming)

## Documentation

- For detailed information about the MemFuse server, its architecture, and advanced configuration, please refer to the [main MemFuse documentation](https://memfuse.vercel.app/).
- SDK-specific documentation and API references will be added here soon.

## Community & Support

Join the MemFuse community:

- **GitHub Discussions:** For roadmap voting, RFCs, Q&A in the [main MemFuse repository](https://github.com/memfuse/memfuse).

If MemFuse is helpful to you, please ⭐ star the [main repo](https://github.com/memfuse/memfuse) and this SDK repo!

## License

This MemFuse Python SDK is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for more details.
(You'll need to add a LICENSE file to this SDK repository, typically a copy of the Apache 2.0 license text).
