# MemFuse Python SDK Testing Strategy Guide

## Overview

This document defines a comprehensive testing strategy for a Python SDK, organized in layers from basic smoke tests to full end-to-end scenarios. Each layer builds upon the previous one, ensuring thorough coverage and early failure detection.

## Testing Framework Instructions

### Core Requirements

1. **Use only pytest** - Never use unittest
2. **Configure pytest using `pytest.ini`** - Do not use pyproject.toml for pytest configuration
3. **Single test runner script** - Use `scripts/run_tests.py` as the single entry point for all tests
4. **Layer-based execution** - Run tests layer by layer with short-circuit on failure
5. **Clear test organization** - Use descriptive test file names following pattern: `test_<layer>_<component>.py`

### Project Structure

```
project/
├── scripts/
│   └── run_tests.py          # Single test runner
├── tests/
│   ├── conftest.py           # Shared fixtures and configuration
│   ├── smoke/                # Layer 1
│   ├── unit/                 # Layer 2
│   ├── error_handling/       # Layer 3
│   ├── integration/          # Layer 4
│   ├── dx/                   # Layer 5
│   ├── e2e/                  # Layer 6
│   ├── benchmarks/           # Layer 7
│   ├── archive/              # Legacy tests
│   └── utils/                # Test utilities
├── pytest.ini                # Pytest configuration
└── src/                      # SDK source code
```

## How to Run Tests

### 1. Run All Tests of All Layers

```bash
poetry run python scripts/run_tests.py
```

This runs all layers in sequence: smoke → unit → error_handling → integration → dx → e2e → benchmarks

### 2. Run Specific Layer

```bash
poetry run python scripts/run_tests.py --layer <layer_name>
```

**Available layers:**
- `poetry run python scripts/run_tests.py --layer smoke`
- `poetry run python scripts/run_tests.py --layer unit`
- `poetry run python scripts/run_tests.py --layer error_handling`
- `poetry run python scripts/run_tests.py --layer integration`
- `poetry run python scripts/run_tests.py --layer dx`
- `poetry run python scripts/run_tests.py --layer e2e`
- `poetry run python scripts/run_tests.py --layer benchmarks`

### 3. Run Tests Up to a Specific Layer

**This feature is not currently implemented** in the test runner. The script only supports:
- All layers (default behavior)
- Single specific layer (`--layer`)

If you need to run tests up to a specific layer, you'd need to either:
1. Modify the script to add a `--stop-at` parameter
2. Run layers individually in sequence until your target layer

### Useful Additional Options

```bash
# List available layers and their status
poetry run python scripts/run_tests.py --list

# Run with verbose output
poetry run python scripts/run_tests.py --layer e2e --verbose

# Hide test output (capture stdout/stderr)
poetry run python scripts/run_tests.py --layer e2e --hide-output

# Run specific test file directly
poetry run pytest tests/smoke/test_smoke_basic.py -v

# Run specific test function
poetry run pytest tests/e2e/test_e2e_memory_followup.py::test_memory_followup_includes_mars_reference -v -s
```

**Note**: The "run up to specific layer" functionality would be a useful enhancement to add to the test runner script.

## Testing Layers

### Layer 1: Baseline "Smoke" Tests

**Purpose**: Verify basic SDK viability before running deeper tests

- Python version compatibility check
- Can import SDK package
- Can import all public classes/functions
- Can instantiate all client types (sync/async)
- Required dependencies are available and correct versions
- Basic connectivity test (can be skipped if no network available)

**Failure behavior**: If any smoke test fails, skip all other layers

### Layer 2: Unit Tests

**Purpose**: Test each component in complete isolation

#### Principles

- **Mock ALL external dependencies**: Use `unittest.mock` to mock `aiohttp.ClientSession`, `requests.Session`
- **Test each component independently**: No real HTTP calls, no real file I/O
- **Mock internal methods appropriately**: When testing high-level methods, mock lower-level ones
- **Test both success and failure paths**: Include edge cases and error conditions
- **Test configuration handling**: Environment variables, defaults, validation
- **Test resource management**: Proper cleanup, context managers, connection pooling

#### Coverage Areas

**Core Client Functionality**

- AsyncMemFuse
  - Session lifecycle (creation, reuse, cleanup)
  - Request building and parameter handling
  - Context manager protocol (`__aenter__`, `__aexit__`)
  - Instance tracking and cleanup
  - Configuration parsing and validation
- MemFuse (sync)
  - Sync session management
  - Sync request building
  - Context manager protocol (`__enter__`, `__exit__`)
  - Thread safety considerations

**API Modules** (test each in isolation with mocked client)

- UsersApi: create, get, update, delete, list operations
- AgentsApi: agent lifecycle and configuration
- SessionsApi: session management and state
- MessagesApi: message handling and streaming
- KnowledgeApi: knowledge base operations

**Memory Classes**

- AsyncMemory: state management, operation delegation
- Memory (sync): sync wrapper behavior
- Proper error propagation
- State consistency

**Utilities and Helpers**

- Serialization/deserialization
- URL construction
- Parameter validation
- Response parsing

### Layer 3: Error Handling Tests

**Purpose**: Verify robust error handling across all components

- Network errors (connection refused, timeout, DNS failure)
- HTTP errors (4xx, 5xx responses)
- Invalid data errors (malformed JSON, missing fields)
- Configuration errors (missing API key, invalid URLs)
- Resource exhaustion (rate limits, quotas)
- Async-specific errors (cancelled operations, event loop issues)

**Testing approach**: Use mocks to simulate each error condition

### Layer 4: Integration Tests

**Purpose**: Test component interactions with mocked HTTP layer

- Full API workflows (create user → create agent → create session → send message)
- Entity relationship cascades
- Cross-module operations
- State consistency across operations
- Pagination and filtering
- Batch operations
- Concurrent operations (for async client)

**Mock strategy**: Mock at HTTP level, allowing real SDK logic to execute

### Layer 5: Developer Experience (DX) Tests

**Purpose**: Ensure SDK is easy to use correctly

- Example code execution
- Type hints validation (using mypy in tests)
- Documentation code snippets
- Common use case patterns
- Error message clarity
- API discoverability

### Layer 6: End-to-End Tests

**Purpose**: Validate against real backend (when available)

#### Current E2E Test Suite

The e2e layer includes comprehensive tests that validate MemFuse's conversational memory capabilities:

**Memory Follow-up Test** (`test_e2e_memory_followup.py`)

- Tests basic conversational memory by asking about Mars, then using "that planet" in follow-up
- Validates that MemFuse injects previous context into LLM prompts
- Uses RAGAS semantic similarity (with Ollama embeddings) for automated verification
- Falls back to regex matching if RAGAS/Ollama unavailable

**Async Memory Follow-up Test** (`test_e2e_async_memory_followup.py`)

- Async version of the memory follow-up test using `AsyncMemFuse` and `AsyncOpenAI`
- Tests both manual cleanup and context manager patterns
- Includes Jupiter/moons conversation test with direct memory operations
- Proper async resource management with cleanup verification

**Multi-turn Memory Test** (`test_e2e_multi_turn_memory.py`)

- Comprehensive 6-turn conversation flow: Moon → Mars → colonization → challenges → resources → Europa
- Tests progressive context building across multiple conversation turns
- Ultimate memory test: "Europa compared to the two we discussed" after 5 intervening turns
- Validates that MemFuse maintains long-term conversational context

#### Test Verification Strategy

**Primary**: RAGAS Semantic Similarity

- Uses Ollama embeddings (`nomic-embed-text` model) for local, cost-free evaluation
- Computes semantic similarity scores between expected and actual responses
- Configurable similarity thresholds (currently 0.15)
- Provides transparency with logged scores and analysis

**Fallback**: Regex Pattern Matching

- Used when RAGAS or Ollama dependencies unavailable
- Pattern-based verification for key terms and concepts
- Ensures tests work in minimal environments

#### Running E2E Tests

**All E2E tests**:

```bash
poetry run python scripts/run_tests.py --layer e2e
```

**Without output logging** (to hide RAGAS scores and debug info):

```bash
poetry run python scripts/run_tests.py --layer e2e --hide-output
```

**Individual test**:

```bash
poetry run pytest tests/e2e/test_e2e_memory_followup.py::test_memory_followup_includes_mars_reference -v -s
```

#### Requirements

- `OPENAI_API_KEY` environment variable (required)
- `MEMFUSE_BASE_URL` environment variable (defaults to `http://127.0.0.1:8000`)
- Running MemFuse server instance
- Optional: Ollama server with `nomic-embed-text` model for RAGAS evaluation

#### Dependencies

E2E tests use additional dev dependencies for enhanced verification:

- `ragas ^0.2.0` - For semantic similarity evaluation
- `langchain-ollama ^0.2.0` - For local embeddings (avoids OpenAI API costs)

**Configuration**: Tests are automatically skipped when required environment variables are missing or server is unavailable

### Layer 7: Benchmark Tests

**Purpose**: Performance testing and accuracy metrics

#### Current Benchmark Test Suite

The benchmarks layer includes tests that validate MemFuse's performance and accuracy:

**MSC Accuracy Test** (`test_msc_accuracy.py`)

- Tests memory accuracy using Multi-Session Chat (MSC) evaluation
- Validates retrieval quality and conversational memory performance

**Retrieval Metrics Test** (`test_retrieval_metrics.py`)

- Tests retrieval system metrics and debugging capabilities
- Validates memory storage and retrieval accuracy

#### Running Benchmark Tests

```bash
poetry run python scripts/run_tests.py --layer benchmarks
```

## Test Implementation Guidelines

### Fixture Best Practices

```python
# conftest.py
import pytest
from unittest.mock import Mock, AsyncMock

@pytest.fixture
def mock_session():
    """Provides a mocked aiohttp session."""
    session = AsyncMock()
    session.request.return_value.__aenter__.return_value.json = AsyncMock()
    return session

@pytest.fixture
def mock_client(mock_session):
    """Provides a client with mocked session."""
    # Implementation here
```

### Assertion Patterns

```python
# Be specific with assertions
assert response.status_code == 200  # Not just assert response
assert user.id == "expected_id"
assert len(users) == 5

# Use pytest features
with pytest.raises(ValueError, match="Invalid API key"):
    client = MemFuse(api_key="")

# Check call counts and arguments
mock_session.request.assert_called_once_with(
    "POST",
    "https://api.example.com/users",
    json={"name": "Test User"}
)
```

### Test Naming Conventions

- `test_<what>_<condition>_<expected_result>`
- Examples:
  - `test_create_user_with_valid_data_returns_user_object`
  - `test_get_session_with_invalid_id_raises_not_found`
  - `test_client_context_manager_closes_session_on_exit`

## Run Test Script Structure

```python
# scripts/run_tests.py
#!/usr/bin/env python3
import sys
import subprocess
from pathlib import Path

LAYERS = [
    ("smoke", "tests/smoke"),
    ("unit", "tests/unit"),
    ("error_handling", "tests/error_handling"),
    ("integration", "tests/integration"),
    ("dx", "tests/dx"),
    ("e2e", "tests/e2e"),
    ("benchmarks", "tests/benchmarks")
]

def run_layer(name, path):
    """Run tests for a specific layer."""
    print(f"\n{'='*60}")
    print(f"Running {name} tests...")
    print(f"{'='*60}")

    result = subprocess.run(
        [sys.executable, "-m", "pytest", path, "-v"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"\n{name} tests FAILED!")
        print(result.stdout)
        print(result.stderr)
        return False

    print(f"{name} tests PASSED!")
    return True

def main():
    """Run all test layers in sequence."""
    parser = argparse.ArgumentParser(description='Run MemFuse Python SDK tests')
    parser.add_argument('--layer', '-l', help='Run specific layer only')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--show-output', '-s', action='store_true', help='Show test output (stdout/print statements)')
    parser.add_argument('--list', action='store_true', help='List available layers')

    args = parser.parse_args()

    if args.list:
        print("Available test layers:")
        for name, path in LAYERS:
            exists = "✅" if Path(path).exists() else "❌"
            print(f"  {exists} {name:<15} - {path}")
        return

    if args.layer:
        success = run_specific_layer(args.layer, args.verbose, args.show_output)
        sys.exit(0 if success else 1)

    for name, path in LAYERS:
        if not Path(path).exists():
            print(f"Skipping {name} tests - path {path} not found")
            continue

        if not run_layer(name, path, args.verbose, args.show_output):
            print(f"\nStopping test run due to {name} layer failure")
            sys.exit(1)

    print("\n✅ All test layers passed!")

if __name__ == "__main__":
    main()
```

## pytest.ini Configuration

```ini
[pytest]
# Test discovery
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Output options
addopts =
    -ra
    --strict-markers
    --strict-config
    --tb=short
    -v

# Markers
markers =
    smoke: Basic functionality tests
    unit: Unit tests with mocked dependencies
    integration: Integration tests with mocked HTTP
    e2e: End-to-end tests requiring real server
    benchmarks: Performance and benchmark tests
    slow: Tests that take significant time
    asyncio: Async tests

# Async configuration (requires pytest-asyncio)
asyncio_mode = auto
asyncio_default_fixture_loop_scope = function

# Timeout (commented out as it requires pytest-timeout)
# timeout = 30
# timeout_method = thread

# Warnings
filterwarnings =
    error
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning
    ignore::pytest.PytestUnraisableExceptionWarning
```

## Additional Considerations

### Performance Testing

- Add benchmarks for critical paths
- Monitor memory usage in long-running tests
- Test concurrent request handling

### Security Testing

- Validate API key handling (not logged, not in errors)
- Test SSL/TLS certificate validation
- Ensure sensitive data is properly masked

### Compatibility Testing

- Test across Python versions (use tox or GitHub Actions matrix)
- Test with different dependency versions
- Verify platform compatibility (Windows, macOS, Linux)

### Documentation Testing

- Use doctest for code examples in docstrings
- Validate all README examples
- Test tutorial code

### Continuous Integration

- Run layers in parallel where possible
- Cache dependencies
- Generate and publish coverage reports
- Fail fast on critical layers

## Common Pitfalls to Avoid

1. **Don't test implementation details** - Test behavior, not internal state
2. **Avoid shared state between tests** - Each test should be independent
3. **Don't use real network calls in unit tests** - Always mock external dependencies
4. **Avoid time-dependent tests** - Use time mocking for time-sensitive logic
5. **Don't ignore async test warnings** - Properly handle event loops
6. **Avoid overly complex test setups** - If setup is complex, consider refactoring the code

## Maintenance Guidelines

- Review and update tests when API changes
- Keep test data minimal and focused
- Regular cleanup of obsolete tests
- Monitor test execution time and optimize slow tests
- Maintain test coverage above 80% for critical paths
