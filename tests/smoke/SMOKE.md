# Smoke Test Review Checklist

This document provides a systematic checklist for reviewing all smoke tests in the `tests/smoke/` directory. Smoke tests focus on basic functionality, import validation, and structural correctness without making real network calls.

## Overview

Smoke tests are designed to:

- ✅ Verify basic package imports and structure
- ✅ Test class instantiation without side effects
- ✅ Validate dependency availability
- ✅ Check method existence and callable structure
- ✅ Ensure Python version compatibility

**Note**: These tests use mocking to avoid real network calls and focus purely on structural validation.

---

## 📋 Review Progress

### 1. Basic Connectivity & Structure Tests

**File**: `test_smoke_basic.py` (11KB, 348 lines)

- [x] Basic connectivity structure tests
- [x] Request method existence validation
- [x] Session management method tests
- [x] Async/sync session creation with mocking
- [x] Health check method structure validation
- [x] Context manager functionality tests
- [x] URL construction validation
- [x] API key handling tests
- [x] Headers construction tests
- [x] Client cleanup method tests

### 2. Dependency Availability Tests

**File**: `test_smoke_dependencies.py` (8.7KB, 308 lines)

- [ ] Core HTTP dependencies (aiohttp, requests, httpx)
- [ ] Data validation dependencies (pydantic)
- [ ] LLM integration dependencies (openai, anthropic, google-genai, ollama)
- [ ] Template rendering dependencies (jinja2)
- [ ] Configuration dependencies (python-frontmatter, python-dotenv)
- [ ] UI dependencies (gradio)
- [ ] Standard library dependencies validation
- [ ] Testing framework dependencies (pytest)
- [ ] Mock utilities availability
- [ ] Dependency version accessibility

### 3. Class Instantiation Tests

**File**: `test_smoke_instantiation.py` (10KB, 346 lines)

- [ ] AsyncMemFuse instantiation with default/custom parameters
- [ ] MemFuse instantiation with default/custom parameters
- [ ] Required API attributes validation (health, users, agents, etc.)
- [ ] Async context manager structure validation
- [ ] Sync context manager structure validation
- [ ] API client class instantiation tests
- [ ] Memory class instantiation tests
- [ ] LLM adapter instantiation tests
- [ ] Environment variable handling validation
- [ ] Default value assignments
- [ ] Client instance tracking
- [ ] String representation methods

### 4. Package Import Tests

**File**: `test_smoke_package_imports.py` (6.8KB, 260 lines)

- [ ] Main package import validation
- [ ] Package version accessibility
- [ ] Public API class imports from `__all__`
- [ ] `__all__` attribute validation
- [ ] Direct client class imports
- [ ] Direct memory class imports
- [ ] Direct API class imports
- [ ] LLM adapter class imports
- [ ] Utility function imports
- [ ] Model class imports

### 5. Python Compatibility Tests

**File**: `test_smoke_python_compatibility.py` (2.0KB, 65 lines)

- [ ] Python version compatibility (≥3.10)
- [ ] Version information accessibility
- [ ] Platform information access
- [ ] Dynamic import mechanism validation

---

## 🎯 Review Guidelines

### For Each Test File:

1. **Structural Review**:

   - Check test organization and naming conventions
   - Verify proper use of `@pytest.mark.smoke` decorators
   - Ensure tests are focused on structure, not behavior

2. **Mocking Strategy**:

   - Validate that external dependencies are properly mocked
   - Ensure no real network calls are made
   - Check that mocks are cleaned up properly

3. **Error Handling**:

   - Verify tests fail gracefully with clear error messages
   - Check that missing dependencies are handled appropriately
   - Ensure proper exception handling in test assertions

4. **Coverage Assessment**:
   - Confirm all public API surfaces are covered
   - Verify critical dependencies are tested
   - Check that both sync and async variants are tested

### Red Flags to Watch For:

- ❌ Real network calls in smoke tests
- ❌ Missing `@pytest.mark.smoke` decorators
- ❌ Tests that depend on external services
- ❌ Overly complex logic in structural tests
- ❌ Missing cleanup or improper mocking

---

## 📝 Review Notes

Add notes here as you review each file:

### test_smoke_basic.py ✅ COMPLETED

- **Reviewed by**: [Reviewer Name]
- **Date**: [Review Date]
- **Notes**: All basic connectivity and structure tests are properly implemented with appropriate mocking.

### test_smoke_dependencies.py

- **Status**: Pending Review
- **Notes**: [Add review notes here]

### test_smoke_instantiation.py

- **Status**: Pending Review
- **Notes**: [Add review notes here]

### test_smoke_package_imports.py

- **Status**: Pending Review
- **Notes**: [Add review notes here]

### test_smoke_python_compatibility.py

- **Status**: Pending Review
- **Notes**: [Add review notes here]

---

## ✅ Final Checklist

- [ ] All smoke tests pass locally
- [ ] No real network dependencies in any test
- [ ] Proper mocking strategy implemented throughout
- [ ] All public API surfaces covered by smoke tests
- [ ] Tests are focused on structure, not complex behavior
- [ ] Error messages are clear and actionable
- [ ] Both sync and async variants properly tested
- [ ] Documentation reflects current test coverage

---

## 🚀 Next Steps

After completing the smoke test review:

1. Run full smoke test suite: `pytest tests/smoke/ -v`
2. Address any identified issues or gaps
3. Move on to integration test review
4. Update this document with final review status
