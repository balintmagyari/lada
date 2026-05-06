# Contributing to LaDa

First off, thank you for considering contributing to LaDa! It is people like you who make this tool better for everyone.

This document outlines the process for contributing to the project. By participating, you are expected to uphold our code of conduct and follow these guidelines to ensure a smooth collaboration.

## Table of Contents
1. [How to Contribute](#how-to-contribute)
2. [Setting Up Your Environment](#setting-up-your-environment)
3. [Coding Standards](#coding-standards)
4. [Pull Request Process](#pull-request-process)

---

## How to Contribute

We use the standard GitHub Fork and Pull Request workflow. Please do not request direct push access to the main repository. 

1. **Fork the repository** to your own GitHub account.
2. **Clone your fork** locally to your machine:
   `git clone https://github.com/YOUR-USERNAME/[Repository-Name].git`
3. **Create a new branch** for your feature or bug fix:
   `git checkout -b feature/your-feature-name` (or `bugfix/issue-description`)
4. **Commit your changes** with clear, descriptive commit messages.
5. **Push the branch** back to your forked repository.

---

## Running Tests Locally

Before submitting a pull request, ensure all tests pass:

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run tests for a specific module
pytest tests/test_parsers/ -v
pytest tests/test_analysis/ -v
```

Tests are automatically run on pull requests via GitHub Actions. Your PR must pass all tests before merging.

### Test Structure

Tests are organized by module:
- `tests/test_parsers/` — Tests for dump, log, and data file parsers
- `tests/test_analysis/` — Tests for analysis calculations

Please add tests for any new functionality.