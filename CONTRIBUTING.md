# Contributing to LLMuxer

Thank you for your interest in contributing to LLMuxer! We welcome contributions from the community.

## How to Contribute

### Reporting Issues

If you find a bug or have a feature request, please open an issue on GitHub:
1. Check if the issue already exists
2. Create a new issue with a clear title and description
3. Include steps to reproduce (for bugs)
4. Add relevant labels

### Submitting Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Install dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```
3. **Make your changes**:
   - Write clean, readable code
   - Follow PEP 8 style guidelines
   - Add docstrings to functions and classes
   - Update tests if needed

4. **Test your changes**:
   ```bash
   pytest tests/
   ```

5. **Commit your changes**:
   - Use clear, descriptive commit messages
   - Reference issue numbers when applicable

6. **Push to your fork** and submit a pull request

### Pull Request Guidelines

- Keep PRs focused on a single feature or fix
- Update documentation if needed
- Ensure all tests pass
- Add tests for new functionality
- Update the CHANGELOG.md if applicable

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/mihirahuja/llmuxer.git
   cd llmuxer
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

4. Set up pre-commit hooks (optional):
   ```bash
   pre-commit install
   ```

## Code Style

- Follow PEP 8
- Use type hints where appropriate
- Maximum line length: 100 characters
- Use descriptive variable names

## Testing

- Write tests for new features
- Ensure existing tests pass
- Aim for good test coverage
- Use pytest for testing

## Documentation

- Update README.md for user-facing changes
- Add docstrings to all public functions
- Include examples in docstrings
- Update API documentation

## Areas for Contribution

We especially welcome contributions in these areas:

- **New model providers**: Add support for more LLM providers
- **Task types**: Implement extraction, generation, and binary tasks
- **Performance**: Optimize evaluation speed
- **Testing**: Improve test coverage
- **Documentation**: Enhance examples and guides
- **Bug fixes**: Help us squash bugs

## Questions?

Feel free to open a discussion on GitHub or reach out via email at mihirahuja09@gmail.com.

Thank you for contributing to LLMuxer!