# Contributing to brainpy.state

Thank you for your interest in contributing to brainpy.state! We welcome contributions from the community and are grateful for your support in making brainpy.state better.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Contribution Workflow](#contribution-workflow)
- [Coding Guidelines](#coding-guidelines)
- [Testing](#testing)
- [Documentation](#documentation)
- [Community](#community)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to chao.brain@qq.com.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the problem
- **Expected vs actual behavior**
- **brainpy.state version** and environment details (Python version, OS, JAX version)
- **Code samples** or test cases that demonstrate the issue
- **Error messages** and stack traces

Submit bug reports via [GitHub Issues](https://github.com/chaobrain/brainpy.state/issues/new).

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Clear use case** - explain the problem you're trying to solve
- **Proposed solution** - describe how you envision the feature working
- **Alternative approaches** you've considered
- **Impact** - who would benefit and how

### Contributing Code

We welcome code contributions! This includes:

- Bug fixes
- New features
- Performance improvements
- Documentation improvements
- Test coverage improvements

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git
- pip or conda

### Setting Up Your Environment

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/brainpy.state.git
   cd brainpy.state
   ```

3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install development dependencies**:
   ```bash
   pip install -e ".[dev,cpu]"  # Use [dev,cuda12] for CUDA support
   ```

5. **Set up pre-commit hooks** (if available):
   ```bash
   pre-commit install
   ```

## Contribution Workflow

### 1. Create a Branch

Create a new branch for your work:
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

Branch naming conventions:
- `feature/` - new features
- `fix/` - bug fixes
- `docs/` - documentation changes
- `refactor/` - code refactoring
- `test/` - test improvements

### 2. Make Your Changes

- Write clean, readable code
- Follow the coding guidelines below
- Add tests for new functionality
- Update documentation as needed
- Keep commits focused and atomic

### 3. Test Your Changes

Run the test suite:
```bash
pytest tests/
```

Run specific tests:
```bash
pytest tests/test_specific.py -v
```

### 4. Commit Your Changes

Write clear, descriptive commit messages:
```bash
git commit -m "Fix issue with delayed connection handling (#123)

- Add validation for delay parameters
- Update tests to cover edge cases
- Fix documentation example
"
```

### 5. Push and Create a Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with:
- **Clear title** describing the change
- **Description** explaining what and why
- **Reference to related issues** (e.g., "Fixes #123")
- **Test results** or screenshots if applicable

## Coding Guidelines

### Python Style

- Follow [PEP 8](https://pep8.org/) style guide
- Use 2 or 4 spaces for indentation (be consistent with surrounding code)
- Maximum line length: 100-120 characters
- Use descriptive variable and function names

### Code Structure

- Keep functions focused and single-purpose
- Use type hints where appropriate
- Add docstrings for public APIs (follow NumPy docstring format)
- Avoid unnecessary complexity

### Example Docstring

```python
def simulate_network(network, duration, dt=0.1):
    """Simulate a neural network for a specified duration.

    Parameters
    ----------
    network : Network
        The neural network to simulate
    duration : float
        Total simulation time in milliseconds
    dt : float, optional
        Time step in milliseconds (default: 0.1)

    Returns
    -------
    results : dict
        Simulation results containing spike times and state variables

    Examples
    --------
    >>> net = Network()
    >>> results = simulate_network(net, duration=1000.0)
    """
    pass
```

## Testing

### Writing Tests

- Place tests in the `tests/` directory
- Use descriptive test function names: `test_feature_does_what_expected`
- Test edge cases and error conditions
- Use fixtures for common setups

### Test Coverage

Aim for high test coverage on new code:
```bash
pytest --cov=brainpy_state brainpy_state/
```

## Documentation

### Updating Documentation

- Update relevant documentation when changing features
- Add examples for new functionality
- Keep the API reference up to date
- Fix typos and improve clarity

### Documenting new APIs

Every new public class or function must ship with documentation:

- Use **NumPy-style docstrings** with `Parameters`, `Returns` (or `Yields`),
  and at least one `Examples` block. The project's Sphinx config sets
  `napoleon_numpy_docstring = True` and registers many custom sections
  (e.g. `State Variables`, `Mathematical Model`, `Parameter Mapping`,
  `Implementation Notes`) — use them where they help readers.
- **BrainPy-style models** — add an `autosummary` entry to the appropriate
  page under `docs/api/brainpy-*.rst`.
- **NEST-compatible models** — add an `autosummary` entry to the appropriate
  page under `docs/api/nest-*.rst`. Parameter names should match the upstream
  NEST documentation. Document any deliberate deviation in the docstring
  under a `Parameter Mapping` or `Implementation Notes` section.
- For any new model file, also add a colocated `*_test.py` that exercises
  default parameters and at least one parameter sweep.
- Export the new symbol in `brainpy_state/__init__.py` (`__all__` plus the
  import) so it appears in the public API surface.

### Marking experimental features

`brainpy.state` ships two model families with different maturity levels — see
the README's "API status and maturity" section.

- **NEST-compatible models** (anything under `brainpy_state/_nest/`) are
  collectively considered **Experimental — In Development**. Individual
  models do not need their own banner; the family-level banner on
  `docs/api/nest-*.rst` and the [NEST status page](docs/nest-status/index.rst)
  cover the whole set.
- **BrainPy-style features** that are not yet stable must include a Sphinx
  `.. warning::` admonition at the top of the docstring labelled
  **Experimental — In Development**, and must be omitted from the README
  maturity table until promoted to Stable.
- **Promoting a feature from Experimental to Stable** requires:
  1. Full test coverage including parameter validation.
  2. Removal of the warning admonition from the docstring.
  3. Addition to the README maturity table.
  4. A CHANGELOG entry under "API stability".
- When in doubt, ship as Experimental. Promotion is cheap; demoting silently
  breaks users.

### Documentation review checklist for PRs

Before requesting review, confirm:

- [ ] Docstrings present on all new public APIs (NumPy style).
- [ ] New module exported via `__all__` in `brainpy_state/__init__.py`.
- [ ] Added to the corresponding `docs/api/*.rst` page.
- [ ] If NEST-compatible: parameter names match the upstream NEST documentation.
- [ ] If experimental: no claims of stability in docstrings or README.
- [ ] `pytest brainpy_state/` passes locally.
- [ ] Any executable code block in updated docs or notebooks runs
      end-to-end without errors.

For the full rendered documentation, see <https://brainx.chaobrain.com/brainpy-state/>.

## Community

### Getting Help

- **Documentation**: https://brainx.chaobrain.com/brainpy-state/
- **GitHub Discussions**: https://github.com/chaobrain/brainpy.state/discussions
- **Issues**: https://github.com/chaobrain/brainpy.state/issues

### Recognition

Contributors are recognized in:
- Release notes
- Contributors file
- Project documentation

Thank you for contributing to brainpy.state! Your efforts help make computational neuroscience more accessible and powerful for everyone.
