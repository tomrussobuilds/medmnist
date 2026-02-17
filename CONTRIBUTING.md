# Contributing to Orchard ML

Thank you for considering contributing to Orchard ML!

## Project Direction

Project goals and roadmap are defined in the [main README](README.md). Before proposing new features, please review the stated objectives to ensure alignment with the project's vision.

**Contributions are welcome in these areas:**
- Bug fixes and behavior improvements
- Performance optimizations
- Test coverage enhancements
- Documentation improvements
- Code quality improvements

**For new features:** Please open an issue first to discuss alignment with project goals.

## Code Quality Standards

All contributions must maintain the project's quality standards:

### 1. Testing Requirements

- **All code changes must include tests**
- **Maintain 100% test coverage** (current standard)
- Tests must pass locally before submitting

```bash
# Run test suite
pytest tests/ -v

# Check coverage
pytest tests/ --cov=orchard --cov-report=term-missing
```

### 2. Quality Checks

Two quality check scripts are available:

```bash
# Quick checks (recommended during development)
bash scripts/check_quality.sh

# Extended checks (run before PRs)
bash scripts/check_quality_full.sh
```

**Quick checks** (`check_quality.sh`):
- Black (code formatting)
- isort (import sorting)
- Flake8 (linting)
- Bandit (security)
- Radon (complexity)
- Pytest with coverage

**Extended checks** (`check_quality_full.sh`) add:
- MyPy (type checking)
- HTML coverage report (`htmlcov/index.html`)

### 3. Code Style

- Follow existing code patterns and architecture
- Use type hints for all function signatures
- Write clear docstrings (Google style)
- Keep functions focused and testable

## Contribution Workflow

1. **Fork** the repository
2. **Create a feature branch** (`git checkout -b fix/issue-description`)
3. **Make your changes** with tests
4. **Run quality checks** (`bash scripts/check_quality_full.sh`)
5. **Commit** with clear messages
6. **Push** and open a Pull Request

## Questions?

Open an issue for:
- Feature proposals (discuss before implementing)
- Bug reports (include reproduction steps)
- Architecture questions
- Documentation clarifications

---

**Note:** This is a personal research project. Contributions are appreciated, but the maintainer reserves final decisions on features and direction.
