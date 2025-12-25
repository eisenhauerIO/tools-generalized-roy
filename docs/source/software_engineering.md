# Software Engineering

We now briefly discuss our software engineering practices that help us to ensure the transparency, reliability, scalability, and extensibility of the `grmpy` package.

## Test Battery

We use [pytest](http://docs.pytest.org) as our test runner. We broadly group our tests in three categories:

### Property-based Testing

We create random model parameterizations and estimation requests and test for a valid return of the program.

### Reliability Testing

We conduct numerous Monte Carlo exercises to ensure that we can recover the true underlying parameterization with an estimation.

### Regression Testing

We provide a regression test. For this purpose we generated random model parameterizations, simulated the corresponding outputs, and saved them. This ensures that the package works accurately even after an update to a new version.

## Documentation

The documentation is created using [Sphinx](http://www.sphinx-doc.org/) and hosted on GitHub Pages.

## Code Review

We use automatic code review tools like [ruff](https://github.com/astral-sh/ruff) to help us improve the readability and maintainability of our code base.

## Continuous Integration Workflow

We set up a continuous integration workflow around our [GitHub Organization](https://github.com/OpenSourceEconomics). We use GitHub Actions for continuous integration.
