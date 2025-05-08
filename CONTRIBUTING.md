# Contribution Guidelines

Instructions and guidelines for how you can contribute to this repository.

## Issues

For reporting bugs, requesting new features and more do so by [creating a new issue][new-issue]. Make sure that:
- The issue clearly states expected vs current behavior
- Includes steps to reproduce bugs
- Look through the [existing issues][existing-issues] to see if your issue or suggestion has already been reported, if it has, add your own context to that issue instead of creating a new one.

## Development

How you can contribute code to this repository yourself.

### Requirements

> We recommend using a Python version and virtual environment manager, such as [asdf][asdf] or [pyenv][pyenv].

- [Git][git]
- [Python][python], version >= 3.11

### Install Dependencies

To install all the dependencies and test/dev dependencies do:
```sh
pip install ."[tests]"
```

> This installs [pre-commit][pre-commit] to enforce consistent formatting using [ruff][ruff]. Remember to activate it to automatically format code locally on all commits with:
> ```sh
> pre-commit install
> ```

#### Updating / Adding Dependencies

Dependencies and their versions are managed by the codeowners based on external requirements.

### Running Tests

Ensure everyting is working as expected by running all the tests with:
```sh
pytest
```

<details>
<summary>How can i see the test coverage?</summary>

Add the `--cov=rock_physics_open` argument to `pytest` to see the coverage across this project:
```sh
pytest --cov=rock_physics_open
```

</details>

<details>
<summary>Im getting <code>No module named '_tkinter'</code> error when running the tests?</summary>

You can skip these graphics tests by doing:
```sh
pytest -m "not use_graphics"
```

</details>

### Pull-Requests

- should only solve a single problem/issue
- can be in the form of a single or multiple commits
- must always be up to date with the `main` branch when:
  - the PR is created
  - before the PR is merged
- must explain **why** and **how** of the proposed changes in the PR description.
- ensure the commit history is clean

#### Commits

- Use [conventional commit messages][conventional-commits], as they are used to generate the changelog and perform releases using [release please][release-please].
- A single commit should only contain a *single logical change*
  - Commit should be as small as possible
  - Commit should contain a complete change

#### Review

A PR must be approved by a *Code Owner*. Once a PR is marked as ready for review _and_ passes all required checks a code owner will look through the PR as soon as they are able to.

If the PR is approved you can either `squash and merge` or `rebase and merge` it with main depending on commits/changes in the PR. If the reviewer had comments, or the request is not approved, address and resolve these comments before re-requesting a review.

<!-- External Links -->
[new-issue]: https://github.com/equinor/rock-physics-open/issues/new/choose
[existing-issues]: https://github.com/equinor/rock-physics-open/issues?q=is%3Aissue
[release-please]: https://github.com/googleapis/release-please?tab=readme-ov-file#release-please
[conventional-commits]: https://www.conventionalcommits.org/en/v1.0.0/

[git]: https://git-scm.com
[python]: https://www.python.org/
[asdf]: https://asdf-vm.com
[pyenv]: https://github.com/pyenv/pyenv
[pre-commit]: https://pre-commit.com
[ruff]: https://docs.astral.sh/ruff/
