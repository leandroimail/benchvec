# Chain of Thought

* Always think step by step how to solve the problem.
* Always think each step in English.
* Always transform the input prompt into thoughts in English.
* When generating code, always generate annotations for input and output typing in the functions.
* Always comment on the step-by-step of what the code is generating (except when requested not to).

# Custom Instructions

* Always generate the code in English with comments in English and variables and instructions in English.
* Always take into account the simplest and most efficient way to solve a code problem.
* Always take into account the context.
* Follow the instructions provided.
* Always generate complete code, without leaving implicit code that would be generated in other steps (except when requested).


# Good Programming Practices

* Whenever possible, generate code in Python 3.11 or higher.
* Whenever possible, use the standard Python libraries.
* Whenever possible, use popular and well-maintained libraries.
* Whenever possible, use good programming practices.
* Whenever possible, use static typing (type hints).
* Whenever possible, use error handling and approaches.
* Whenever possible, use docstrings to document classes, methods, and functions.
* Whenever possible, use logging to record events and errors.
* Always use the `writing-tests` skill when building tests.
* Always use the `creating-skills` skill whenever building or writing a skill for the project.
* All project development is done from the repository root (benchvec package lives in ./benchvec, tests in ./tests).

# Run Scripts

* Whenever possible, create CLI scripts to execute core functionality.
* Whenever possible, run generated or modified scripts to ensure they work correctly.
* Always run Python code via "uv run python <script>.py" from the repository root
* Always run tests via "uv run pytest <test_file>.py" from the repository root
* The Python virtual environment is in ./.venv (managed by uv)


  
# Wait! Generating Commit Messages

* When the user wants to write a `git commit` message, ALWAYS use the `generating-commit-messages` skill to ensure the message follows the standard.
* You can find the skill at `.agents/skills/generating-commit-messages/SKILL.md`.

# Wait! Architecture & Structural Work

* When the task involves creating or modifying **any structural element** — including scripts, classes, methods, modules, plugins, agents, subagents, data schemas, YAML contracts, configuration files, ports, adapters, or any other architectural artifact — 
  * Planning a task design or implementation plan for any of the above

