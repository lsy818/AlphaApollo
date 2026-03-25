# AlphaApollo Bash Tool Expansion Mini-Project

## Overview of Core Changes

To enable bash command execution (similar to OpenClaw) within the AlphaApollo environment, we extended the existing `InformalMathToolGroup` tool system and added support for parsing `<bash>...</bash>` tags.

### 1. New File
- **`alphaapollo/core/tools/bash.py`**:
  - Implements `execute_bash_command`, a dedicated function for safe terminal command execution.
  - **Safety mechanism design choices**:
    - **Timeout**: Prevents deadlocks from infinite loops or interactive commands that require user input; defaults to a 30-second timeout.
    - **Process isolation**: All terminal commands run via `subprocess.run(["/bin/bash", "-c", command])` in isolated subprocesses.
    - **Dangerous command blocklist**: Before handing any command to `subprocess`, a keyword scan is performed (covering `rm -rf`, `sudo`, `reboot`, `wget`, `curl`, etc.). This ensures that even if the Agent develops harmful intent or is adversarially prompted, the tool layer proactively rejects destructive instructions.

### 2. Modified Existing Files
- **`alphaapollo/core/tools/manager.py`**:
  - Added `@tool def bash(self, command: str)` to `InformalMathToolGroup`, serving as the Agent-side entry point. Execution results are standardized into `<tool_response>...</tool_response>` format.
- **`alphaapollo/core/environments/informal_math_training/env.py`** and **`_evolving/env.py`**:
  - Added `<bash>` to `TOOL_PATTERNS` and routing logic (`_parse_action` and `step()`), enabling the environment to correctly extract model-generated bash scripts and dispatch them to the manager for execution.
- **`alphaapollo/core/environments/prompts/*.py`** (System Prompts):
  - Appended `<bash>` usage instructions to all policy agent prompt templates: `"2. <bash>...</bash>: If system/file operations or bash utilities (e.g., bc, grep) are helpful, emit exactly ONE <bash>...</bash> block containing a pure bash command. Inspect the <tool_response> (stdout/stderr). If it fails or disagrees with your reasoning, correct yourself."` This informs the language model about the new tool's capabilities.

## Getting Started

Follow these steps to reproduce the Bash tool tests and demos from scratch:

### 1. Environment Setup
```bash
conda create -n alphaapollo python==3.12 -y
conda activate alphaapollo

git clone https://github.com/tmlr-group/AlphaApollo.git
cd AlphaApollo

bash installation.sh
```

### 2. Running the Demo
To quickly verify the Bash engine's effectiveness and mixed-tool usage, we provide `demo.py` in the project root. It runs **five independent test cases** through `InformalMathTrainingEnv` without requiring a running LLM service — predefined Agent text actions are fed directly into the environment.

```bash
conda run -n alphaapollo python demo.py
```

The five test cases are:

| Case | Type | Description |
|------|------|-------------|
| **Case 1** | Backward Compatibility | `python_code` only — compute the sum of the first 100 positive integers |
| **Case 2** | Backward Compatibility | `python_code` only — compute C(20, 5) using `math.comb` |
| **Case 3** | Mixed Usage #1 | `python_code` → `bash` — compute 2^10 + 3^5 with Python, then cross-verify with `bc` |
| **Case 4** | Mixed Usage #2 | `bash` → `python_code` — compute 17×23 + 11×13 with `bc`, then verify with Python |
| **Case 5** | Bash-Only | Enumerate all prime numbers below 50 using a pure bash script, then count them |

### 3. Running Full Evaluation (Optional)
For complete reasoning-verification tasks such as AIME 2024, use the original workflow:
```bash
python3 -m alphaapollo.workflows.test \
  --model.path=Qwen/Qwen2.5-7B-Instruct \
  --preprocess.data_source=math-ai/aime24 \
  --env.informal_math.enable_python_code=true \
  --env.informal_math.enable_bash=true \
  --env.max_steps=4
```

## Reproduction Results Summary

### Task A — Baseline Tool Environment (Qwen2.5-7B-Instruct)
- **Status**: Routing and execution work correctly under the original `python_code` configuration.
- **Tool execution flow**:
  1. `env.py` obtains text predictions from the LLM.
  2. `env.py#_parse_action()` uses regex to match tags (e.g., `<python_code>`) and packages them as `(tool_name, payload)`, while detecting the `<answer>` termination signal.
  3. `env.py#step()` forwards the request via `super()._execute_tool(...)` to the corresponding tool method in `manager.py`.
  4. After execution, results are wrapped in `\n<tool_response>...</tool_response>\n` and appended to `chat_history` for the next turn. Prompt templates in the `prompts/` directory guide the model's generation format.

### Task B — Bash Integration Test Results (Demo Cases)
- **Cases 1 & 2 (Backward Compatibility)**:
  Pure `python_code` actions are parsed, executed, and return correct results (5050 and 15504 respectively), confirming that the bash integration does not break existing functionality.

- **Cases 3 & 4 (Mixed Usage)**:
  Both tool types (`python_code` and `bash`) are correctly parsed and dispatched within the same trajectory. Cross-verification between tools succeeds, demonstrating seamless interleaving in both `python → bash` and `bash → python` orders.

- **Case 5 (Bash-Only)**:
  A multi-line bash script with loops and conditionals enumerates primes below 50, returning `Count: 15`. This proves that bash can serve as a standalone computational tool without any Python dependency.
