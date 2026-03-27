# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from alphaapollo.core.environments.prompts.informal_math_evolving import *
from alphaapollo.core.environments.prompts.informal_math_training import *
import re

def filter_prompt_tools(template: str, tool_config: dict) -> str:
    """Dynamically remove disabled tools from prompt templates and renumber the list."""
    if tool_config is None:
        return template
        
    enable_python_code = tool_config.get("enable_python_code", True)
    enable_local_rag = tool_config.get("enable_local_rag", True)
    enable_bash = tool_config.get("enable_bash", True)
    
    lines = template.split('\n')
    filtered_lines = []
    
    for line in lines:
        if '<python_code>' in line and not enable_python_code:
            continue
        if '<local_rag>' in line and not enable_local_rag:
            continue
        if '<bash>' in line and not enable_bash:
            continue
        filtered_lines.append(line)
        
    final_lines = []
    counter = 1
    for line in filtered_lines:
        match = re.match(r'^(\s*)(\d+)([\.\)])\s*(<[a-z_]+>.*)', line)
        if match:
            indent, _, punctuation, rest = match.groups()
            final_lines.append(f"{indent}{counter}{punctuation} {rest}")
            counter += 1
        else:
            final_lines.append(line)
            
    return '\n'.join(final_lines)
