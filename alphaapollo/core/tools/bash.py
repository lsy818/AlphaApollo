import logging
import subprocess
import os
from typing import Dict, Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_TIMEOUT = 30

def check_dangerous_commands(command: str) -> bool:
    """Check for dangerous bash commands."""
    forbidden_keywords = [
        "rm -rf", "sudo ", "mkfs", "dd ", "reboot", "shutdown", 
        "chmod -R 777", "chown -R", "iptables", "docker", "wget ", "curl "
    ]
    # Simple check for now
    lower_command = command.lower()
    for kw in forbidden_keywords:
        if kw in lower_command:
            return True
    return False

def execute_bash_command(
    command: str,
    timeout: int = DEFAULT_TIMEOUT,
    log_requests: bool = True,
) -> Dict[str, Any]:
    """
    Execute Bash command locally using subprocess.
    
    Args:
        command: The bash command to execute.
        timeout: The timeout for code execution in seconds.
        log_requests: Whether to log execution details.
    
    Returns:
        A dictionary containing execution results:
        {
            "stdout": str,
            "stderr": str,
            "returncode": int,
            "run_status": str  # "Finished", "Timeout", or "Error"
        }
    """
    if not command or not command.strip():
        return {
            "stdout": "",
            "stderr": "No command provided.",
            "returncode": -1,
            "run_status": "Error"
        }
    
    if log_requests:
        logger.info(f"Executing Bash command locally (timeout: {timeout}s)")
        logger.debug(f"Command to execute:\n{command[:200]}...")
    
    if check_dangerous_commands(command):
        return {
            "stdout": "",
            "stderr": "Forbidden or dangerous command detected. Execution rejected.",
            "returncode": -1,
            "run_status": "Error"
        }
    
    try:
        # Use subprocess to run bash explicitly
        result = subprocess.run(
            ["/bin/bash", "-c", command],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=None,
            env=os.environ.copy()
        )
        
        run_status = "Finished" if result.returncode == 0 else "Error"
        
        if log_requests:
            if result.returncode == 0:
                logger.info(f"Bash execution successful (returncode: {result.returncode})")
            else:
                logger.warning(f"Bash execution failed (returncode: {result.returncode})")
                if result.stderr:
                    logger.debug(f"Stderr: {result.stderr[:500]}")
        
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "run_status": run_status
        }
        
    except subprocess.TimeoutExpired:
        if log_requests:
            logger.warning(f"Bash execution timed out after {timeout} seconds")
        return {
            "stdout": "",
            "stderr": f"Bash execution timed out after {timeout} seconds",
            "returncode": -1,
            "run_status": "Timeout"
        }
    except Exception as e:
        if log_requests:
            logger.error(f"Error executing bash command: {e}")
        return {
            "stdout": "",
            "stderr": f"Exception during bash execution: {str(e)}",
            "returncode": -1,
            "run_status": "Error"
        }