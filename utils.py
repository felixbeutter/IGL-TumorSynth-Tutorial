import shlex
import subprocess
from datetime import datetime
from typing import Optional
from pathlib import Path

def log(msg: str, timestamp: bool = True, newline: bool = True) -> None:
    """
    Utility: Prints a message to stdout with an optional timestamp.

    Args:
        msg: String message to print.
        timestamp: Whether to prefix the message with a timestamp (default: True).
        newline: Whether to end the print with a newline (default: True).

    Returns:
        None
    """
    # Format and prefix log message with current date/time if requested.
    time_prefix = f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] ' if timestamp else ''
    # Output to stdout, flushing immediately to support real-time pipeline monitoring.
    print(time_prefix + msg, end='\n' if newline else '', flush=True)

class CmdResult:
    """
    Utility: Representation of the result of a subprocess command execution.

    Args:
        cmd: The command string that was executed.
        returncode: The return code of the command process.
        stdout: Standard output text from the command.
        stderr: Standard error text from the command.
    """
    def __init__(self, cmd: str, returncode: int, stdout: str, stderr: str):
        self.cmd = cmd
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr

def run_cmd(
    cmd: list[str], 
    cwd: Optional[Path] = None, 
    env: Optional[dict] = None,
    print_cmd: bool = False
) -> CmdResult:
    """
    Utility: Executes an external command via subprocess and captures its output.

    Args:
        cmd: List of command arguments to run.
        cwd: Working directory where the command should execute (default: None).
        env: Environment variables dictionary to pass (default: None).
        print_cmd: If True, prints the command execution string before running (default: False).

    Returns:
        CmdResult: Object containing command, exit code, stdout, and stderr.
    """
    # Format the command parameters as a safe shell command string for printing or debugging.
    cmd_str = ' '.join(shlex.quote(c) for c in cmd)
    
    if print_cmd:
        log(f'Running: {cmd_str}')
    
    # Execute the external command, capturing stdout and stderr into memory buffers.
    proc = subprocess.run(
        cmd, 
        cwd=str(cwd) if cwd else None, 
        env=env, 
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, 
        text=True, 
        check=False
    )
    
    # Package the result parameters into our helper CmdResult class.
    res = CmdResult(cmd=cmd_str, returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)

    # Check for execution failure and raise a detailed exception with stdout/stderr.
    if proc.returncode != 0:
        raise RuntimeError(
            f'Command failed (code={proc.returncode}):\n'
            f'{cmd_str}\n\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}'
        )
    
    return res
