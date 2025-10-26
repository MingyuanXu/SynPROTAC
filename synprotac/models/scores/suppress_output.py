import os
import sys
from contextlib import contextmanager

@contextmanager
def suppress_output():
    """在 Unix 上同时屏蔽 Python 层和 C 层的 stdout/stderr。"""
    devnull = os.open(os.devnull, os.O_RDWR)
    # 保存原 fd
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)
    try:
        # 重定向 stdout/stderr 到 /dev/null
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        # 恢复原来的 fds
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        os.close(devnull)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)
