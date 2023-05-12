import platform
import re
import subprocess
from typing import Dict, Union

import cpuinfo
import psutil

from config.config import logger


def get_gpu_info() -> Dict[str, Union[str, int]]:
    """
    Get information about the available GPUs.

    Returns:
        Dict[str, Union[str, int]]: A dictionary containing GPU information with keys:
            - 'name': The name of the GPU.
            - 'memory.total': The total memory of the GPU.
            - 'memory.free': The free memory of the GPU.
            - 'memory.used': The used memory of the GPU.
    """
    try:
        command = "nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used --format=csv"
        output = subprocess.check_output(command, shell=True, universal_newlines=True)

        lines = output.strip().split("\n")
        alphanumeric_line = re.sub(r"[^a-zA-Z0-9\s,]", "", lines[0])
        header = [item.strip() for item in alphanumeric_line.split(",")]
        values = [item.strip() for item in lines[1].split(",")]
        gpu_info = {header[i]: values[i] for i in range(len(header))}

    except Exception as e:
        logger.info(e)
        gpu_info = {"name": "N/A", "memory.total": "N/A", "memory.free": "N/A", "memory.used": "N/A"}

    return gpu_info


def get_cpu_info() -> Dict[str, Union[str, int]]:
    """
    Get information about the CPU.

    Returns:
        Dict[str, Union[str, int]]: A dictionary containing CPU information with keys:
            - 'CPU/Name': The name of the CPU.
            - 'CPU/Cores': The number of physical cores.
            - 'CPU/Threads': The number of logical threads.
            - 'CPU/Architecture': The CPU architecture.
            - 'CPU/Frequency': The CPU frequency in a human-friendly format.
    """
    cpu_info = cpuinfo.get_cpu_info()
    return {
        "CPU/Name": cpu_info["brand_raw"],
        "CPU/Cores": psutil.cpu_count(logical=False),
        "CPU/Threads": psutil.cpu_count(logical=True),
        "CPU/Architecture": cpu_info["arch"],
        "CPU/Frequency": cpu_info["hz_actual_friendly"],
    }


def get_OS_info() -> Dict[str, Union[str, int]]:
    """
    Get information about the operating system.

    Returns:
        Dict[str, Union[str, int]]: A dictionary containing OS information with keys:
            - 'OS/OS': The name of the operating system.
            - 'OS/Release': The release version of the operating system.
            - 'OS/Version': The detailed version information of the operating system.
    """
    return {
        "OS/OS": platform.system(),
        "OS/Release": platform.release(),
        "OS/Version": platform.version(),
    }
