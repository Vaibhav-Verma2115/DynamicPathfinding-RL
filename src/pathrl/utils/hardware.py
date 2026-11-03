"""Hardware and software disclosure, saved alongside every result file.

Wall-clock times are only meaningful together with the machine that produced
them, so each experiment records this snapshot.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime, timezone


def hardware_info() -> dict:
    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
    }
    try:
        import numpy as np
        import torch

        info["numpy"] = np.__version__
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        info["mps_available"] = torch.backends.mps.is_available()
        info["torch_threads"] = torch.get_num_threads()
    except ImportError:
        info["torch"] = None
    return info


if __name__ == "__main__":
    print(json.dumps(hardware_info(), indent=2))
