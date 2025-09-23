import dataclasses
import json
from typing import Any

from torch.nn import Module


class JsonEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        if isinstance(o, type):
            return o.__name__
