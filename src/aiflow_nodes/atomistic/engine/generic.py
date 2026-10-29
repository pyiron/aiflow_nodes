from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OutputEngine:
    calculator: callable | None = field(default=None)
    engine_id: int | None = field(default=None)
    # parameters: Optional[wf_data_class] = field(
    #     default=None, metadata=wfMetaData(log_level=10)
    # )
    _do_not_serialize: bool = True
