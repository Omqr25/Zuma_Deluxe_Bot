from dataclasses import dataclass


@dataclass(frozen=True)
class Window:
    x: int
    y: int
    w: int
    h: int
    
    