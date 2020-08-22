from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, DefaultDict, Any, Dict, List, NamedTuple, Tuple
import dataclasses
from enum import Enum, auto


@dataclass
class Context:
    buffer: List[str]
    memory: Dict[int, Tuple[Any, ...]] = field(default_factory=dict)

    def evolve(self, **kwargs: Any):
        return dataclasses.replace(self, **kwargs)


class ActionParamSlot(Enum):
    MEM = auto()
    BUF = auto()
    UNARY_OP = auto()
    BINARY_OP = auto()


@dataclass(frozen=True)
class ActionParam:
    slot: ActionParamSlot
    value: Any


@dataclass(frozen=True)
class Action:
    params: List[ActionParam]
    op: Callable

    @property
    def name(self):
        return self.op.__name__

    def __call__(self, ctx: Context):
        vals = [p.value for p in self.params]
        return self.op(ctx, *vals)

    def __repr__(self):
        return f"Action<{self.name}{[p.value for p in self.params]}>"


OpType = str


def push(context: Context, buffer_loc: int, loc: int):
    if buffer_loc >= len(context.buffer):
        raise InvalidAction(f"push:{buffer_loc}:{loc}")

    mem: Tuple[Any, ...]
    if not loc in context.memory:
        mem = (buffer_loc,)
    elif len(context.memory[loc]) >= 3:
        raise InvalidAction(f"push:{buffer_loc}:{loc}")
    else:
        mem = context.memory[loc] + (buffer_loc,)
    new_mem = context.memory.copy()
    new_mem[loc] = mem
    return context.evolve(memory=new_mem)


class InvalidAction(Exception):
    pass


def op(context: Context, op_type: OpType, loc: int):
    if not loc in context.memory:
        raise InvalidAction(f"op:{op_type}:{loc}")

    mem = context.memory[loc]
    new_mem = context.memory.copy()
    new_mem[loc] = (op_type,) + mem

    return context.evolve(memory=new_mem)


def join(context: Context, op_type: OpType, source_loc: int, target_loc: int):
    if not (source_loc in context.memory and target_loc in context.memory):
        raise InvalidAction(f"join:{op_type}:{source_loc}:{target_loc}")

    mem_source = context.memory[source_loc]
    mem_target = context.memory[target_loc]

    new_mem = context.memory.copy()
    new_mem[source_loc] = (op_type, mem_source, mem_target)
    del new_mem[target_loc]
    return context.evolve(memory=new_mem)


def produce(context: Context, loc: int):
    if loc not in context.memory:
        raise InvalidAction()

    return context.memory[loc]
