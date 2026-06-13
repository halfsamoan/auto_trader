"""Minimal behavior tree node primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class NodeStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RUNNING = "RUNNING"


@dataclass
class TickContext:
    data: dict
    path: list[str] = field(default_factory=list)


class Node:
    def __init__(self, name: str) -> None:
        self.name = name

    def tick(self, context: TickContext) -> NodeStatus:
        raise NotImplementedError


class Selector(Node):
    def __init__(self, name: str, children: list[Node]) -> None:
        super().__init__(name)
        self.children = children

    def tick(self, context: TickContext) -> NodeStatus:
        context.path.append(self.name)
        for child in self.children:
            status = child.tick(context)
            if status in {NodeStatus.SUCCESS, NodeStatus.RUNNING}:
                return status
        return NodeStatus.FAILURE


class Sequence(Node):
    def __init__(self, name: str, children: list[Node]) -> None:
        super().__init__(name)
        self.children = children

    def tick(self, context: TickContext) -> NodeStatus:
        context.path.append(self.name)
        for child in self.children:
            status = child.tick(context)
            if status != NodeStatus.SUCCESS:
                return status
        return NodeStatus.SUCCESS


class ConditionNode(Node):
    def __init__(self, name: str, predicate: Callable[[dict], bool]) -> None:
        super().__init__(name)
        self.predicate = predicate

    def tick(self, context: TickContext) -> NodeStatus:
        context.path.append(self.name)
        return NodeStatus.SUCCESS if self.predicate(context.data) else NodeStatus.FAILURE


class ActionNode(Node):
    def __init__(self, name: str, action: Callable[[dict], None]) -> None:
        super().__init__(name)
        self.action = action

    def tick(self, context: TickContext) -> NodeStatus:
        context.path.append(self.name)
        self.action(context.data)
        return NodeStatus.SUCCESS
