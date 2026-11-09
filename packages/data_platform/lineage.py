"""Data Lineage: DAG edge table from raw source -> fact -> metric -> evidence -> claim."""
from __future__ import annotations

from collections import deque
from typing import Optional

from sqlalchemy import select

from .db import LineageEdgeRow, get_session_factory
from .schema import LineageEdge


class LineageRecorder:
    def __init__(self, db_url: Optional[str] = None, run_id: Optional[str] = None):
        self.Session = get_session_factory(db_url)
        self.run_id = run_id
        self.buffer: list[LineageEdge] = []
        self._seen: set[tuple[str, str]] = set()

    def add(self, parent: str, child: str, parent_type: str, child_type: str, tool: str, version: str = "v1") -> None:
        if (parent, child) in self._seen:
            return
        self._seen.add((parent, child))
        self.buffer.append(LineageEdge(parent_entity_id=parent, child_entity_id=child, parent_type=parent_type,
                                       child_type=child_type, transform_tool=tool, formula_version=version,
                                       run_id=self.run_id))

    def flush(self) -> int:
        if not self.buffer:
            return 0
        with self.Session() as s:
            s.add_all(LineageEdgeRow(**e.model_dump()) for e in self.buffer)
            s.commit()
        n = len(self.buffer)
        self.buffer = []
        return n

    def upstream(self, entity_id: str, max_depth: int = 8) -> dict:
        """Walk parents (buffered + persisted) back to raw sources. Returns nodes and edges."""
        edges: list[dict] = []
        nodes = {entity_id}
        q = deque([(entity_id, 0)])
        with self.Session() as s:
            while q:
                cur, depth = q.popleft()
                if depth >= max_depth:
                    continue
                parents = [e.model_dump() for e in self.buffer if e.child_entity_id == cur]
                stmt = select(LineageEdgeRow).where(LineageEdgeRow.child_entity_id == cur)
                if self.run_id:
                    stmt = stmt.where((LineageEdgeRow.run_id == self.run_id) | (LineageEdgeRow.run_id.is_(None)))
                for row in s.scalars(stmt):
                    parents.append({"parent_entity_id": row.parent_entity_id, "child_entity_id": row.child_entity_id,
                                    "parent_type": row.parent_type, "child_type": row.child_type,
                                    "transform_tool": row.transform_tool, "formula_version": row.formula_version})
                for e in parents:
                    key = (e["parent_entity_id"], e["child_entity_id"])
                    if any((x["parent_entity_id"], x["child_entity_id"]) == key for x in edges):
                        continue
                    edges.append(e)
                    if e["parent_entity_id"] not in nodes:
                        nodes.add(e["parent_entity_id"])
                        q.append((e["parent_entity_id"], depth + 1))
        return {"root": entity_id, "nodes": sorted(nodes), "edges": edges}
