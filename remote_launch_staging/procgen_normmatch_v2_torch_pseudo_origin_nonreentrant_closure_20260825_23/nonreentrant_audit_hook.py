#!/usr/bin/env python3
"""Minimal, non-reentrant audit recorder for first-level import/filesystem events."""
import os
import sys
import threading


_EVENTS = frozenset({
    "import", "open", "os.mkdir", "os.rename", "os.remove", "os.rmdir",
    "os.listdir", "os.scandir",
})
_SAFE_TYPES = (str, bytes, int, float, bool, type(None))


class NonReentrantAuditRecorder:
    def __init__(self, frame_limit=24):
        self._local = threading.local()
        self._getframe = sys._getframe
        self._getpid = os.getpid
        self._gettid = threading.get_ident
        self._frame_limit = frame_limit
        self.events = []
        self.reentrant_total = 0
        self.reentrant_by_event = {}

    def __call__(self, event, args):
        if event not in _EVENTS:
            return
        if getattr(self._local, "active", False):
            self.reentrant_total += 1
            self.reentrant_by_event[event] = self.reentrant_by_event.get(event, 0) + 1
            return
        self._local.active = True
        try:
            safe_args = []
            for value in args:
                if isinstance(value, _SAFE_TYPES):
                    safe_args.append(value)
                elif isinstance(value, tuple):
                    safe_args.append(tuple(item for item in value if isinstance(item, _SAFE_TYPES)))
                else:
                    safe_args.append(None)
            frames = []
            frame = self._getframe(1)
            count = 0
            while frame is not None and count < self._frame_limit:
                code = frame.f_code
                frames.append({
                    "file": code.co_filename,
                    "name": code.co_name,
                    "line": frame.f_lineno,
                })
                frame = frame.f_back
                count += 1
            self.events.append({
                "event": event,
                "args": safe_args,
                "pid": self._getpid(),
                "tid": self._gettid(),
                "frames": frames,
            })
        finally:
            self._local.active = False

    def ledger(self):
        return {
            "result": "NONREENTRANT_AUDIT_HOOK_PASS" if self.reentrant_total == 0 else "REENTRANT_EVENTS_OBSERVED",
            "first_level_event_count": len(self.events),
            "reentrant_total": self.reentrant_total,
            "reentrant_by_event": dict(sorted(self.reentrant_by_event.items())),
        }
