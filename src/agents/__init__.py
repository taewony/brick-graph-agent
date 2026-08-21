"""brick.agent — role entry points and behavior wrappers.

Each role lives under `src/agents/<role>/` and is a thin layer over the
reusable `src/core/` packages: SQL re-wraps `src.core.targets.sql.agent`,
OKF wraps `src.core.targets.okf` (Phase 2) with event-driven behaviors
(Phase 3), and Mentor re-wraps `src.core.loop`.
"""
