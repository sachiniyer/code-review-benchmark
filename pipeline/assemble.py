"""Pipeline stage: Assemble enriched PR data into unified PRRecord (DB-backed).

Reuses the pure-function assembly logic from assemble.py but reads/writes DB instead of files.
"""

from __future__ import annotations

import json
import logging

from config import DBConfig
from db.connection import DBAdapter
from db.repository import PRRepository

# Import the pure assembly functions from the original module
from assemble import (
    _build_review_threads,
    _build_timeline_events,
    _compute_stats,
    _determine_roles,
    _enrich_timeline_with_threads,
    _extract_pr_metadata,
    _parse_timestamp,
)

logger = logging.getLogger(__name__)


def _json_load(val: str | list | dict | None) -> list | dict | None:
    """Parse a JSONB column value — may be a string (SQLite) or already parsed (Postgres)."""
    if val is None:
        return None
    if isinstance(val, str):
        return json.loads(val)
    return val


def assemble_pr_from_row(pr_row: dict, chatbot_username: str) -> dict | None:
    """Assemble a PRRecord dict from a database row.

    Returns the assembled record as a dict, or None if required data is missing.
    """
    bq_events = _json_load(pr_row.get("bq_events"))
    commits = _json_load(pr_row.get("commits"))
    reviews = _json_load(pr_row.get("reviews"))
    raw_threads = _json_load(pr_row.get("review_threads"))
    commit_details = _json_load(pr_row.get("commit_details"))

    if bq_events is None:
        logger.warning(f"No BQ events for PR {pr_row['repo_name']}#{pr_row['pr_number']} — skipping assembly")
        return None

    meta = _extract_pr_metadata(bq_events)
    timeline = _build_timeline_events(bq_events, commits, commit_details, reviews)
    threads = _build_review_threads(raw_threads)
    _enrich_timeline_with_threads(timeline, raw_threads)
    timeline.sort(key=lambda e: (_parse_timestamp(e.timestamp), e.data.get("order_index", 0)))
    stats = _compute_stats(chatbot_username, timeline, threads)
    roles = _determine_roles(chatbot_username, timeline, meta["pr_author"])

    return {
        "pr_url": pr_row["pr_url"],
        "repo_name": pr_row["repo_name"],
        "pr_number": pr_row["pr_number"],
        "pr_title": meta["pr_title"],
        "pr_author": meta["pr_author"],
        "pr_created_at": meta["pr_created_at"],
        "pr_merged": meta["pr_merged"],
        "target_user_roles": roles,
        "events": [e.to_dict() for e in timeline],
        "review_threads": [t.to_dict() for t in threads],
        "stats": stats.to_dict(),
    }


async def assemble_pr(
    repo: PRRepository,
    pr_row: dict,
    chatbot_username: str,
) -> bool:
    """Assemble a single PR and save to DB. Returns True if successful."""
    record = assemble_pr_from_row(pr_row, chatbot_username)
    if record is None:
        return False

    await repo.mark_assembled(pr_row["id"], record)

    # Also update metadata from BQ events
    await repo.update_metadata(
        pr_row["id"],
        pr_title=record["pr_title"],
        pr_author=record["pr_author"],
        pr_created_at=record["pr_created_at"],
        pr_merged=record["pr_merged"],
    )

    logger.debug(f"Assembled {pr_row['repo_name']}#{pr_row['pr_number']}")
    return True


async def assemble_enriched_prs(
    db: DBAdapter,
    chatbot_id: int,
    chatbot_username: str,
) -> int:
    """Assemble all enriched PRs for a chatbot. Returns count of assembled PRs."""
    repo = PRRepository(db)

    # Get all enriched PRs that haven't been assembled yet
    rows = await db.fetchall(
        *db._translate_params(
            "SELECT * FROM prs WHERE chatbot_id = $1 AND status = 'enriched' ORDER BY discovered_at",
            (chatbot_id,),
        )
    )

    assembled = 0
    for row in rows:
        try:
            if await assemble_pr(repo, row, chatbot_username):
                assembled += 1
        except Exception as e:
            logger.error(f"Error assembling {row['repo_name']}#{row['pr_number']}: {e}")
            await repo.mark_error(row["id"], f"Assembly error: {e}")

    logger.info(f"Assembled {assembled}/{len(rows)} enriched PRs for {chatbot_username}")
    return assembled
