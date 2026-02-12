"""Reset all analyzed PRs back to 'assembled' so analysis can be re-run with a different model.

Usage: python reset_analyses.py [--dry-run]
"""

import argparse
import os
import sqlite3

from dotenv import load_dotenv

load_dotenv(override=True)


def main():
    parser = argparse.ArgumentParser(description="Delete all LLM analyses and reset PRs to 'assembled'.")
    parser.add_argument("--dry-run", action="store_true", help="Show counts without modifying anything")
    args = parser.parse_args()

    url = os.environ.get("DATABASE_URL", "sqlite:///pr_review.db")
    if url.startswith("postgresql"):
        import psycopg
        conn = psycopg.connect(url, autocommit=False)
    else:
        path = url.replace("sqlite:///", "")
        conn = sqlite3.connect(path)

    cur = conn.execute("SELECT COUNT(*) FROM llm_analyses")
    analysis_count = cur.fetchone()[0]
    cur = conn.execute("SELECT COUNT(*) FROM prs WHERE status = 'analyzed'")
    pr_count = cur.fetchone()[0]

    print(f"Found {analysis_count} analyses, {pr_count} analyzed PRs")

    if args.dry_run:
        conn.close()
        return

    if analysis_count == 0:
        print("Nothing to reset.")
        conn.close()
        return

    conn.execute("DELETE FROM llm_analyses")
    conn.execute("UPDATE prs SET status = 'assembled', analyzed_at = NULL WHERE status = 'analyzed'")
    conn.commit()
    conn.close()

    print(f"Deleted {analysis_count} analyses, reset {pr_count} PRs to 'assembled'")


if __name__ == "__main__":
    main()
