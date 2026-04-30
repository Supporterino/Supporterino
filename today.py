"""
GitHub Profile README Generator

Fetches GitHub statistics via GraphQL API and renders them into
SVG profile cards using Jinja2 templates.

Originally based on Andrew6rant's profile README generator.
Rewritten for clarity, reliability, and maintainability.
"""

from __future__ import annotations

import datetime
import json
import os
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import requests
from dateutil import relativedelta
from jinja2 import Environment, FileSystemLoader

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GRAPHQL_URL = "https://api.github.com/graphql"
BIRTHDAY = datetime.datetime(1997, 6, 24)
CACHE_DIR = Path("cache")
TEMPLATE_DIR = Path(".")

MAX_RETRIES = 3
RETRY_BACKOFF = 2  # seconds, doubles each retry


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class GitHubStats:
    """All stats needed to render the profile SVGs."""

    age: str
    commits: int
    stars: int
    repos: int
    contributed_repos: int
    followers: int
    loc_added: int
    loc_deleted: int
    loc_net: int

    def format_for_template(self) -> dict[str, str]:
        """Return a dict with all values formatted for display (comma-separated numbers)."""
        return {
            "age": self.age,
            "commits": f"{self.commits:,}",
            "stars": f"{self.stars:,}",
            "repos": f"{self.repos:,}",
            "contributed_repos": f"{self.contributed_repos:,}",
            "followers": f"{self.followers:,}",
            "loc_added": f"{self.loc_added:,}",
            "loc_deleted": f"{self.loc_deleted:,}",
            "loc_net": f"{self.loc_net:,}",
        }


# ---------------------------------------------------------------------------
# GitHub GraphQL API client
# ---------------------------------------------------------------------------


class GitHubAPI:
    """Handles all GitHub GraphQL API communication with retry logic."""

    def __init__(self, token: str, username: str) -> None:
        self.headers = {"authorization": f"token {token}"}
        self.username = username
        self.owner_id: dict[str, str] | None = None
        self.query_counts: dict[str, int] = {}

    def _request(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        """
        Execute a GraphQL request with exponential backoff retry.

        Retries on 502 (server error) and 403 (rate limit) responses.
        """
        caller = self._get_caller_name()
        self.query_counts[caller] = self.query_counts.get(caller, 0) + 1

        for attempt in range(MAX_RETRIES):
            response = requests.post(
                GRAPHQL_URL,
                json={"query": query, "variables": variables},
                headers=self.headers,
                timeout=30,
            )
            if response.status_code == 200:
                return response.json()
            if response.status_code in (502, 403) and attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF * (2**attempt)
                print(f"  Retry {attempt + 1}/{MAX_RETRIES} after {response.status_code}, waiting {wait}s...")
                time.sleep(wait)
                continue
            raise APIError(
                f"GraphQL request failed: {response.status_code} - {response.text}"
            )
        # Should not be reached, but just in case
        raise APIError("Max retries exceeded")

    @staticmethod
    def _get_caller_name() -> str:
        """Get the name of the calling method for query counting."""
        import inspect

        frame = inspect.currentframe()
        # Go up 2 frames: _request -> calling method
        if frame and frame.f_back and frame.f_back.f_back:
            return frame.f_back.f_back.f_code.co_name
        return "unknown"

    def get_user(self) -> tuple[dict[str, str], str]:
        """Return the user's ID and account creation date."""
        query = """
        query($login: String!) {
            user(login: $login) {
                id
                createdAt
            }
        }"""
        data = self._request(query, {"login": self.username})
        user = data["data"]["user"]
        self.owner_id = {"id": user["id"]}
        return self.owner_id, user["createdAt"]

    def get_followers(self) -> int:
        """Return the user's follower count."""
        query = """
        query($login: String!) {
            user(login: $login) {
                followers {
                    totalCount
                }
            }
        }"""
        data = self._request(query, {"login": self.username})
        return int(data["data"]["user"]["followers"]["totalCount"])

    def get_repos_or_stars(
        self,
        count_type: str,
        owner_affiliation: list[str],
        cursor: str | None = None,
    ) -> int:
        """
        Return total repo count or total star count.

        Args:
            count_type: Either 'repos' or 'stars'
            owner_affiliation: List of affiliations to filter by
            cursor: Pagination cursor
        """
        query = """
        query ($owner_affiliation: [RepositoryAffiliation], $login: String!, $cursor: String) {
            user(login: $login) {
                repositories(first: 100, after: $cursor, ownerAffiliations: $owner_affiliation) {
                    totalCount
                    edges {
                        node {
                            ... on Repository {
                                nameWithOwner
                                stargazers {
                                    totalCount
                                }
                            }
                        }
                    }
                    pageInfo {
                        endCursor
                        hasNextPage
                    }
                }
            }
        }"""
        variables = {
            "owner_affiliation": owner_affiliation,
            "login": self.username,
            "cursor": cursor,
        }
        data = self._request(query, variables)
        repos = data["data"]["user"]["repositories"]

        if count_type == "repos":
            return repos["totalCount"]

        # count_type == 'stars'
        total_stars = sum(
            edge["node"]["stargazers"]["totalCount"] for edge in repos["edges"]
        )
        if repos["pageInfo"]["hasNextPage"]:
            total_stars += self.get_repos_or_stars(
                count_type, owner_affiliation, repos["pageInfo"]["endCursor"]
            )
        return total_stars

    def get_all_repos(
        self,
        owner_affiliation: list[str],
        cursor: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Fetch all repositories (paginated) with their commit counts.
        Returns a flat list of repo edge nodes.
        """
        query = """
        query ($owner_affiliation: [RepositoryAffiliation], $login: String!, $cursor: String) {
            user(login: $login) {
                repositories(first: 60, after: $cursor, ownerAffiliations: $owner_affiliation) {
                    edges {
                        node {
                            ... on Repository {
                                nameWithOwner
                                defaultBranchRef {
                                    target {
                                        ... on Commit {
                                            history {
                                                totalCount
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    pageInfo {
                        endCursor
                        hasNextPage
                    }
                }
            }
        }"""
        variables = {
            "owner_affiliation": owner_affiliation,
            "login": self.username,
            "cursor": cursor,
        }
        data = self._request(query, variables)
        repos = data["data"]["user"]["repositories"]
        edges = repos["edges"]

        if repos["pageInfo"]["hasNextPage"]:
            edges += self.get_all_repos(
                owner_affiliation, repos["pageInfo"]["endCursor"]
            )
        return edges

    def get_repo_loc(
        self,
        owner: str,
        repo_name: str,
        cursor: str | None = None,
        additions: int = 0,
        deletions: int = 0,
        my_commits: int = 0,
    ) -> tuple[int, int, int]:
        """
        Recursively fetch all commits for a repo and count LOC authored by this user.
        Returns (additions, deletions, my_commits).
        """
        query = """
        query ($repo_name: String!, $owner: String!, $cursor: String) {
            repository(name: $repo_name, owner: $owner) {
                defaultBranchRef {
                    target {
                        ... on Commit {
                            history(first: 100, after: $cursor) {
                                totalCount
                                edges {
                                    node {
                                        ... on Commit {
                                            committedDate
                                        }
                                        author {
                                            user {
                                                id
                                            }
                                        }
                                        deletions
                                        additions
                                    }
                                }
                                pageInfo {
                                    endCursor
                                    hasNextPage
                                }
                            }
                        }
                    }
                }
            }
        }"""
        variables = {"repo_name": repo_name, "owner": owner, "cursor": cursor}
        data = self._request(query, variables)

        branch_ref = data["data"]["repository"]["defaultBranchRef"]
        if branch_ref is None:
            return 0, 0, 0

        history = branch_ref["target"]["history"]

        for edge in history["edges"]:
            node = edge["node"]
            if node["author"]["user"] == self.owner_id:
                my_commits += 1
                additions += node["additions"]
                deletions += node["deletions"]

        if history["edges"] and history["pageInfo"]["hasNextPage"]:
            return self.get_repo_loc(
                owner,
                repo_name,
                history["pageInfo"]["endCursor"],
                additions,
                deletions,
                my_commits,
            )

        return additions, deletions, my_commits


class APIError(Exception):
    """Raised when a GitHub API request fails after retries."""


# ---------------------------------------------------------------------------
# JSON-based LOC cache
# ---------------------------------------------------------------------------


class LOCCache:
    """
    JSON-based cache for lines-of-code data.

    Stores per-repo commit counts and LOC to avoid re-fetching
    unchanged repositories on every run.

    Cache file structure:
    {
        "username": "Supporterino",
        "repos": {
            "Supporterino/some-repo": {
                "total_commits": 150,
                "my_commits": 42,
                "additions": 5000,
                "deletions": 1200
            }
        }
    }
    """

    def __init__(self, username: str, cache_dir: Path = CACHE_DIR) -> None:
        self.username = username
        self.cache_file = cache_dir / "loc_cache.json"
        self.data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        """Load cache from disk, or return empty structure."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r") as f:
                    data = json.load(f)
                if data.get("username") == self.username:
                    return data
                print(f"  Cache username mismatch, rebuilding...")
            except (json.JSONDecodeError, KeyError):
                print(f"  Cache file corrupted, rebuilding...")
        return {"username": self.username, "repos": {}}

    def _save(self) -> None:
        """Atomically write cache to disk using tempfile + rename."""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        # Write to temp file in the same directory, then atomically rename
        fd, tmp_path = tempfile.mkstemp(
            dir=self.cache_file.parent, suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self.data, f, indent=2)
            os.replace(tmp_path, self.cache_file)
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def get_repo(self, repo_name: str) -> dict[str, int] | None:
        """Get cached data for a repository, or None if not cached."""
        return self.data["repos"].get(repo_name)

    def set_repo(
        self,
        repo_name: str,
        total_commits: int,
        my_commits: int,
        additions: int,
        deletions: int,
    ) -> None:
        """Update cached data for a repository and persist to disk."""
        self.data["repos"][repo_name] = {
            "total_commits": total_commits,
            "my_commits": my_commits,
            "additions": additions,
            "deletions": deletions,
        }
        self._save()

    def remove_stale_repos(self, current_repos: set[str]) -> None:
        """Remove cached repos that no longer exist."""
        stale = set(self.data["repos"].keys()) - current_repos
        if stale:
            for repo in stale:
                del self.data["repos"][repo]
            print(f"  Removed {len(stale)} stale repos from cache")
            self._save()

    def get_totals(self) -> tuple[int, int, int, int]:
        """
        Return aggregate totals from all cached repos.
        Returns (total_additions, total_deletions, total_loc_net, total_my_commits).
        """
        additions = 0
        deletions = 0
        my_commits = 0
        for repo_data in self.data["repos"].values():
            additions += repo_data["additions"]
            deletions += repo_data["deletions"]
            my_commits += repo_data["my_commits"]
        return additions, deletions, additions - deletions, my_commits


# ---------------------------------------------------------------------------
# LOC computation (orchestrates API + cache)
# ---------------------------------------------------------------------------


def compute_loc(api: GitHubAPI, cache: LOCCache) -> tuple[int, int, int, int]:
    """
    Compute total LOC across all repos, using cache where possible.

    Returns (additions, deletions, loc_net, total_commits).
    """
    affiliation = ["OWNER", "COLLABORATOR", "ORGANIZATION_MEMBER"]
    edges = api.get_all_repos(affiliation)

    # Track current repos to clean stale cache entries
    current_repos: set[str] = set()

    for edge in edges:
        node = edge["node"]
        repo_name = node["nameWithOwner"]
        current_repos.add(repo_name)

        # Get current commit count from API response
        try:
            api_commit_count = node["defaultBranchRef"]["target"]["history"]["totalCount"]
        except TypeError:
            # Empty repo (no default branch)
            cache.set_repo(repo_name, 0, 0, 0, 0)
            continue

        # Check if cache is still valid
        cached = cache.get_repo(repo_name)
        if cached and cached["total_commits"] == api_commit_count:
            continue  # Cache hit - no changes

        # Cache miss or stale - re-fetch LOC for this repo
        owner, name = repo_name.split("/")
        print(f"  Fetching LOC for {repo_name}...")
        try:
            additions, deletions, my_commits = api.get_repo_loc(owner, name)
            cache.set_repo(repo_name, api_commit_count, my_commits, additions, deletions)
        except APIError as e:
            print(f"  Warning: Failed to fetch LOC for {repo_name}: {e}")
            # Keep existing cache entry if available, otherwise set zeros
            if not cached:
                cache.set_repo(repo_name, 0, 0, 0, 0)

    # Remove repos that no longer exist
    cache.remove_stale_repos(current_repos)

    return cache.get_totals()


# ---------------------------------------------------------------------------
# Age / uptime calculation
# ---------------------------------------------------------------------------


def calculate_age(birthday: datetime.datetime) -> str:
    """
    Return a human-readable age string.
    e.g. '28 years, 9 months, 2 days'
    """
    diff = relativedelta.relativedelta(datetime.datetime.today(), birthday)
    parts = [
        (diff.years, "year"),
        (diff.months, "month"),
        (diff.days, "day"),
    ]
    formatted = ", ".join(
        f"{value} {unit}{'s' if value != 1 else ''}" for value, unit in parts
    )
    if diff.months == 0 and diff.days == 0:
        formatted += " \U0001f382 "  # birthday cake emoji
    return formatted


# ---------------------------------------------------------------------------
# SVG rendering with Jinja2
# ---------------------------------------------------------------------------


def dots_filter(value: str, length: int) -> str:
    """
    Jinja2 filter that generates dot-leader padding.

    Given a value string and target length, produces a dot string
    that right-justifies the value visually in the SVG.
    """
    text = str(value)
    padding = max(0, length - len(text))
    if padding == 0:
        return ""
    if padding == 1:
        return " "
    if padding == 2:
        return ". "
    return " " + ("." * padding) + " "


def render_svgs(stats: GitHubStats) -> None:
    """Render Jinja2 SVG templates to output files."""
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        # Keep whitespace exactly as-is (critical for SVG text layout)
        keep_trailing_newline=True,
        lstrip_blocks=False,
        trim_blocks=False,
    )
    env.filters["dots"] = dots_filter

    template_data = stats.format_for_template()

    templates = [
        ("dark_mode.svg.j2", "dark_mode.svg"),
        ("light_mode.svg.j2", "light_mode.svg"),
    ]

    for template_name, output_name in templates:
        template = env.get_template(template_name)
        rendered = template.render(**template_data)
        with open(output_name, "w", encoding="utf-8") as f:
            f.write(rendered)
        print(f"  Rendered {output_name}")


# ---------------------------------------------------------------------------
# Performance timing helper
# ---------------------------------------------------------------------------


def timed(label: str, func, *args, **kwargs):
    """Run a function, print its execution time, and return the result."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    if elapsed > 1:
        time_str = f"{elapsed:.4f} s"
    else:
        time_str = f"{elapsed * 1000:.4f} ms"
    print(f"  {label:<25} {time_str:>12}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    token = os.environ["ACCESS_TOKEN"]
    username = os.environ["USER_NAME"]

    print("GitHub Profile README Generator")
    print("=" * 40)

    api = GitHubAPI(token, username)
    cache = LOCCache(username)

    # Fetch user identity (needed for LOC author filtering)
    print("\nFetching data:")
    owner_id, acc_date = timed("Account data", api.get_user)

    # Calculate age/uptime
    age = timed("Age calculation", calculate_age, BIRTHDAY)

    # Fetch LOC (most expensive operation - uses cache)
    loc_added, loc_deleted, loc_net, total_commits = timed(
        "Lines of code", compute_loc, api, cache
    )

    # Fetch remaining stats
    stars = timed("Stars", api.get_repos_or_stars, "stars", ["OWNER"])
    repos = timed("Repos (owned)", api.get_repos_or_stars, "repos", ["OWNER"])
    contributed_repos = timed(
        "Repos (contributed)",
        api.get_repos_or_stars,
        "repos",
        ["OWNER", "COLLABORATOR", "ORGANIZATION_MEMBER"],
    )
    followers = timed("Followers", api.get_followers)

    # Build stats object
    stats = GitHubStats(
        age=age,
        commits=total_commits,
        stars=stars,
        repos=repos,
        contributed_repos=contributed_repos,
        followers=followers,
        loc_added=loc_added,
        loc_deleted=loc_deleted,
        loc_net=loc_net,
    )

    # Render SVGs
    print("\nRendering:")
    render_svgs(stats)

    # Print summary
    print("\nAPI call summary:")
    total_calls = sum(api.query_counts.values())
    print(f"  Total calls: {total_calls}")
    for name, count in sorted(api.query_counts.items()):
        print(f"    {name:<25} {count:>4}")


if __name__ == "__main__":
    main()
