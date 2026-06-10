"""Read-only data tools exposed to the Strands agent.

Each function is decorated with Strands' ``@tool`` so it can be invoked by the
model. The tools simulate a read-only analytics backend. They contain a
*secondary* line of defense: even though the governance kernel inspects tool
arguments before dispatch, ``query_sales_data`` independently rejects
destructive SQL so the data store can never be mutated.
"""

from __future__ import annotations

import re

from strands import tool

# Keywords that must never reach the (simulated) database. This mirrors the
# policy in ``policies/production.yaml`` but lives close to the data so the
# protection holds even if a tool is invoked outside the governed agent.
_DESTRUCTIVE_SQL = re.compile(
    r"\b(DROP\s+TABLE|DROP\s+DATABASE|TRUNCATE|DELETE\s+FROM|ALTER\s+TABLE|UPDATE\s+)",
    re.IGNORECASE,
)

# A tiny in-memory "database" of reports for the mock backend.
_REPORTS: dict[str, str] = {
    "R-2024-Q3": "Q3 2024 net sales: $4.2M across 1,204 orders (+12% QoQ).",
    "R-2024-Q2": "Q2 2024 net sales: $3.75M across 1,090 orders (+8% QoQ).",
    "R-2024-Q1": "Q1 2024 net sales: $3.47M across 1,011 orders (+5% QoQ).",
}


@tool
def query_sales_data(query: str) -> str:
    """Run a read-only SQL query against the sales data warehouse.

    Args:
        query: A read-only SQL ``SELECT`` statement.

    Returns:
        A string describing the (simulated) query result set.

    Raises:
        ValueError: If destructive SQL keywords are detected. This is a
            secondary line of defense behind the governance policy.
    """
    if _DESTRUCTIVE_SQL.search(query):
        raise ValueError(
            f"Refusing to execute destructive SQL in a read-only tool: {query!r}"
        )

    # Simulated read-only result — no real database is contacted.
    return (
        f"[sales_data] query executed (read-only): {query!r}\n"
        "Returned 3 rows: {region: 'NA', revenue: 4_200_000}, "
        "{region: 'EU', revenue: 2_800_000}, {region: 'APAC', revenue: 1_900_000}"
    )


@tool
def get_report_summary(report_id: str) -> str:
    """Return a human-readable summary for a known report.

    Args:
        report_id: The report identifier, e.g. ``"R-2024-Q3"``.

    Returns:
        A summary string for the report, or a not-found message.
    """
    summary = _REPORTS.get(report_id)
    if summary is None:
        return f"[report] No report found with id {report_id!r}."
    return f"[report {report_id}] {summary}"


@tool
def list_available_reports() -> list[str]:
    """List the identifiers of all reports available to the analyst.

    Returns:
        A list of report identifier strings.
    """
    return list(_REPORTS.keys())
