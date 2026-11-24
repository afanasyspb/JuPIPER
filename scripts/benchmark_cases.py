"""Select the Docking Benchmark 4.0 enzyme/inhibitor cohort."""


def benchmark_cases(document):
    return [row for row in document["rows"] if row["category"] == "enzyme_inhibitor"]


def completed_cases(document):
    return [row for row in benchmark_cases(document) if row["status"] == "ok"]


def recovered(rows, cutoff):
    return sum(0 < row["native_rank"] <= cutoff for row in rows)
