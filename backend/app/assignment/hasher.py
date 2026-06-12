import hashlib


def assign_variant(experiment_id: str, user_id: str, variants: list[dict]) -> dict:
    """
    Deterministically assign a user to a variant using consistent hashing.

    Args:
        experiment_id: UUID string of the experiment
        user_id: User identifier string
        variants: List of dicts with 'id' and 'traffic_pct' keys, sorted by id

    Returns:
        The selected variant dict

    How it works:
        1. Hash sha256(experiment_id + "." + user_id)
        2. Take hash_value % 10000 to get a bucket (0-9999)
        3. Walk through variants accumulating traffic_pct
        4. First variant whose cumulative percentage covers the bucket wins

    This is deterministic: same user always gets same variant for same experiment.
    """
    hash_input = f"{experiment_id}.{user_id}".encode("utf-8")
    hash_hex = hashlib.sha256(hash_input).hexdigest()
    bucket = int(hash_hex, 16) % 10000

    cumulative = 0.0
    # Sort variants by id for consistency
    sorted_variants = sorted(variants, key=lambda v: str(v["id"]))
    for variant in sorted_variants:
        cumulative += variant["traffic_pct"] * 100  # traffic_pct is 0-100, bucket is 0-9999
        if bucket < cumulative:
            return variant

    # Fallback to last variant (shouldn't happen if traffic_pct sums to 100)
    return sorted_variants[-1]
