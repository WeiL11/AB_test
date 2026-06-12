from app.assignment.hasher import assign_variant


class TestAssignVariant:
    def setup_method(self):
        self.variants = [
            {"id": "aaaa-1111", "traffic_pct": 50.0},
            {"id": "bbbb-2222", "traffic_pct": 50.0},
        ]

    def test_deterministic(self):
        """Same inputs always produce same output."""
        result1 = assign_variant("exp-1", "user-42", self.variants)
        result2 = assign_variant("exp-1", "user-42", self.variants)
        assert result1["id"] == result2["id"]

    def test_different_users_can_get_different_variants(self):
        """Over many users, both variants should be assigned."""
        assigned_ids = set()
        for i in range(100):
            result = assign_variant("exp-1", f"user-{i}", self.variants)
            assigned_ids.add(result["id"])
        assert len(assigned_ids) == 2

    def test_roughly_equal_split(self):
        """50/50 split should be roughly equal over many users."""
        counts = {"aaaa-1111": 0, "bbbb-2222": 0}
        n = 10000
        for i in range(n):
            result = assign_variant("exp-1", f"user-{i}", self.variants)
            counts[result["id"]] += 1

        # Each should be roughly 50% (within 3%)
        for count in counts.values():
            assert abs(count / n - 0.5) < 0.03

    def test_different_experiment_different_assignment(self):
        """Same user in different experiments can get different variants."""
        results = set()
        for exp_id in [f"exp-{i}" for i in range(50)]:
            result = assign_variant(exp_id, "user-1", self.variants)
            results.add(result["id"])
        # Over 50 experiments, should get both variants at least once
        assert len(results) == 2

    def test_unequal_split(self):
        """90/10 split should reflect in assignment distribution."""
        variants = [
            {"id": "aaaa-1111", "traffic_pct": 90.0},
            {"id": "bbbb-2222", "traffic_pct": 10.0},
        ]
        counts = {"aaaa-1111": 0, "bbbb-2222": 0}
        n = 10000
        for i in range(n):
            result = assign_variant("exp-1", f"user-{i}", variants)
            counts[result["id"]] += 1

        assert counts["aaaa-1111"] / n > 0.85
        assert counts["bbbb-2222"] / n < 0.15
