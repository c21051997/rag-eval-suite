# Unit tests for the retrieval metrics

from ragscope.metrics.retrieval import hit_rate, mrr

def test_hit_rate_success():
    """Tests that hit_rate returns True when a ground truth doc is present."""
    retrieved = ["doc_c", "doc_a", "doc_d"]
    ground_truth = ["doc_a", "doc_b"]
    assert hit_rate(retrieved, ground_truth) == True

def test_hit_rate_failure():
    """Tests that hit_rate returns False when no ground truth doc is present."""
    retrieved = ["doc_c", "doc_d", "doc_e"]
    ground_truth = ["doc_a", "doc_b"]
    assert hit_rate(retrieved, ground_truth) == False

def test_mrr_success():
    """Tests that MRR calculates the correct score when a doc is found."""
    retrieved = ["doc_c", "doc_a", "doc_d"]
    ground_truth = ["doc_a", "doc_b"]
    # The first correct doc is at rank 2, so the score should be 1/2 = 0.5
    assert mrr(retrieved, ground_truth) == 0.5

def test_mrr_failure():
    """Tests that MRR returns 0.0 when no correct doc is found."""
    retrieved = ["doc_c", "doc_d", "doc_e"]
    ground_truth = ["doc_a", "doc_b"]
    assert mrr(retrieved, ground_truth) == 0.0