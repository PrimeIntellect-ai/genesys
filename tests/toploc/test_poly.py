import pytest
import torch
import base64
from genesys.toploc.poly import (
    find_injective_modulus,
    build_proofs,
    build_proofs_base64,
    ProofPoly,
)


def test_find_injective_modulus():
    """Test finding injective modulus"""
    x = torch.randint(0, 4_000_000_000, (100,)).tolist()
    modulus = find_injective_modulus(x)
    assert isinstance(modulus, int)
    # Check that all values are unique under modulus
    modded = [i % modulus for i in x]
    assert len(set(modded)) == len(x)


@pytest.fixture
def sample_poly():
    return ProofPoly([1, 2, 3, 4], 65497)


def test_proof_poly_init(sample_poly):
    """Test initialization of ProofPoly"""
    assert sample_poly.coeffs == [1, 2, 3, 4]
    assert sample_poly.modulus == 65497


def test_proof_poly_call(sample_poly):
    """Test polynomial evaluation"""
    x = 42
    result = sample_poly(x)
    assert isinstance(result, int)
    assert result == (1 + 2 * x + 3 * x**2 + 4 * x**3) % 65497


def test_proof_poly_len(sample_poly):
    """Test length of polynomial"""
    assert len(sample_poly) == 4


def test_proof_poly_null():
    """Test null polynomial creation"""
    length = 5
    null_poly = ProofPoly.null(length)
    assert len(null_poly) == length
    assert null_poly.modulus == 0
    assert null_poly.coeffs == [0] * length


def test_proof_poly_from_points_list():
    """Test creation from list points"""
    x = [1, 2, 3]
    y = [4, 5, 6]
    poly = ProofPoly.from_points(x, y)
    assert isinstance(poly, ProofPoly)
    assert len(poly.coeffs) > 0


def test_proof_poly_from_points_tensor():
    """Test creation from tensor points"""
    x = torch.tensor([1, 2, 3])
    y = torch.tensor([4, 5, 6])
    poly = ProofPoly.from_points(x, y)
    assert isinstance(poly, ProofPoly)
    assert len(poly.coeffs) == 3
    assert poly(1) == 4
    assert poly(2) == 5
    assert poly(3) == 6


def test_proof_poly_from_points_bfloat16():
    """Test creation from bfloat16 tensor"""
    x = torch.tensor([1, 2, 3])
    y = torch.tensor([4, 5, 6], dtype=torch.bfloat16)
    poly = ProofPoly.from_points(x, y)
    assert isinstance(poly, ProofPoly)
    assert len(poly.coeffs) == 3


def test_proof_poly_to_base64(sample_poly):
    """Test base64 encoding"""
    encoded = sample_poly.to_base64()
    assert isinstance(encoded, str)
    # Verify it's valid base64
    base64.b64decode(encoded)


def test_proof_poly_to_bytes(sample_poly):
    """Test bytes conversion"""
    byte_data = sample_poly.to_bytes()
    assert isinstance(byte_data, bytes)
    assert len(byte_data) > 0


def test_proof_poly_from_bytes(sample_poly):
    """Test creation from bytes"""
    byte_data = sample_poly.to_bytes()
    reconstructed = ProofPoly.from_bytes(byte_data)
    assert reconstructed.coeffs == sample_poly.coeffs
    assert reconstructed.modulus == sample_poly.modulus


def test_proof_poly_from_base64(sample_poly):
    """Test creation from base64"""
    encoded = sample_poly.to_base64()
    reconstructed = ProofPoly.from_base64(encoded)
    assert reconstructed.coeffs == sample_poly.coeffs
    assert reconstructed.modulus == sample_poly.modulus


def test_proof_poly_repr(sample_poly):
    """Test string representation"""
    repr_str = repr(sample_poly)
    assert isinstance(repr_str, str)
    assert str(65497) in repr_str
    assert str([1, 2, 3, 4]) in repr_str


@pytest.fixture
def sample_activations():
    DIM = 16
    a = [torch.randn(3, DIM, dtype=torch.bfloat16)]
    for _ in range(3 * 2 + 1):
        a.append(torch.randn(DIM, dtype=torch.bfloat16))
    return a


def test_build_proofs(sample_activations):
    """Test building proofs"""
    proofs = build_proofs(sample_activations, decode_batching_size=2, topk=5)
    assert isinstance(proofs, list)
    assert all(isinstance(p, bytes) for p in proofs)
    assert len(proofs) == 5


def test_build_proofs_base64(sample_activations):
    """Test building base64 proofs"""
    proofs = build_proofs_base64(sample_activations, decode_batching_size=2, topk=5)
    assert isinstance(proofs, list)
    assert all(isinstance(p, str) for p in proofs)
    # Verify each proof is valid base64
    for proof in proofs:
        base64.b64decode(proof)
    assert len(proofs) == 5


def test_build_proofs_skip_prefill(sample_activations):
    """Test building proofs with skip_prefill"""
    proofs = build_proofs(sample_activations, decode_batching_size=2, topk=5, skip_prefill=True)
    assert isinstance(proofs, list)
    assert all(isinstance(p, bytes) for p in proofs)
    assert len(proofs) == 4


def test_build_proofs_error_handling():
    """Test error handling in proof building"""
    invalid_activations = [
        torch.randn(0, 16, dtype=torch.bfloat16),
        torch.randn(16, dtype=torch.bfloat16),
    ]
    proofs = build_proofs(invalid_activations, decode_batching_size=2, topk=5)
    assert isinstance(proofs, list)
    assert all(isinstance(p, bytes) for p in proofs)

    nullproof = ProofPoly.null(5).to_bytes()
    assert all(p == nullproof for p in proofs)


def test_build_proofs_edge_cases(sample_activations):
    """Test edge cases for proof building"""
    # Test with minimal topk
    proofs_min = build_proofs(sample_activations, decode_batching_size=2, topk=1)
    assert len(proofs_min) > 0

    # Test with large batching size
    proofs_large_batch = build_proofs(sample_activations, decode_batching_size=10, topk=5)
    assert len(proofs_large_batch) > 0

    # Test with only one prefill activation
    proofs_one = build_proofs(sample_activations[:1], decode_batching_size=2, topk=5)
    assert len(proofs_one) == 1

    # Test with only one activation and skip_prefill
    proofs_one_skip = build_proofs(sample_activations[:1], decode_batching_size=2, topk=5, skip_prefill=True)
    assert len(proofs_one_skip) == 0
