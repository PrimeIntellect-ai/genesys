import base64
from .ndd import compute_newton_coefficients, evaluate_polynomial
import torch


def find_injective_modulus(x: list[int]) -> int:
    for i in range(65497, 2**15, -1):
        if len(set([j % i for j in x])) == len(x):
            return i
    raise ValueError("No injective modulus found!")


class ProofPoly:
    def __init__(self, coeffs: list[int], modulus: int):
        self.coeffs = coeffs
        self.modulus = modulus

    def __call__(self, x: int):
        return evaluate_polynomial(self.coeffs, x % self.modulus)

    def __len__(self):
        return len(self.coeffs)

    @classmethod
    def from_points(cls, x: list[int] | torch.Tensor, y: list[int] | torch.Tensor) -> "ProofPoly":
        if isinstance(x, torch.Tensor):
            x = x.tolist()
        if isinstance(y, torch.Tensor):
            if y.dtype == torch.bfloat16:
                y = y.view(dtype=torch.uint16).tolist()
            elif y.dtype == torch.float32:
                raise NotImplementedError("float32 not supported yet because interpolate has hardcode prime")
            else:
                y = y.tolist()
        modulus = find_injective_modulus(x)
        x = [i % modulus for i in x]
        return cls(compute_newton_coefficients(x, y), modulus)

    def to_base64(self):
        base64_encoded = base64.b64encode(self.to_bytes()).decode("utf-8")
        return base64_encoded

    def to_bytes(self):
        return self.modulus.to_bytes(2, byteorder="big", signed=False) + b"".join(
            coeff.to_bytes(2, byteorder="big", signed=False) for coeff in self.coeffs
        )

    @classmethod
    def from_bytes(cls, byte_data: bytes) -> "ProofPoly":
        modulus = int.from_bytes(byte_data[:2], byteorder="big", signed=False)
        coeffs = [
            int.from_bytes(byte_data[i : i + 2], byteorder="big", signed=False) for i in range(2, len(byte_data), 2)
        ]
        return cls(coeffs, modulus)

    @classmethod
    def from_base64(cls, base64_encoded: str) -> "ProofPoly":
        byte_data = base64.b64decode(base64_encoded)
        return cls.from_bytes(byte_data)

    def __repr__(self) -> str:
        return f"ProofPoly[{self.modulus}]({self.coeffs})"


def build_proofs(
    activations: list[torch.Tensor], decode_batching_size: int, topk: int, skip_prefill: bool = False
) -> list[bytes]:
    return [proof.to_bytes() for proof in _build_proofs(activations, decode_batching_size, topk, skip_prefill)]


def build_proofs_base64(
    activations: list[torch.Tensor], decode_batching_size: int, topk: int, skip_prefill: bool = False
) -> list[str]:
    return [proof.to_base64() for proof in _build_proofs(activations, decode_batching_size, topk, skip_prefill)]


def _build_proofs(
    activations: list[torch.Tensor], decode_batching_size: int, topk: int, skip_prefill: bool = False
) -> list[ProofPoly]:
    proofs = []

    # Prefill
    if not skip_prefill:
        flat_view = activations[0].view(-1)
        topk_indices = flat_view.abs().topk(topk).indices
        topk_values = flat_view[topk_indices]
        proof = ProofPoly.from_points(topk_indices, topk_values)
        proofs.append(proof)

    # Batched Decode
    for i in range(1, len(activations), decode_batching_size):
        flat_view = torch.cat([i.view(-1) for i in activations[i : i + decode_batching_size]])
        topk_indices = flat_view.abs().topk(topk).indices
        topk_values = flat_view[topk_indices]
        proof = ProofPoly.from_points(topk_indices, topk_values)
        proofs.append(proof)

    return proofs
