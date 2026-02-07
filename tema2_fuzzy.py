import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class GaussianMF:
    center: float
    sigma: float

    def __call__(self, x: float) -> float:
        return math.exp(-((x - self.center) ** 2) / (2 * self.sigma**2))


def sample_universe(start: int, end: int) -> List[int]:
    return list(range(start, end + 1))


def compute_memberships(
    xs: Iterable[int],
    mf_a: GaussianMF,
    mf_b: GaussianMF,
) -> List[Tuple[int, float, float, float, float, float]]:
    results = []
    for x in xs:
        mu_a = mf_a(x)
        mu_b = mf_b(x)
        mu_union = max(mu_a, mu_b)
        mu_intersection = min(mu_a, mu_b)
        mu_complement_a = 1.0 - mu_a
        results.append((x, mu_a, mu_b, mu_union, mu_intersection, mu_complement_a))
    return results


def print_table(rows: List[Tuple[int, float, float, float, float, float]]) -> None:
    header = (
        "x",
        "μA(x)",
        "μB(x)",
        "μA∪B(x)=max",
        "μA∩B(x)=min",
        "μĀ(x)=1-μA",
    )
    print("{:>2} {:>8} {:>8} {:>14} {:>14} {:>12}".format(*header))
    for row in rows:
        x, mu_a, mu_b, mu_union, mu_intersection, mu_complement_a = row
        print(
            f"{x:>2} {mu_a:>8.3f} {mu_b:>8.3f} {mu_union:>14.3f} {mu_intersection:>14.3f} {mu_complement_a:>12.3f}"
        )


def plot_sets(xs: List[int], values: List[float], title: str, label: str, filename: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(xs, values, marker="o", label=label)
    plt.ylim(0, 1.05)
    plt.xlabel("x")
    plt.ylabel("μ")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_two_sets(xs: List[int], values_a: List[float], values_b: List[float], title: str, filename: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(xs, values_a, marker="o", label="A")
    plt.plot(xs, values_b, marker="o", label="B")
    plt.ylim(0, 1.05)
    plt.xlabel("x")
    plt.ylabel("μ")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def main() -> None:
    xs = sample_universe(1, 10)

    mf_a = GaussianMF(center=6.5, sigma=1.8)
    mf_b = GaussianMF(center=3.5, sigma=1.6)

    rows = compute_memberships(xs, mf_a, mf_b)
    print("Ecuaciones:")
    print("μA(x) = exp(-(x-6.5)^2 / (2·1.8^2))")
    print("μB(x) = exp(-(x-3.5)^2 / (2·1.6^2))")
    print("μA∪B(x) = max(μA(x), μB(x))")
    print("μA∩B(x) = min(μA(x), μB(x))")
    print("μĀ(x) = 1 - μA(x)")
    print("\nTabla de valores:")
    print_table(rows)

    mu_a = [row[1] for row in rows]
    mu_b = [row[2] for row in rows]
    mu_union = [row[3] for row in rows]
    mu_intersection = [row[4] for row in rows]
    mu_complement_a = [row[5] for row in rows]

    plot_two_sets(xs, mu_a, mu_b, "Conjuntos difusos A y B", "sets_ab.png")
    plot_sets(xs, mu_union, "Unión de μA(x) y μB(x)", "μA∪B(x)", "union.png")
    plot_sets(xs, mu_intersection, "Intersección de μA(x) y μB(x)", "μA∩B(x)", "intersection.png")
    plot_sets(xs, mu_complement_a, "Complemento de μA(x)", "μĀ(x)", "complemento_a.png")


if __name__ == "__main__":
    main()
