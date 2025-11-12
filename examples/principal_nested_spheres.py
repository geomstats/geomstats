# -*- coding: utf-8 -*-
"""Example of using Principal Nested Spheres for dimension reduction.

This example demonstrates the Principal Nested Spheres (PNS) method
for reducing the dimensionality of data on hyperspheres through
nested subsphere fitting.
"""

import matplotlib.pyplot as plt

import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.principal_nested_spheres import PrincipalNestedSpheres


def main():
    """Run the Principal Nested Spheres example."""
    # Set random seed for reproducibility
    gs.random.seed(42)

    # Create a sphere and generate data
    print("Principal Nested Spheres Example")
    print("=" * 50)

    sphere = Hypersphere(dim=2)
    print("\nGenerating data on S^2...")

    # Generate concentrated data using Von Mises-Fisher distribution
    n_samples = 100
    kappa = 10  # Concentration parameter
    X = sphere.random_von_mises_fisher(kappa=kappa, n_samples=n_samples)

    print(f"  Number of samples: {n_samples}")
    print(f"  Data shape: {X.shape}")
    print(f"  Concentration parameter (kappa): {kappa}")

    # Fit PNS
    print("\nFitting Principal Nested Spheres...")
    pns = PrincipalNestedSpheres(space=sphere, n_init=10, verbose=False)
    pns.fit(X)

    # Transform to reduced representation
    X_reduced = pns.transform(X)

    print("\nResults:")
    print(f"  Reduced data shape: {X_reduced.shape}")
    print(f"  Number of nested subspheres: {len(pns.nested_spheres_)}")

    for i, (normal, height) in enumerate(pns.nested_spheres_):
        sphere_type = "great" if abs(height) < 1e-6 else "small"
        print(f"  Subsphere {i + 1}: {sphere_type} sphere (height={height:.6f})")

    print(f"\nMean on S^1: {pns.mean_}")

    # Compute statistics
    residuals = pns.residuals_[0]
    mean_residual = gs.mean(gs.abs(residuals))
    std_residual = gs.std(residuals)

    print("\nResidual statistics:")
    print(f"  Mean |residual|: {mean_residual:.6f}")
    print(f"  Std residual: {std_residual:.6f}")

    # Visualization
    try:
        print("\nCreating visualization...")
        fig = plt.figure(figsize=(15, 5))

        # Original data on S^2
        ax1 = fig.add_subplot(131, projection="3d")
        ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c="blue", alpha=0.6, s=50)
        ax1.set_title("Original Data on S^2", fontsize=14, fontweight="bold")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")

        # Reduced data on S^1
        ax2 = fig.add_subplot(132)
        angles = gs.arctan2(X_reduced[:, 1], X_reduced[:, 0])
        circle = plt.Circle(
            (0, 0), 1, fill=False, color="gray", linestyle="--", linewidth=2
        )
        ax2.add_patch(circle)
        ax2.scatter(X_reduced[:, 0], X_reduced[:, 1], c="red", alpha=0.6, s=50)
        ax2.scatter(
            pns.mean_[0],
            pns.mean_[1],
            c="green",
            s=300,
            marker="*",
            label="Mean",
            edgecolors="black",
            linewidth=2,
        )
        ax2.set_xlim(-1.5, 1.5)
        ax2.set_ylim(-1.5, 1.5)
        ax2.set_aspect("equal")
        ax2.set_title("Reduced Data on S^1", fontsize=14, fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Residuals histogram
        ax3 = fig.add_subplot(133)
        residuals_np = gs.to_numpy(residuals)
        ax3.hist(residuals_np, bins=20, alpha=0.7, edgecolor="black")
        ax3.set_title("Residuals Distribution", fontsize=14, fontweight="bold")
        ax3.set_xlabel("Signed Distance")
        ax3.set_ylabel("Frequency")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            "principal_nested_spheres_example.png", dpi=100, bbox_inches="tight"
        )
        print("  Saved visualization to 'principal_nested_spheres_example.png'")

        # Optionally show the plot
        # plt.show()
        plt.close()

    except Exception as e:
        print(f"  Visualization skipped: {e}")

    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
