import setuptools
from pathlib import Path


def read_requirements():
    """Read requirements from requirements.txt if it exists."""
    req_path = Path(__file__).parent / "requirements.txt"
    if req_path.exists():
        with open(req_path) as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return []


setuptools.setup(
    name="safety_bigym",
    version="0.1.0",
    author="Ayush Patel",
    description="Safety wrapper for BiGym with SMPL-H humans and ISO 15066 monitoring",
    packages=setuptools.find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "bigym",  # Core dependency - installed from local path
        "numpy>=1.26,<2.0",
        "mujoco>=3.1.5",
        "scipy",  # For rotation conversions
    ],
    package_data={
        "": [str(p.resolve()) for p in Path("safety_bigym/assets").glob("**/*")]
    },
    extras_require={
        "dev": ["pytest", "pytest-cov"],
    },
)
