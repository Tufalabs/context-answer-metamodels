"""Apply the paper's complete metric suite to all unique OOD prompts."""

import sys
from cam.evaluation import domains as original


def main():
    if "--help" in sys.argv or "-h" in sys.argv:
        return original.main()
    n = int(sys.argv[sys.argv.index("--n-train") + 1])
    original.GAUSSIAN_RANKS = tuple(r for r in (64, 256, 512) if r <= n * 3)
    load = original.matrix.load_partition

    def load_full_weird(root, domain, partition, family, limit):
        if domain == "weirdchat" and partition == "test":
            partition = "all"
        return load(root, domain, partition, family, limit)

    original.matrix.load_partition = load_full_weird
    original.main()


if __name__ == "__main__":
    main()
