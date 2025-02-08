import os


class Dir:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    test_sources_dir = os.path.join(root, "tests", "sources")
