import os


class Dir:
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    result_dir = os.path.join(root_dir, "result")
