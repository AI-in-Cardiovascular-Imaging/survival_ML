from collections import defaultdict


class NestedDefaultDict(defaultdict):
    """Nested dict, which can be dynamically expanded"""

    def __init__(self, *args, **kwargs):
        super().__init__(NestedDefaultDict, *args, **kwargs)

    def __repr__(self):
        return repr(dict(self))