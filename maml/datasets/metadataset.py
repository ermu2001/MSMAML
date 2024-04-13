from collections import defaultdict, namedtuple
_fields = ['x', 'y', 'task_info', 'gt']
Task = namedtuple('Task', _fields, defaults=(None,) * len(_fields))