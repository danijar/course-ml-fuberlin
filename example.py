class Example:
    """
    Immutable class representing one example in a dataset.
    """
    __slots__ = ('data', 'target')

    def __init__(self, data, target):
        super().__setattr__('data', data)
        super().__setattr__('target', target)

    def __setattr__(self, *args):
        raise TypeError

    def __delattr__(self, *args):
        raise TypeError

    def __getstate__(self):
        return {'data': self.data, 'target': self.target}

    def __setstate__(self, state):
        super().__setattr__('data', state['data'])
        super().__setattr__('target', state['target'])

    def __repr__(self):
        data = ' '.join(str(round(x, 2)) for x in self.data)
        target = ' '.join(str(round(x, 2)) for x in self.target)
        return '({})->({})'.format(data, target)
