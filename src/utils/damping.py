class Damping:

    def __init__(self, initial_damping: float, min_damping: float, max_damping: float,
                 damping_factor_damping: float):
        self.damping = initial_damping

        # TODO: should any of these values come from config?
        self.min_damping = min_damping
        self.max_damping = max_damping
        self.damping_factor_damping = damping_factor_damping

    def update(self):
        self.damping -= self.damping_factor_damping * (self.damping -
                                                       self.min_damping)

    def boost(self, new_damping: float = 0.0):
        if new_damping > 0.0:
            self.damping = new_damping
        else:
            self.damping = self.max_damping

    def value(self):
        return self.damping
