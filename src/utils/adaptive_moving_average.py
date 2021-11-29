import math
import warnings


class AdaptiveMovingAverage:

    adaptive_average = 0.0
    alpha = 0.0

    def __init__(self, alpha_min=0.01, alpha_max=1.0, alpha_gain=0.05, track_value: bool = False):
        self.alpha_tracker = AlphaTracker(alpha_min, alpha_max, alpha_gain)
        self.track_value = track_value
        if self.track_value:
            self.adaptive_average_array = []
            self.alpha_array = []

    def update(self, sample):
        self.alpha_tracker.update(sample)
        self.alpha = self.alpha_tracker.get_alpha()

        self.adaptive_average = self.alpha * sample + (1 - self.alpha) * self.adaptive_average

        if self.track_value:
            self.adaptive_average_array.append(self.adaptive_average)
            self.alpha_array.append(self.alpha)

    def get_state(self):
        return self.adaptive_average

    def get_alpha(self):
        return self.alpha


class AlphaTracker:
    low_power = 0.5
    max_shrink = 0.9
    last_sample = None

    def __init__(self, alpha_min=0.01, alpha_max=1.0, alpha_gain=0.05):
        self.alpha = alpha_max
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.alpha_gain = alpha_gain

        if alpha_min > alpha_max:
            raise Exception(f"max value of alpha must be larger than minimum value of alpha. Values given:"
                            f" alpha_min = {alpha_min}, alpha_max = {alpha_max}")

        self.gain = ExponentialMovingAverage(alpha_gain)
        self.loss = ExponentialMovingAverage(alpha_gain)

    @staticmethod
    def sign(x):
        return 1 if x > 0 else -1

    def get_instability(self, avg_gain, avg_loss):
        abs_elevation = abs(avg_gain + avg_loss)
        distance_travelled = avg_gain - avg_loss

        if distance_travelled < 0 or abs_elevation > distance_travelled:
            raise Exception("Invalid distance or elevation")

        if distance_travelled == 0:
            return 1.0

        r = abs_elevation / distance_travelled
        d = 2 * (r - 0.5)
        instability = 0.5 + (self.sign(d) * pow(abs(d), self.low_power)) / 2.0

        if instability < 0 or instability > 1:
            raise Exception(f"Invalid instability: {instability}")

        return instability

    def update(self, sample):
        slope = 0 if not self.last_sample else (sample - self.last_sample)
        self.gain.update(max(slope, 0))
        self.loss.update(min(slope, 0))

        self.last_sample = sample

        instability = self.get_instability(self.gain.get_state(), self.loss.get_state())

        alpha_ideal = self.alpha_min + (self.alpha_max - self.alpha_min) * instability
        self.alpha = max(alpha_ideal, self.max_shrink * self.alpha)

    def get_alpha(self):
        return self.alpha

    def get_avg_gain(self):
        return self.gain.get_state()

    def get_avg_loss(self):
        return self.loss.get_state()


class ExponentialMovingAverage:

    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.age = 0
        self.moving_average = 0.0
        self.age_min = int(math.ceil(1.0 / alpha))

    def set_alpha(self, alpha):
        if alpha < 0 or alpha > 1:
            old_alpha = alpha
            alpha = min(alpha, 1)
            alpha = max(alpha, 0)

            warnings.warn(f"Alpha should be between 0 and 1. Alpha was set to {old_alpha} and constrained to {alpha}")

        self.alpha = alpha

    def update(self, state):
        self.age += 1

        if self.age <= 1:
            self.moving_average = state
            return

        if self.age > self.age_min:
            alpha_t = self.alpha
        else:
            alpha_t = 1.0 / self.age

        self.moving_average = (1 - alpha_t) * self.moving_average + alpha_t * state

    def get_age(self):
        return self.age

    def get_state(self):
        return self.moving_average