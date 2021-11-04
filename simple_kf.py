class SimpleKF:
    def __init__(self, measure_err, estimate_err, q):
        self.measure_err = measure_err
        self.estimate_err = estimate_err
        self.q = q
        self.kalman_gain = 0
        self.current_estimate = 0
        self.last_estimate = 0

    def update_estimate(self, measure):
        self.kalman_gain = self.estimate_err / (self.estimate_err + self.measure_err)
        self.current_estimate = self.last_estimate + self.kalman_gain * (measure - self.last_estimate)
        self.estimate_err = (1. - self.kalman_gain) * self.estimate_err + abs(self.last_estimate - self.current_estimate) * self.q
        self.last_estimate = self.current_estimate
        return self.current_estimate
