
class LinearSchedule:
    def __init__(self, start, end, total_steps):
        self.start, self.end, self.total = start, end, total_steps
    def __call__(self, t):
        f = min(max(t / self.total, 0.0), 1.0)
        return self.start + f * (self.end - self.start)


