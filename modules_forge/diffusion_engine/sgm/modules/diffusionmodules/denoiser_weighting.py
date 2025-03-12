class EpsWeighting:
    def __call__(self, sigma):
        return sigma**-2.0


class VWeighting:
    def __init__(self, sigma_data=1.0):
        self.sigma_data = sigma_data

    def __call__(self, sigma):
        return (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
