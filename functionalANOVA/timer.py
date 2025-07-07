from tqdm import tqdm

class TimeBar:

    def __init__(self, total, desc="Processing"):
        self.pbar = tqdm(total=total, desc=desc)

    def progress(self):
        self.pbar.update(1)

    def stop(self):
        self.pbar.close()

    def delete(self):
        self.pbar.close()

def setUpTimeBar(method, total):
    return TimeBar(total, f"{method}")
