from datetime import datetime


class Message:

    def __init__(self, text, user, timestamp):
        self.text = text
        self.user = user
        self.timestamp = timestamp
        self.date = datetime.fromtimestamp(float(timestamp.split('.')[0]))

    def __str__(self):
        return f'{self.date} >> {self.user}:: {self.text}'

    def __repr__(self):
        return f'{self.date} >> {self.user}:: {self.text}'
