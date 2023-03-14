from datetime import datetime


class Message:
    """
    Class wrapper around slack message.
    """

    def __init__(self, text, user, timestamp):
        self.text = text
        self.user = user
        self.timestamp = timestamp
        self.date = datetime.fromtimestamp(float(timestamp.split('.')[0]))

    def __str__(self):
        return f'{self.date} >> {self.user}:: {self.text}'

    def __repr__(self):
        return f'{self.date} >> {self.user}:: {self.text}'


class Reaction:
    """
    Class wrapper around slack reactions
    """

    def __init__(self, name, count):
        self.name = name
        self.count = count

    def __str__(self):
        return f'{self.name} {self.count}x'
