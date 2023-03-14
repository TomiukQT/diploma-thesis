from datetime import datetime


class Message:
    """
    Class wrapper around slack message.
    """

    def __init__(self, text: str, user: str, timestamp: str, reactions: [] = None):
        self.text = text
        self.user = user
        self.timestamp = timestamp
        self.date = datetime.fromtimestamp(float(timestamp.split('.')[0]))
        if reactions is None:
            self.reactions = []
        else:
            self.reactions = reactions.copy()

    def __str__(self):
        return f'{self.date} >> {self.user}::{self.text}:: with {len(self.reactions)} reactions'

    def __repr__(self):
        return f'{self.date} >> {self.user}::{self.text}\n Reactions: {self.reactions}'


class Reaction:
    """
    Class wrapper around slack reactions
    """

    def __init__(self, name, count):
        self.name = name
        self.count = count

    def __str__(self):
        return f'{self.name} {self.count}x'
