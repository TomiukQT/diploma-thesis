import os
import numpy as np
import slack
from flask import Flask, Response, request
from slack.errors import SlackApiError
from slackeventsapi import SlackEventAdapter
from helpers.message import Message
from analyzer.analyzer import Analyzer

TOKEN = os.environ['SLACK_BOT_TOKEN']
SIGNING_SECRET = os.environ['SLACK_BOT_SIGNING_SECRET']

app = Flask(__name__)
slack_event_adapter = SlackEventAdapter(SIGNING_SECRET, '/slack/events', app)

client = slack.WebClient(token=TOKEN)
BOT_ID = client.api_call('auth.test')['user_id']
analyzer = Analyzer('models', 'model.sav', 'vectorizer.sav')


@slack_event_adapter.on('challenge')
def url_auth(payload):
    print(payload)


@slack_event_adapter.on('message')
def message(payload):
    event = payload.get('event', {})
    channel_id = event.get('channel')
    user_id = event.get('user')
    text = event.get('text')
    if user_id == BOT_ID:
        return
    client.chat_postMessage(channel=channel_id, text=f'{text} to you!')
    if text == "image":
        try:
            response = client.files_upload(
                file='F:\Obrázky\image.png',
                initial_comment='This is a sample Image',
                channels=channel_id)
        except SlackApiError as e:
            # You will get a SlackApiError if "ok" is False
            assert e.response["ok"] is False
            # str like 'invalid_auth', 'channel_not_found'
            assert e.response["error"]
            print(f"Got an error: {e.response['error']}")


@app.route('/analyze', methods=['POST'])
def analyze():

    data = request.form
    # Load history
    # print(data)
    channel_id = data.get('channel_id')
    history = load_channel_history(channel_id)

    filtered_history = filter_history(history)
    #print(filtered_history)
    # Analyze messages
    sa = analyzer.get_sentiment_analysis([m.text for m in filtered_history])
    # Analyze SA
    # Print Graph{
    msg = f'Analysed {len(filtered_history)} messages: Min: {min(sa)} Max: {max(sa)} Mean: {np.mean(sa)}'
    client.chat_postMessage(channel=channel_id, text=msg)
    return Response(), 200


def load_channel_history(channel_id: str) -> []:
    try:
        response = client.conversations_history(channel=channel_id, limit=200)
        history = response["messages"]
        while response['has_more']:
            # sleep(1)
            response = client.conversations_history(channel=channel_id, limit=200,
                                                    cursor=response['response_metadata']['next_cursor'])
            history = history + response["messages"]
        # Print results
        print(f'{len(history)} messages found')
        return history
    except SlackApiError as e:
        print("Error creating conversation: {}")
    return []


def filter_history(history: []):
    filtered = []
    for msg in history:
        if msg['user'] != BOT_ID:
            filtered.append(Message(msg['text'], msg['user'], msg['ts']))
    return filtered

