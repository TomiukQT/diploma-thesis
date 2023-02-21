import os
import slack
from flask import Flask, Response, request
from slack.errors import SlackApiError
from slackeventsapi import SlackEventAdapter

TOKEN = os.environ['SLACK_BOT_TOKEN']
SIGNING_SECRET = os.environ['SLACK_BOT_SIGNING_SECRET']

app = Flask(__name__)
slack_event_adapter = SlackEventAdapter(SIGNING_SECRET, '/slack/events', app)

client = slack.WebClient(token=TOKEN)
#client.chat_postMessage(channel='#bot_test', text='Hello')
BOT_ID = client.api_call('auth.test')['user_id']


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
    print(data)
    channel_id = data.get('channel_id')
    history = client.conversations_history(channel_id)
    client.chat_postMessage(channel=channel_id, text=f'Message count: {len(history)}')
    return Response(), 200


if __name__ == '__main__':
    app.run(debug=True)
