import json
import os
import numpy as np
import slack
from flask import Flask, Response, request, stream_with_context
from slack.errors import SlackApiError
from slackeventsapi import SlackEventAdapter
from helpers.message import Message, Reaction
from analyzer.analyzer import Analyzer
from analyzer.time_analyzer import TimeSeriesAnalyzer
from analyzer.data_transformation import MessageTranslator
from helpers.data_uploader import DataUploader
from collections import namedtuple
from datetime import datetime
import schedule

TOKEN = os.environ['SLACK_BOT_TOKEN']
SIGNING_SECRET = os.environ['SLACK_BOT_SIGNING_SECRET']

app = Flask(__name__)
slack_event_adapter = SlackEventAdapter(SIGNING_SECRET, '/slack/events', app)

client = slack.WebClient(token=TOKEN)
BOT_ID = client.api_call('auth.test')['user_id']
analyzer = Analyzer('models', 'LogisticRegression-class_weightNonen_jobs-1penaltynonesolvernewton-cholesky-bal-1680096073.552102.sav', 'vectorizer.sav')
ts_analyzer = TimeSeriesAnalyzer()
data_uploader = DataUploader()
message_translator = MessageTranslator()

daily_report_channels = {}


@slack_event_adapter.on('challenge')
def url_auth(payload):
    print(payload)


@slack_event_adapter.on('message')
def message(payload):
    """
    TEST EVENT
    TODO: TO BE REMOVED PROBABLY
    :param payload:
    :return:
    """
    event = payload.get('event', {})
    channel_id = event.get('channel')
    user_id = event.get('user')
    text = event.get('text')
    return Response(), 200  # TMP
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


def channel_analysis(channel_id: str, args_text: str = '', user_args: str = None) -> Response:
    args = parse_args(args_text)
    user_id = parse_user_args(args_text)
    history = load_channel_history(channel_id)
    filtered_history = filter_history(history, channel_id, date_range=args, user=user_id)
    if len(filtered_history) <= 0:
        client.chat_postMessage(channel=channel_id, text='No data to analyze')
        return Response(), 200
    # Save messages
    data_uploader.save_file(data_uploader.messages_to_file(filtered_history))

    # Translate if needed
    message_translator.translate_messages(filtered_history)

    # Analyze messages
    sa = analyzer.get_sentiment_analysis([m.text for m in filtered_history])
    # Analyze SA
    date_indexed_data = ts_analyzer.index_dates(sa, [m.date for m in filtered_history])
    trend = ts_analyzer.extract_trend(date_indexed_data, start_date=ts_analyzer.parse_date('last_week'))
    # Predictions
    prediction_data = ts_analyzer.get_predictions(date_indexed_data)

    # Print Graph
    graph_path = analyzer.get_plot(plot_path='out/graphs/foo.png', trend_data=trend, predictions_data=prediction_data)

    msg = f'Analysed {len(filtered_history)} messages: Min: {min(sa)} Max: {max(sa)} Mean: {np.mean(sa)}'
    # client.chat_postMessage(channel=channel_id, text=msg)
    try:
        response = client.files_upload(
            file=graph_path,
            initial_comment=msg,
            channels=channel_id)
        return Response(), 200
    except SlackApiError as e:
        # You will get a SlackApiError if "ok" is False
        assert e.response["ok"] is False
        # str like 'invalid_auth', 'channel_not_found'
        assert e.response["error"]
        print(f"Got an error: {e.response['error']}")
        return Response(), 500
    return Response(), 200


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Slash command analyze. Function is loading and filtering history in channel, where the command was called. Then
    filtered history is analyzed and finally, bot will send graphs to channel.
    :return: Response
    """

    data = request.form
    # Load history
    # print(data)
    channel_id = data.get('channel_id')
    args_text = data.get('text')
    return channel_analysis(channel_id, args_text)


@app.route('/daily_report', methods=['POST'])
def subscribe_to_daily_report():
    data = request.form
    channel_id = data.get('channel_id')
    if channel_id not in daily_report_channels.keys():
        job = schedule.every().day.at("00:00").do(channel_analysis, channel_id=channel_id)
        daily_report_channels[channel_id] = job
        client.chat_postMessage(channel=channel_id, text=f'Successfully subscribe to daily analysis in #{channel_id}')
    else:
        schedule.cancel_job(daily_report_channels[channel_id])
        daily_report_channels.pop(channel_id)
        client.chat_postMessage(channel=channel_id, text=f'Successfully unsubscribe daily analysis.')



@app.route('/user_analysis', methods=['POST'])
def user_analysis():
    """
    Slash command analyze. Function is loading and filtering history in channel, where the command was called. Then
    filtered history is analyzed and finally, bot will send graphs to channel.
    :return: Response
    """

    data = request.form
    channel_id = data.get('channel_id')
    args_text = data.get('text')
    # Convert name to slack id
    return channel_analysis(channel_id, args_text, args_text)


# New function which will create leaderboard of users chatting in channel based on sentiment analysis
@app.route('/leaderboard', methods=['POST'])
def leaderboard():
    """
    Slash command analyze. Function is loading and filtering history in channel, where the command was called. Then
    filtered history is analyzed and finally, bot will send graphs to channel.
    :return: Response
    """

    data = request.form
    channel_id = data.get('channel_id')
    args_text = data.get('text')
    # Convert name to slack id
    args = parse_args(args_text)
    # Load channel history
    history = load_channel_history(channel_id)
    filtered_history = filter_history(history, channel_id, date_range=args)
    messages_by_user = {}
    if filtered_history is None or len(filtered_history) <= 0:
        client.chat_postMessage(channel=channel_id, text='No data to analyze')
        return Response(), 200
    
    for msg in filtered_history:
        if msg.user not in messages_by_user:
            messages_by_user[msg.user] = []
        messages_by_user[msg.user].append(msg.text)
    sa = {}
    for user in messages_by_user:
        sa[user] = analyzer.get_sentiment_analysis(messages_by_user[user])
    sa = {k: v for k, v in sorted(sa.items(), key=lambda item: item[1])}
    # post message to channel with leaderboard but include only top and bottom 3
    if len(sa) <= 6:
        client.chat_postMessage(channel=channel_id, text=f'Leaderboard: {sa}')
        return Response(), 200
    top = {k: sa[k] for k in list(sa)[:3]}
    bottom = {k: sa[k] for k in list(sa)[-3:]}
    client.chat_postMessage(channel=channel_id, text=f'Top 3 users: {top}')
    client.chat_postMessage(channel=channel_id, text=f'Bottom 3 users: {bottom}')

    return Response(), 200


def parse_user_args(args_text: str):
    if args_text is None or len(args_text) <= 0:
        return None
    response = client.users_lookupByEmail(email=args_text)
    if response['ok']:
        return response['user']['id']
    return None


def load_channel_history(channel_id: str) -> []:
    """
    Loads all historical messages in channel.
    :param channel_id: Target Channel ID
    :return: Channel history of messages
    """
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


def filter_history(history: [], channel_id: str, date_range=None, user=None) -> []:
    """
    Filter history. Remove messages from bot and filter messages outside Date Range, if specified.
    :param channel_id: Channel ID
    :param history: History to be filtered
    :param date_range: Filter option Date Range
    :param user: Filter option User
    :return:
    """
    filtered = []
    for msg in history:
        add = True
        # Flag if message is from bot
        if msg['user'] == BOT_ID:
            add = False
        # Flag if message is not in specified date range, if date range arg was passed
        if date_range is not None:
            _from, _to = date_range
            if _from is not None and _to is not None:
                stamp = round(float(msg['ts']))
                date = datetime.fromtimestamp(stamp)
                if _from > date or date > _to:
                    add = False
        # Flag if message is not from specified user, if user arg was passed
        if user is not None:
            if msg['user'] != user:
                add = False


        # If not flagged append to filtered history
        if add:
            filtered.append(Message(msg['text'], msg['user'], msg['ts'], get_reactions(msg, channel_id)))
        thread_msg = extract_thread(msg, channel_id)
        if len(thread_msg) > 0:
            filtered.extend(thread_msg)
    return filtered


def extract_thread(msg: {}, channel_id: str):
    """
    Check if message is thread. If yes get all messages from thread.

    :param channel_id: ID of channel
    :param msg: Message to be checked
    :return: List of all messages from thread (parent message excluded). Empty list if any error or message is not
    thread.
    !! RETURNS list of Message (from message.py) class wrappers, not raw message !!
    """
    if 'thread_ts' not in msg:
        return []

    thread_ts = msg['thread_ts']
    try:
        response = client.conversations_replies(channel=channel_id, ts=thread_ts)
        thread_messages = response["messages"][1:]
        while response['has_more']:
            # sleep(1)
            response = client.conversations_replies(channel=channel_id, ts=thread_ts, limit=200,
                                                    cursor=response['response_metadata']['next_cursor'])
            thread_messages = thread_messages + response["messages"]
        # Print results
        print(f'{len(thread_messages)} messages found in thread.')
        return list(map(lambda m: Message(m['text'], m['user'], m['ts']), thread_messages))
    except SlackApiError as e:
        print("Error in loading threaded msgs.")
    return []


def get_reactions(msg: {}, channel_id: str) -> []:
    reactions = []
    try:
        response = client.reactions_get(channel=channel_id, timestamp=msg['ts'], full=True)
        if msg['ok']:
            reactions = list(map(lambda r: Reaction(r['name'], int(r['count'])), response['message']['reactions']))
            print(f'Get {len(reactions)} reactions')
    except Exception:
        # print('Fetching reactions failed')
        pass
    return reactions


def parse_args(text: str) -> namedtuple:
    """
    Parse arguments from text wrote after analyze command. Date Range is always correct or empty.
    :param text: Arg text
    :return: Date Range parsed from text
    """
    # Definition of named tuple to make DateRange simplier to use
    date_range = namedtuple("DateRange", ["date_from", "date_to"])
    try:
        split = text.split(' ')
        # If no args were given, return empty DateRange
        if len(split) == 0:
            return date_range(None, None)
        # Parse one or two dates
        if len(split) == 1:
            datetime_from = datetime.strptime(split[0], '%d/%m/%Y')
            return date_range(datetime_from, datetime.today())
        else:
            datetime_from = datetime.strptime(split[0], '%d/%m/%Y')
            datetime_to = datetime.strptime(split[1], '%d/%m/%Y')
            delta = datetime_to - datetime_from
            # If dates were flipped, flip them to make correct DateRange
            if delta.days < 0:
                return date_range(datetime_to, datetime_from)
            return date_range(datetime_from, datetime_to)
    # If anything fails, return empty DateRange
    except Exception:
        print('Bad format of args.')
        return date_range(None, None)


if __name__ == '__main__':
    app.run(debug=True)
