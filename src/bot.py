import json
import os
import statistics

from collections import namedtuple
import numpy as np
import slack
from flask import Flask, Response, request, stream_with_context
from slack.errors import SlackApiError
from slackeventsapi import SlackEventAdapter
from helpers.message import Message, Reaction
from helpers.argparser import parse_args, data_range_from_args
from analyzer.analyzer import Analyzer
from analyzer.time_analyzer import TimeSeriesAnalyzer
from analyzer.data_transformation import MessageTranslator
from helpers.data_uploader import DataUploader

from datetime import datetime
import schedule

TOKEN = os.environ['SLACK_BOT_TOKEN']
SIGNING_SECRET = os.environ['SLACK_BOT_SIGNING_SECRET']

app = Flask(__name__)
slack_event_adapter = SlackEventAdapter(SIGNING_SECRET, '/slack/events', app)

client = slack.WebClient(token=TOKEN)
BOT_ID = client.api_call('auth.test')['user_id']
analyzer = Analyzer('models', 'old/model_new.sav', 'vectorizer.sav')
ts_analyzer = TimeSeriesAnalyzer()
data_uploader = DataUploader()
message_translator = MessageTranslator()

daily_report_channels = {}


@slack_event_adapter.on('challenge')
def url_auth(payload):
    print(payload)


def channel_analysis(channel_id: str, args: {}) -> Response:
    user_id = args.get('user')
    date_range = data_range_from_args(args)
    history = load_channel_history(channel_id)
    filtered_history = filter_history(history, channel_id, date_range=date_range, user=user_id)
    if len(filtered_history) <= 0:
        client.chat_postMessage(channel=channel_id, text='No data to analyze')
        return Response(), 200
    # Save messages
    data_uploader.save_file(data_uploader.messages_to_file(filtered_history))

    # Translate if needed
    message_translator.translate_messages(filtered_history)

    # Analyze messages
    sa = analyzer.get_sentiment_analysis(filtered_history)
    # Analyze SA
    date_indexed_data = ts_analyzer.index_dates(sa, [m.date for m in filtered_history])
    trend = ts_analyzer.extract_trend(date_indexed_data) #start_date=ts_analyzer.parse_date('last_week')
    # Predictions
    prediction_data = ts_analyzer.get_predictions(date_indexed_data)

    # Print Graph
    graph_path1, graph_path2, graph_path3 = analyzer.get_plot(plot_path='out/graphs/out_graph', trend_data=trend, predictions_data=prediction_data)

    msgs = [f'Analysed {len(filtered_history)} messages: Min: {min(sa)} Max: {max(sa)} Mean: {np.mean(sa)}',
            f'Trend: {trend}',
            f'Predictions: {prediction_data.trend}']
    # client.chat_postMessage(channel=channel_id, text=msg)
    try:
        for graph_path, msg in zip([graph_path1, graph_path2, graph_path3], msgs):
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
    channel_id = data.get('channel_id')
    args_text = data.get('text')
    return channel_analysis(channel_id, parse_args(args_text, client))


@app.route('/analyze_channel', methods=['POST'])
def analyze_channel():
    """
    Slash command analyze. Function is loading and filtering history in channel, where the command was called. Then
    filtered history is analyzed and finally, bot will send graphs to channel.
    :return: Response
    """

    data = request.form
    channel_id = data.get('channel_id')
    args_text = data.get('text')
    args = parse_args(args_text, client)
    if args['channel'] is None:
        client.chat_postMessage(channel=channel_id, text='No channel specified')
        return Response(), 200
    return channel_analysis(args['channel'], args)

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
    args = parse_args(args_text)
    # Convert name to slack id
    return channel_analysis(channel_id, args)


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
    date_range = data_range_from_args(args)
    # Load channel history
    history = load_channel_history(channel_id)
    filtered_history = filter_history(history, channel_id, date_range=date_range)
    messages_by_user = {}
    if filtered_history is None or len(filtered_history) <= 0:
        client.chat_postMessage(channel=channel_id, text='No data to analyze')
        return Response(), 200
    
    for msg in filtered_history:
        if msg.user not in messages_by_user:
            messages_by_user[msg.user] = []
        messages_by_user[msg.user].append(msg)
    sa = {}
    for user in messages_by_user:
        sa[user] = statistics.mean(analyzer.get_sentiment_analysis(messages_by_user[user]))
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





if __name__ == '__main__':
    app.run(debug=True, port=5002)
