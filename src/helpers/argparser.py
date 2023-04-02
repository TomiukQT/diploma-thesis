from collections import namedtuple
from datetime import datetime
import re


def parse_args(args: str, client) -> dict:
    """
    Parse arguments from text wrote after analyze command. Date Range is always correct or empty.
    :param args: Arg text
    :return: Dict with parsed args
    """
    response = {}
    split = args.split(' ')
    # If no args were given, return empty args dict
    if len(split) == 0:
        return response
    # Parse one or two dates
    for s in split:
        arg_name, arg_value = s.split('=')
        if arg_name == 'user':
            response[arg_name] = parse_user_args(arg_value, client)
        elif arg_name == 'date_from' or arg_name == 'date_to':
            response[arg_name] = parse_date_args(arg_value)
        else:
            response[arg_name] = arg_value
        return response


def parse_user_args(args_text: str, client):
    if args_text is None or len(args_text) <= 0:
        return None
    response = client.users_lookupByEmail(email=args_text)
    if response['ok']:
        return response['user']['id']
    return None


def parse_date_args(args_text: str) -> datetime:
    """
    Parse arguments from text wrote after analyze command. Date Range is always correct or empty.
    :param args_text: Arg text
    :return: parsed datetime
    """
    try:
        parsed_datetime = datetime.strptime(args_text, '%d/%m/%Y')
        return parsed_datetime
    # If anything fails, return None
    except Exception:
        print('Bad format of date args.')
        return None


def data_range_from_args(args: {}) -> namedtuple:
    date_range = namedtuple("DateRange", ["date_from", "date_to"])
    date_from = args.get('date_from')
    date_to = args.get('date_to')
    if date_from is None and date_to is None:
        return date_range(None, None)
    if date_from is None:
        return date_range(datetime.strptime('01/01/1970', '%d/%m/%Y'), date_to)
    if date_to is None:
        return date_range(date_from, datetime.today())
    delta = date_to - date_from
    if delta.days < 0:
        return date_range(date_to, date_from)
    return date_range(date_from, date_to)

