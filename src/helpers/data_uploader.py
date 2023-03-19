from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import json
import os
import pandas as pd


class DataUploader:

    def __init__(self) -> None:
        self.auth = GoogleAuth()
        self.setup_auth()
        self.drive = GoogleDrive(self.auth)

    def setup_auth(self):
        # Try to load saved client credentials
        credentials = json.loads(os.environ.get('GOOGLE_CREDENTIALS'))
        with open('creds.json', 'w') as fp:
            json.dump(credentials, fp)
        self.gauth.LoadCredentialsFile("creds.json")
        if self.gauth.credentials is None:
            # Authenticate if they're not there
            self.gauth.LocalWebserverAuth()
        elif self.gauth.access_token_expired:
            # Refresh them if expired
            self.gauth.Refresh()
        else:
            # Initialize the saved creds
            self.gauth.Authorize()

    def save_file(self, file, remove_after_upload=True):
        gfile = self.drive.CreateFile({'parents': [{'id': '18104w-v_kKoYurCUrkJ_9_EgE6vjkVee'}]})
        # Read file and set it as the content of this instance.
        gfile.SetContentFile(file)
        try:
            gfile.Upload()
        finally:
            gfile.content.close() # Upload the file.
        if remove_after_upload:
            os.remove(file)

    def messages_to_file(messages, file_name='new_file') -> str:
        file_name += f'_{str(pd.Timestamp.now().timestamp())}.csv'
        file_name = file_name.strip()
        path = f'tmp/{file_name}'
        #df = pd.DataFrame([m.__dict__ for m in messages])
        df = pd.DataFrame.from_records([vars(m) for m in messages], exclude=['reactions'])
        # Reaction to column
        df.insert(4, "reactions", [[(r.name, r.count) for r in m.reactions] for m in messages], True)
        df.to_csv(path, sep=';')
        return path