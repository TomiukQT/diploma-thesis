from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


class DataUploader:

    def __init__(self) -> None:
        self.auth = GoogleAuth()
        self.drive = GoogleDrive(self.auth)

    def save_file(self, file):
        gfile = drive.CreateFile({'parents': [{'id': '1pzschX3uMbxU0lB5WZ6IlEEeAUE8MZ-t'}]})
        # Read file and set it as the content of this instance.
        gfile.SetContentFile(upload_file)
        gfile.Upload() # Upload the file.

    def messages_to_file(self, messages):
        []