from flask import Flask

UPLOAD_FOLDER = r'C:\Users\207065\Desktop\workspace\backend\Files'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


