from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import json
import numpy as np
import tensorflow as tf
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
from datetime import date
import os


app = Flask(__name__)

nltk.download('punkt')

spam_model = tf.keras.models.load_model('models\spam_classifier_model.h5')
word2vec_model = Word2Vec.load('models\word2vec_model.model') 

def preprocess_text(text):
    tokens = word_tokenize(text.lower()) 
    tokens = [word for word in tokens if word.isalpha()]  
    return tokens

def get_word2vec_embeddings(tokens):
    embeddings = []
    for word in tokens:
        if word in word2vec_model.wv:
            embeddings.append(word2vec_model.wv[word])
        else:
            embeddings.append([0] * 100)  
    return embeddings

def predict_spam_or_ham(text):
  
    tokens = preprocess_text(text)
    embeddings = get_word2vec_embeddings(tokens)
    max_length = 1000
    embeddings = np.array(embeddings)
    embeddings = tf.keras.preprocessing.sequence.pad_sequences([embeddings], maxlen=max_length, padding='post', truncating='post', dtype='float32')
    prediction = spam_model.predict(embeddings)    
    return 'spam' if prediction > 0.5 else 'ham'


db_dir = os.path.join(os.getcwd(), 'database')
os.makedirs(db_dir, exist_ok=True)
db_path = os.path.join(db_dir, 'emails.db')


app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


class Email(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sender = db.Column(db.String(200), nullable=False)
    subject = db.Column(db.String(200), nullable=False)
    date = db.Column(db.String(100), nullable=False)
    spam = db.Column(db.Boolean, nullable=False)
    content = db.Column(db.Text, nullable=False)

with app.app_context():
    
    db.create_all()


@app.route("/load_emails")
def load_emails():
    with open("database\data.json", "r") as f:
        emails = json.load(f)
    return {"emails": emails}    


@app.route("/add")
def add():

    with open("database\data.json", "r") as f:
        emails = json.load(f)
    for i in emails:            
         email1 = Email(sender = i["sender"],subject=i["subject"],date = i["date"],spam = i["spam"], content=i["content"],)
         db.session.add(email1)  
         db.session.commit()
    return "sucessfull"
   


@app.route('/')
def email_list():
    emails = Email.query.order_by(Email.id.desc()).all()
   
    return render_template('email_list.html', emails=emails)


@app.route('/email/<int:email_id>')
def email_content(email_id):
    email = Email.query.get_or_404(email_id)  
    spm = predict_spam_or_ham(email.content)
    return render_template('email_content.html', email=email, isSpam = spm)


@app.route('/compose')
def compose_email():   
    return render_template('compose.html')


@app.route("/send_email",methods = ["POST"])
def send_mail():
    data = request.form
    current_date = date.today()
    formatted_date = current_date.strftime("%Y-%m-%d")       
    email1 = Email(sender = data["to"],subject=data["subject"],date = formatted_date,spam = False, content=data["content"],)
    db.session.add(email1)  
    db.session.commit()
    return redirect("/")