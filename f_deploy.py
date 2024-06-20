from flask import Flask,request, url_for, redirect, render_template,session
import pickle
import io
import re
from PyPDF2 import PdfReader
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))


app.secret_key = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

with app.app_context():
    db.create_all()

@app.route('/')
def hello_world():
    return render_template("home1.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    prediction=model.predict(final)
    #output='{0:.{1}f}'.format(prediction[0][1], 2)
    output = str(prediction)
    if output>str(0.5):
        return render_template('index.html',pred='Your Forest is in Danger.\nProbability of fire occuring is {}'.format(output))
    else:
        return render_template('index.html',pred='Your Forest is safe.\n Probability of fire occuring is {}'.format(output))
@app.route('/login')
def login():
     return render_template('login.html')
@app.route('/register_data', methods=['POST'])
def register_data():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password)
        new_user = User(username=username, password_hash=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/register')
def register():
    return render_template("register.html")

@app.route('/ml')
def history():
    return render_template("history.html")

@app.route('/index')
def index():
    return render_template('index.html')
@app.route('/login_data', methods=['POST'])
def login_data():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            session['username'] = username
            return redirect(url_for('index'))
        else:
            return render_template('login.html', message='Invalid username or password')
    return render_template('index.html')
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    if file:
       
        pdf_content = file.read()
        text = process_pdf(pdf_content)
        scores = ml_model(text)
        print(scores)
        total_marks = sum(scores)
    
        result = 'Pass' if total_marks >= 5 else 'Fail'
    
        return render_template('result.html', scores=scores, total_marks=total_marks, result=result)
   

def ml_model(students_paragraph):
    scores = []
    from transformers import BertTokenizer, BertModel
    import torch
    from gensim.models import Word2Vec
    import gensim.downloader as api
    wv = api.load('word2vec-google-news-300')
    
    from scipy import spatial
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

   
    def tokenize_sentence(sentence):
        tokens = tokenizer.tokenize(sentence)
        token_ids = [tokenizer.cls_token_id]  

        for token in tokens:
            try:
                token_ids.append(tokenizer.convert_tokens_to_ids(token))
            except KeyError:
               
                token_ids.append(tokenizer.convert_tokens_to_ids('[UNK]'))

        token_ids.append(tokenizer.sep_token_id)  # End with [SEP] token
        attention_mask = [1] * len(token_ids)

        return token_ids, attention_mask

    # Function to extract embeddings from the [CLS] token
    def get_cls_embedding(sentence):
        token_ids, attention_mask = tokenize_sentence(sentence)

        # Convert to tensor
        token_ids = torch.tensor([token_ids])
        attention_mask = torch.tensor([attention_mask])

        # Obtain contextual embeddings
        with torch.no_grad():
            outputs = model(token_ids, attention_mask=attention_mask)

        # Extract embeddings for the [CLS] token
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()

        return cls_embedding

    # Load Word2Vec model for semantic similarity


    # Function to calculate semantic similarity between two sentences
    def calculate_semantic_similarity(sentence1, sentence2):
        tokens1 = sentence1.split()
        tokens2 = sentence2.split()

        # Calculate average word embeddings
        avg_embedding1 = sum([wv[word] for word in tokens1 if word in wv]) / len(tokens1)
        avg_embedding2 = sum([wv[word] for word in tokens2 if word in wv]) / len(tokens2)

        # Calculate cosine similarity
        similarity = 1 - spatial.distance.cosine(avg_embedding1, avg_embedding2)

        return similarity
    paragraphs_teacher = [
    "1.Machine Learning is a subset of artiﬁcial intelligence (AI) that focuses on developing algorithms and statistical models that enable computers to learn from and make predictions or decisions based on data, without being explicitly programmed to perform the task. ML algorithms allow computers to learn from experience (data) and improve their performance over time without human intervention. This is achieved by training the algorithm on a dataset to learn patterns and relationships within the data, which can then be used to make predictions or decisions on new, unseen data. ML is used in various applications such as image recognition, natural language processing, and recommendation systems.",
    "2.Deep Learning is a subﬁeld of machine learning that focuses on neural networks with multiple layers (deep neural networks). Deep learning algorithms attempt to model high-level abstractions in data by using multiple processing layers, with each layer learning to represent data at a diﬀerent level of abstraction. Deep learning has been particularly successful in areas such as computer vision, natural language processing, and speech recognition, where it has achieved state-of-the-art performance in various tasks. Deep learning models are capable of learning complex patterns and relationships in data, making them suitable for tasks that require a high level of understanding and abstraction."
    ]

    # Input sentences
    for i in range(len(paragraphs_teacher)):
        student_answer = students_paragraph[i]
       
        teacher_answer = paragraphs_teacher[i]
      

        import re
        import nltk
        nltk.download('stopwords')
        from nltk.corpus import stopwords
        from nltk.stem.porter import PorterStemmer
        corpus = []

        review = re.sub('[^a-zA-Z]', ' ', student_answer)
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        stud_review = sorted(review)

        review = re.sub('[^a-zA-Z]', ' ', teacher_answer)
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        teacher_review = sorted(review)
        def check_keyword(stud_review, teacher_review):
            if stud_review == teacher_review:
                return 1

        # Calculate cosine similarity between BERT embeddings
        sentence_embedding_teacher = get_cls_embedding(teacher_answer)
        sentence_embedding_student = get_cls_embedding(student_answer)
        cosine_similarity = torch.cosine_similarity(sentence_embedding_teacher, sentence_embedding_student, dim=0).item()

        # Calculate semantic similarity between Word2Vec embeddings
        semantic_similarity = calculate_semantic_similarity(teacher_answer, student_answer)

        # Combine both similarities
        combined_similarity = (cosine_similarity + semantic_similarity) / 2

        print("Cosine Similarity:", cosine_similarity)
        print("Semantic Similarity:", semantic_similarity)
        print("Combined Similarity:", combined_similarity)

        # Determine if the answer is correct based on combined similarity
        if combined_similarity > 0.96 or check_keyword(stud_review, teacher_review)==1:
            print("score:5")
            scores.append(5)
        elif combined_similarity > 0.94 and combined_similarity <= 0.959:
            print("score:3")
            scores.append(3)
        else:


            print("score:0")
            scores.append(0)
    return scores




def process_pdf(pdf_content):
    pdf_reader = PdfReader(io.BytesIO(pdf_content))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    

    paragraphs = []
    current_paragraph = ""

    for line in text.split("\n"):
        if re.match(r'^\d+\.', line.strip()):
            if current_paragraph:
                paragraphs.append(current_paragraph.strip())
            current_paragraph = line.strip()
        else:
            current_paragraph += " " + line.strip()
    if current_paragraph:
        paragraphs.append(current_paragraph.strip())
    
    return paragraphs





if __name__ == '__main__':
    app.run(debug=True)