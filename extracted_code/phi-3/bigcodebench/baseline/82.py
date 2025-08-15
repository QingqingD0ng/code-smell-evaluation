from flask import Flask, render_template, redirect, url_for
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Length
from werkzeug.security import generate_password_hash, check_password_hash

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=25)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=8, max=80)])
    submit = SubmitField('Log In')

login_manager = LoginManager()

@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)

class User(UserMixin):
    # Dummy user class
    pass

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'  # Replace with your secret key
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['TEMPLATES_LOAD_PREFIX'] = ''
app.config['TEMPLATES_APP_DIRS'] = True
app.config['TEMPLATES_FLASH_CACHE_MAX_AGE'] = 5 * 60
app.config['TEMPLATES_FLASH_CACHE_TYPE'] = 'null'
app.config['SECURITY_PASSWORD_SALT'] = 'your_password_salt'  # Replace with your password salt

login_manager.init_app(app)

@app.route('/')
def index():
    return render_template('index.html')  # Replace with your homepage template

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.get(form.username.