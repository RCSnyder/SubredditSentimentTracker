## cd into ./backend

## set up a python venv
> python3 -m venv env

## activate it
windows:
> env/Scripts/activate

#### alternative for windows
> pip install virtualenv
>
> virtualenv <env-name>
>
> <env-name>\Scripts\activate.bat

## linux:
> source env/bin/activate

## update pip
> python3 -m pip install -U pip

## install requirements
> python3 -m pip install -r requirements.txt

## run flask app
> flask run

## if you want auto reloading upon file change
> python3 app.py


###additional install commands
#####for cuda 10.1:
> pip install torch==1.7.0+cu101 torchvision==0.8.1+cu101 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
> pip install flair

#USAGE:
###set PRAW OAuth variables from reddit tutorial https://praw.readthedocs.io/en/latest/getting_started/authentication.html

######CLIENT_ID = 'kdsdfhdfhgHdQ'
######CLIENT_SECRET = 'MZSsQk9IBTisdfsdfsdfpIS__w'
######PASSWORD = 'Nnotmypassword3'
######USERAGENT = '<short desc with reddit username in it>>'
######USERNAME = '<reddit username>'

###Set up tbm tone analyzer
####lite version 2500 max calls
######IBM_API = "ch7customapifromwebsitesCjxxROa4up"
######IBM_URL = "https://api.us-south.tone-analyzer.watson.cloud.ibm.com/instances/253346b6a9-2ccd-4asdfb-aa91-6sdfsdf98d4ea"

####Methods of extraction are done by modules that take in a csv file and output a csv file 