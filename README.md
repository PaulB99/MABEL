Welcome to the reposity for my final year project! To get started, ensure you have at least Python 3.6 installed with the following libraries:
for core functionality:
warnings
torch
torchtext
transformers
simpletransformers

for web app:
flask
gevent

for command line app:
simple_term_menu

for testing and training scripts:
matplotlib
pandas
unittest
datasets


The pretrained models can be found at the following onedrive link:

https://bham-my.sharepoint.com/personal/pab734_student_bham_ac_uk/_layouts/15/guestaccess.aspx?guestaccesstoken=b8CXLs6W1W1lVDYiSQ%2BkHkllrxZWmdb0JjKYoE%2B1yP0%3D&folderid=2_07d752226ee0c45eeb1b196cce577e3ae&rev=1&e=8b7bLa

It is restricted to people in the University of Birmingham so be aware of that 

The SHA256 hashes to go with them are as follows:

distilbert.pt - 9ef3ec2aa64aafe13dece274acf3dd32f3cf0eecc59c21769494cd74d167aece
base_model.pt - 1f8162c1382f3af7e458fe3811f9e12bc1313956d829b7051f102f345c2740a7
seq2seq.pt - 1147846a06df0f635a322ed4012743d779b7ed70c562a3735f6b05d485cfb4b6
miniseq2seq.pt - f9c867e40cff5e574842240c673a7f0384b0075b5e74380a2c6d189d6a4d28ff
bart/pytorch_model.bin - 4ab5c5bf619f05aeb3ad3faed11a908a4644cf10449ed2dc2c6edc17f8788b49

Download the models of choice, or the whole lot!

Put them into /cache but maintain the file structure from the onedrive (/cache/taggers or /cache/neutralisers)

To run the web app, run the Python file /src/app/app.py

To run the command line app, run the Python file /src/app/cmd_app.py

Enjoy!