from flask import Flask, render_template, request
from transformer import Transformer
import json

app = Flask('Project')
t = Transformer()

def convert_to_json_tree(raw_tree):
    if isinstance(raw_tree, list) or \
       isinstance(raw_tree, tuple):
        (l, n, r) = raw_tree
        return {
            'text': {'name': n['conj']},
            'children': [convert_to_json_tree(l),
                         convert_to_json_tree((r))]
        }
    else:
        return {'text': {'title': raw_tree['phrase'],
                         'desc': json.dumps(raw_tree['args'], indent=2)}}

@app.route("/", methods=['GET', 'POST'])
def index():
    converted = None
    text = 'Find all attendees whose post contains dog but not who liked a positive post'
    if 'text' in request.form:
        text = request.form['text']
        raw_tree = t.transform(request.form['text'])
        raw_tree = [{'type': 'phrase', 'phrase': 'whose post contains dog ', 'class': 'post', 'args': [{'post.text': 'dog'}]}, {'type': 'conj', 'conj': 'but'}, {'type': 'phrase', 'phrase': 'not who liked a positive post', 'class': 'like', 'args': []}]
        print(raw_tree)
        converted = convert_to_json_tree(raw_tree)

    return render_template('index.html',
                           tree=converted,
                           text=text)



app.run(debug=True, extra_files=['./templates/index.html'])