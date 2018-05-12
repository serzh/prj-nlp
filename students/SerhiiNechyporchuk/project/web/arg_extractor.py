import re
import itertools
from kanren import run, eq, membero, var, conde, Relation, facts, fact, unifiable
import collections

def replace_quotes_to_double_quotes(text):
    return re.sub(r"(\W)'(\w)", r'\1"\2', re.sub(r"([^s])'(\W|$)", r'\1"\2', text))

def replace_search_strings(phrases):
    guitar_replacements = []
    final_with_guitar = []
    for phrase in phrases:
        repl = []
        for i, match in enumerate(re.finditer(r"\"[^\"]*\"", phrase)):
            repl.append({'match': match[i]})
        phrase = re.sub(r"\"[^\"]*\"", '"guitar"', phrase)
        for i, match in enumerate(re.finditer(r'"guitar"', phrase)):
            repl[i]['start'] = match.start()
            repl[i]['end'] = match.end()
        guitar_replacements.append(repl)
        final_with_guitar.append(phrase)
    return final_with_guitar, guitar_replacements


### RULES
#########

DEPS = [
    "acl",
    "acomp",
    "advcl",
    "advmod",
    "agent",
    "amod",
    "appos",
    "attr",
    "aux",
    "auxpass",
    "cc",
    "ccomp",
    "compound",
    "complm",
    "conj",
    "cop",
    "csubj",
    "csubjpass",
    "dep",
    "det",
    "dobj",
    "expl",
    "hmod",
    "hyph",
    "infmod",
    "intj",
    "iobj",
    "mark",
    "meta",
    "neg",
    "nmod",
    "nn",
    "npadvmod",
    "nsubj",
    "nsubjpass",
    "num",
    "nummod",
    "number",
    "oprd",
    "obj",
    "obl",
    "parataxis",
    "partmod",
    "pcomp",
    "pobj",
    "poss",
    "possessive",
    "preconj",
    "prep",
    "prt",
    "punct",
    "relcl",
    "quantmod",
    "rcmod",
    "root",
    "xcomp",
    "case",
    "dative"
]

def one_of(R, idx, lst):
    l = var('l'+str(idx))
    return conde((R['LEMMA'](idx, l), membero(l, lst)))

def gather_inside_outside_quote_facts(R, doc):
    inside = False
    R['insideq'] = Relation('insideq')
    R['outsideq'] = Relation('outsideq')
    for token in doc:
        if token.text == '"':
            inside = not inside
        else:
            if inside:
                fact(R['insideq'], token.i)
            else:
                fact(R['outsideq'], token.i)

def gather_facts(doc):
    R = {'LEMMA': Relation('LEMMA'),
         'root': Relation('root'),
         'head': Relation('head')}
    for rel in DEPS:
        R[rel] = Relation(rel)
    for tok in doc:
        facts(R['LEMMA'], (tok.i, tok.lemma_))
        if not tok.pos_ in R:
            R[tok.pos_] = Relation(tok.pos_)
        fact(R[tok.pos_], (tok.i))

        facts(R[tok.dep_ if tok.head.i != tok.i else 'root'],
              (tok.head.i if tok.head.i != tok.i else -1, tok.i))
        facts(R['head'], (tok.head.i if tok.head.i != tok.i else -1, tok.i))

        if not tok.ent_type_ in R:
            R[tok.ent_type_] = Relation(tok.ent_type_)
        fact(R[tok.ent_type_], (tok.i))

    gather_inside_outside_quote_facts(R, doc)

    return R

quantifiers = {'at most {}': '<={}',
               'at least {}': '>={}',
               '{} or more': '>={}',
               '{} or less': '<={}',
               'more than {}': '>{}',
               'less than {}': '<{}'}

s2n = {'one': "1", 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6',
       'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10', 'eleven': '11', 'twelve': '12'}

number_regex = re.compile(r'\d+')

def extract_count(doc, replacements, i):
    match = doc[i].text
    mi = len(doc)-1
    for q, r in quantifiers.items():
        if q.format(match) in doc.text[doc[i-2].idx:doc[min(i+3, mi)].idx+len(doc[min(i+3, mi)])]:
            res = None
            if re.match(number_regex, match):
                res = match
            else:
                res = s2n.get(match)
            if res:
                return r.format(res)
    return match

RULES = collections.OrderedDict()
def N(rule):
    RULES[rule.__name__] = rule

def post_replies1(R, not_used):
    def locate():
        action = var('action')
        reply = var('reply')
        post_replies = var('post.repliesCount')
        reply_syns = {'reply', 'comment'}
        return (
            (post_replies, (reply,)),
            [
                conde((R['dobj'](action, reply),),
                      (R['pobj'](action, reply),)),
                one_of(R, reply, reply_syns),
                R['nummod'](reply, post_replies),
                membero(reply, not_used),
                membero(post_replies, not_used)
            ]
        )

    def extract(doc, replacements, i):
        return doc[i].text

    return locate, extract_count


N(post_replies1)

def post_text1(R, not_used):
    def locate():
        about = var('about')
        post_text = var('post.text')
        return (
            (post_text, (about,)),
            [
                R['LEMMA'](about, 'about'),
                R['pobj'](about, post_text),
                membero(post_text, not_used)
            ]
        )

    def extract(doc, replacements, i):
        matches = [repl for repl in replacements if (repl['start'] >= i) and (i < repl['end'])]
        if matches:
            return matches[0]['match']
        else:
            return doc[i].text

    return locate, extract


N(post_text1)

def post_text2(R, not_used):
    def locate():
        action = var('action')
        post_text = var('post.text')
        return (
            (post_text, (action,)),
            [
                R['LEMMA'](action, 'contain'),
                R['dobj'](action, post_text),
                membero(post_text, not_used)
            ]
        )

    def extract(doc, replacements, i):
        matches = [repl for repl in replacements if (repl['start'] >= i) and (i < repl['end'])]
        if matches:
            return matches[0]['match']
        else:
            return doc[i].text

    return locate, extract


N(post_text2)

def post_text3(R, not_used):
    def locate():
        post_text = var('post.text')
        prep = var('prep')
        return (
            (post_text, (prep,)),
            [
                one_of(R, prep, {'with'}),
                R['pobj'](prep, post_text),
                R['outsideq'](prep),
                R['insideq'](post_text),
                membero(prep, not_used),
                membero(post_text, not_used)
            ]
        )

    def extract(doc, replacements, i):
        matches = [repl for repl in replacements if (repl['start'] >= i) and (i < repl['end'])]
        if matches:
            return matches[0]['match']

    return locate, extract


N(post_text3)

def post_text4(R, not_used):
    def locate():
        post_text = var('post.text')
        post = var('post')
        post_syns = {'message', 'content', 'post'}
        contain = var('contain')
        return (
            (post_text, (contain, post_text)),
            [
                one_of(R, post, post_syns),
                R['LEMMA'](contain, 'contain'),
                conde((R['relcl'](post, contain),),
                      (R['acl'](post, contain),)),
                R['oprd'](contain, post_text),
                membero(contain, not_used),
                membero(post_text, not_used)
            ]
        )

    def extract(doc, replacements, i):
        matches = [repl for repl in replacements if (repl['start'] >= i) and (i < repl['end'])]
        if matches:
            return matches[0]['match']

    return locate, extract


N(post_text4)

def post_likes1(R, not_used):
    def locate():
        with_prep = var('with')
        with_what = var('with_what')
        post_likes = var('post.likes_count')
        return (
            (post_likes, (with_what,)),
            [
                R['LEMMA'](with_prep, 'with'),
                R['pobj'](with_prep, with_what),
                R['LEMMA'](with_what, 'like'),
                R['nummod'](with_what, post_likes),
                membero(post_likes, not_used),
                membero(with_what, not_used)
            ]
        )

    def extract(doc, replacements, i):
        return doc[i].text

    return locate, extract_count


N(post_likes1)

def post_likes2(R, not_used):
    def locate():
        like_action = var('like')
        times = var('times')
        post_likes = var('post.likes_count')
        return (
            (post_likes, (like_action, times)),
            [
                R['LEMMA'](like_action, 'like'),
                R['npadvmod'](like_action, times),
                R['nummod'](times, post_likes),
                membero(like_action, not_used),
                membero(times, not_used),
                membero(post_likes, not_used)
            ]
        )

    def extract(doc, replacements, i):
        return doc[i].text

    return locate, extract_count


N(post_likes2)

def extract_mention(doc, replacements, i):
    children = list(doc[i].children)
    compound = list(filter(lambda x: x.dep_ == 'compound', children))
    sort = sorted(compound + [doc[i]], key=lambda tok: tok.i)
    s = sort[0].idx
    e = sort[-1].idx + len(sort[-1].text)
    return doc.text[s:e]

def post_mention1(R, not_used):
    def locate():
        mention = var('mention')
        post_mention = var('post.mention')
        return (
            (post_mention, (mention,)),
            [
                R['LEMMA'](mention, 'mention'),
                R['dobj'](mention, post_mention),
                membero(mention, not_used),
                membero(post_mention, not_used)
            ]
        )

    def extract(doc, replacements, i):
        return extract_mention(doc, replacements, i)

    return locate, extract_mention


N(post_mention1)

def post_mention2(R, not_used):
    def locate():
        mention = var('mention')
        post_mention = var('post.mention')
        return (
            (post_mention, (mention,)),
            [
                R['LEMMA'](mention, 'mention'),
                R['poss'](mention, post_mention),
                membero(mention, not_used),
                membero(post_mention, not_used)
            ]
        )

    def extract(doc, replacements, i):
        return doc[i].text

    return locate, extract_mention


N(post_mention2)

def post_mention3(R, not_used):
    def locate():
        mention = var('mention')
        prep = var('prep')
        post_mention = var('post.mention')
        return (
            (post_mention, (mention, prep)),
            [
                one_of(R, mention, {'mention'}),
                R['prep'](mention, prep),
                one_of(R, prep, {'of'}),
                R['pobj'](prep, post_mention),
                membero(mention, not_used),
                membero(post_mention, not_used),
                membero(prep, not_used)
            ]
        )

    def extract(doc, replacements, i):
        return doc[i].text

    return locate, extract_mention


N(post_mention3)

def post_sentiment1(R, not_used):
    def locate():
        post_sentiment = var('post.sentiment')
        post = var('post')
        return (
            (post_sentiment, ()),
            [
                R['LEMMA'](post, 'post'),
                R['amod'](post, post_sentiment),
                membero(post_sentiment, not_used)
            ]
        )

    def extract(doc, replacements, i):
        return doc[i].text

    return locate, extract


N(post_sentiment1)

def post_sentiment2(R, not_used):
    def locate():
        post_sentiment = var('post.sentiment')
        sentiment = var('sentiment')
        return (
            (post_sentiment, (sentiment,)),
            [
                R['LEMMA'](sentiment, 'sentiment'),
                R['amod'](sentiment, post_sentiment),
                membero(post_sentiment, not_used)
            ]
        )

    def extract(doc, replacements, i):
        return doc[i].text

    return locate, extract


N(post_sentiment2)

def post_count1(R, not_used):
    def locate():
        post_count = var('post.count')
        post = var('post')
        post_syns = {'message', 'content', 'post'}
        return (
            (post_count, ()),
            [
                one_of(R, post, post_syns),
                R['nummod'](post, post_count),
                membero(post_count, not_used)
            ]
        )

    def extract(doc, replacements, i):
        return doc[i].text

    return locate, extract_count


N(post_count1)

def run_rule(R, not_used, locate_fn):
    (vars, stmnts) = locate_fn()
    r = run(1, vars, *stmnts)
    if r:
        results = itertools.takewhile(lambda x: not isinstance(x, tuple), r[0])
        used = list(itertools.dropwhile(lambda x: not isinstance(x, tuple), r[0]))[0]
        return {variable.token: val for variable,val in zip(vars,results)},used
    else:
        return {},[]

def run_rules_on_doc(doc, grepls, rules=RULES, verbose=False):
    R = gather_facts(doc)
    bindings = {}
    not_used = set(range(len(doc)))
    for name, rule in rules.items():
        locate_fn, extract_fn = rule(R, not_used)
        new_bindings, used = run_rule(R, not_used, locate_fn)
        new_bindings = {variable: extract_fn(doc, grepls, val) for variable,val in new_bindings.items()}
        if new_bindings and verbose:
            print('Applied rule', name, new_bindings)
        new_bindings.update(bindings)
        bindings = new_bindings
        not_used = not_used - (set(used) | set(new_bindings.values()))
    return bindings

def run_rules(docs, grepls_col):
    res = []
    for doc, grepls in zip(docs, grepls_col):
        res.append(run_rules_on_doc(doc, grepls))
    return res

def extract_args(nlp, part, verbose=False):
    if verbose:
        print("Extracting args: {}".format(part))
    if part['class'] != 'post':
        part['args'] = []
    else:
        phrase = part['phrase']
        text = replace_quotes_to_double_quotes(phrase)
        final, grepls = replace_search_strings([text.strip()])
        final = [nlp(f) for f in final]
        part['args'] = run_rules(final, grepls)
    return part