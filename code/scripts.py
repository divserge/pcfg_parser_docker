from __future__ import print_function

import copy
import sys, time
import numpy as np

from nltk.corpus import treebank
from nltk import Tree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from pparser import Parser

def parse_grammars(path):
    grammars = {}
    root_syms = {}

    with open(path, 'r') as file:
        for line in file:
            content = line.split()
            if content[0] in ['interminals', 'preterminals']:
                for sym in content[1:]:
                    grammars[sym] = []
            elif content[0] == 'root':
                root_syms[content[1]] = float(content[-1].split(':')[-1])
            elif content[0] == 'term':
                fr = content[1]
                to = content[3]
                prob = float(content[-1].split(':')[-1])
                grammars[fr].append((to, prob))
            elif content[0] == 'binary':
                fr = content[1]
                to = content[3] + ' ' + content[4]
                prob = float(content[-1].split(':')[-1])
                grammars[fr].append((to, prob))

    return grammars, root_syms

def tuple_to_str(inp_tuple):
    out_string = "("
    for elem in inp_tuple:
        if type(elem) == str:
            out_string += elem + " "
        else:
            out_string += tuple_to_str(elem)
    out_string += ")"
    return out_string


def writeout_parses(parser, input_file, output_file):
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        k = 0
        for l in fin:
            sent = [w.split('^')[0] for w in l.split()[:-1]][:20]
            line = ' '.join(sent)
            kek = parser.parse(line)
            k += 1
            if k % 100 == 0:
                fout.flush()
            print(tuple_to_str(kek), file=fout)


def test_parser(approximation, rules_path, rank, input_file, output_folder):
    rules, root = parse_grammars(path=rules_path)
    p = Parser(rules, root, approximation=approximation, rank=rank)
    writeout_parses(
        p,
        input_file,
        '{}/test_{}_{}.txt'.format(output_folder, approximation, rank)
    )


#Import functions for measure F1-score
def precision(golds, parses ):
    """Return the proportion of brackets in the suggested parse tree that are
    in the gold standard. """

    total = 0
    successes = 0

    for (gold, parse) in zip(golds, parses):
        if parse is not None:
            parsebrackets = list_brackets(parse)
            goldbrackets = list_brackets(gold)
            
            candidate = parsebrackets
            gold = goldbrackets

            total += len(candidate)
            for bracket in candidate:
                if bracket in gold:
                    successes += 1
    if total == 0 :
        return 0.0 
    return float(successes) / float(total)


def recall(golds, parses):
    """Return the proportion of brackets in the gold standard that are in the
    suggested parse tree."""

    total = 0
    successes = 0

    for (gold, parse) in zip(golds, parses):
        
        goldbrackets = list_brackets(gold)
        gold = goldbrackets

        total += len(gold)

        if parse is not None:
            parsebrackets = list_brackets(parse)
            candidate = parsebrackets

            for bracket in gold:
                if bracket in candidate:
                    successes += 1
    if total==0:
        return 0 
    return float(successes) / float(total)

def f1_score( p , r ):
    """Return the F1 score of the parse with respect to the gold standard"""
    if ( p == 0 and r == 0 ):
        return 0 
    f1 = (2.0*p*r)/(p+r)
    return f1
    
def words_to_indexes(tree):
    """Return a new tree based on the original tree, such that the leaf values
    are replaced by their indeces."""

    out = copy.deepcopy(tree)
    leaves = out.leaves()
    for index in range(0, len(leaves)):
        path = out.leaf_treeposition(index)
        out[path] = index + 1
    return out

def list_brackets(tree):
    tree = words_to_indexes(tree)

    def not_pos_tag(tr):
        return tr.height() > 2

    subtrees = tree.subtrees(filter=not_pos_tag)
    return [(sub.leaves()[0], sub.leaves()[-1]) for sub in subtrees]


def measure_score_and_plot(gt_predictions_file, test_predictions_file, name, output_file):
    
    with open(gt_predictions_file) as f:
        gold = f.readlines()
    gt = [x.strip() for x in gold]

    with open(test_predictions_file) as f:
        test = f.readlines()
    test = [x.strip() for x in test]

    gold = [ Tree.fromstring(gt[i]) for i in range(len(gt)) ]
    parse = [ Tree.fromstring(test[i]) for i in range(len(test)) ]

    # Calculate the scores
    pscore = np.asarray([ precision([gold[i]], [parse[i]]) for i in range(len(gold)) ])
    rscore = np.asarray([ recall([gold[i]], [parse[i]]) for i in range(len(gold)) ])
    f1score =  np.asarray( [ f1_score( pscore[i] , rscore[i]) for i in range(len(gold)) ] ) 
    f1score = f1score[ f1score != 0 ]

    n, bins, patches = plt.hist(f1score, 50, normed=1, facecolor='green', alpha=0.75)

    # add a 'best fit' line
    y = mlab.normpdf( bins, f1score.mean(), np.sqrt(f1score.var()))
    l = plt.plot(bins, y, 'r--', linewidth=1)

    plt.xlabel('F1-Score')
    plt.ylabel('Frequency')
    plt.title(r'$\mathrm{F1 Score\ of\ %s \ vs\ Exact:}\ \mu=%.4f,\ \sigma=%.4f$' %(name, f1score.mean(), np.sqrt(f1score.var())))
    plt.axis([0, 1, 0, 10])
    plt.grid(True)

    plt.savefig(output_file)


def measure_efficiency_and_plot(rules_file, sentences_file, sample_size, output_file):
    
    rules, root = parse_grammars(rules_file)
    sentences = []

    with open (sentences_file) as f:
        for l in f:
            sent = [w.split('^')[0] for w in l.split()[:-1]][:20]
            sentences.append(' '.join(sent))

    np.random.seed(42)
    subsample = np.random.choice(sentences, sample_size)

    times = {}

    for decomposition in ['tt']:
        times[decomposition] = {}
        
        for rank in [10, 50, 75, 100, 125, 150]:
            
            times[decomposition][rank] = []
            p = Parser(rules=rules, roots=root, approximation=decomposition, rank=rank)
            for sent in subsample:
                start = time.time()
                p.parse(sent)
                end = time.time()
                times[decomposition][rank].append((end - start) / 1.0)

    times['exact'] = []
    p = Parser(rules=rules, roots=root, approximation='exact', rank=0)
    for sent in subsample:
        start = time.time()
        p.parse(sent)
        end = time.time()
        times['exact'].append(end - start)

    lengths = [len(sent.split()) for sent in subsample]
    max_length = np.where(np.array(lengths) == np.max(lengths))[0]

    ranks = [150, 125, 100, 75, 50, 10]

    lines_tt = []
    for sent_index in max_length:
        lines_tt.append([])
        for i, rank in enumerate(ranks):
            lines_tt[-1].append(times['tt'][rank][sent_index])
            
    mean_line_tt = np.array(lines_tt).mean(axis = 0)
    mean_line_exact = [np.array(times['exact'])[max_length].mean()] * len(ranks)


    plt.figure(figsize=(7,5))
    for line in lines_tt:
        plt.plot(ranks, line, c='r', alpha=0.4, linestyle='--')

    plt.plot(ranks, mean_line_tt, c='r', alpha=0.8, label='tensor train')
    plt.plot(ranks, mean_line_exact, c='black', alpha=1.0, label='exact')
    plt.xlabel('rank')
    plt.ylabel('parse time in seconds')
    plt.title('time measurements of parsing on several sentences of length 20')
    plt.xticks(ranks)
    plt.legend()
    plt.grid()

    plt.savefig(output_file)