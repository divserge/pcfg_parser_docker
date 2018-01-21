import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
from scripts import test_parser, measure_score_and_plot, measure_efficiency_and_plot

if __name__ == '__main__':

    nltk.download('treebank')

    print('measuring time')
    measure_efficiency_and_plot(
        '../data/grammar-prune.txt',
        '../data/test.toparse',
        5,
        '../tex/tt-time.png'
    )

    print('making exact predictions')
    test_parser(
        'exact',
        '../data/grammar-prune.txt',
        0,
        '../data/test.toparse',
        '../results'
        )

    print('making predictions with tt-10')
    test_parser(
        'tt',
        '../data/grammar-prune.txt',
        10,
        '../data/test.toparse',
        '../results'
        )

    print('making predictions with tt-100')
    test_parser(
        'tt',
        '../data/grammar-prune.txt',
        100,
        '../data/test.toparse',
        '../results'
        )

    print('drawing pictures')
    measure_score_and_plot(
        '../results/test_exact_0.txt',
        '../results/test_tt_10.txt',
        'TT Rank 10',
        '../tex/test_tt_10.png'
    )

    measure_score_and_plot(
        '../results/test_exact_0.txt',
        '../results/test_tt_100.txt',
        'TT Rank 100',
        '../tex/test_tt_100.png'
    )