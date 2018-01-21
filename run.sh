#!/bin/bash
# preparing the data: the experiments might take a long time, therefore think carefully about 
# how many sentences to take
head -10 data/test_lim.toparse >> data/test.toparse
cd code && python run.py
cd ../tex && pdflatex paper.tex && cp paper.pdf ../results/
