import io

with io.open('datasets/twitter.csv','r',encoding='latin-1',errors='ignore') as infile, \
     io.open('datasets/twitter_parsed.csv','w',encoding='ascii',errors='ignore') as outfile:
    for line in infile:
        print(*line.split(), file=outfile)