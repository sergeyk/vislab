FEAT_DIR=$HOME/work/vislab-git/data/feats
find $FEAT_DIR -name "*.txt.gz" -exec sh -c "echo {} && gzcat {} | cut -d ' ' -f 2 | cut -c 3- > {}.ids.txt" \;
