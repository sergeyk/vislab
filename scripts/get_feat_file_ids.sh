FEAT_DIR=$HOME/work/aphrodite/data/feats
find $FEAT_DIR -name "*.txt.gz" -exec sh -c "echo {} && zcat {} | cut -d ' ' -f 2 | cut -c 3- > {}.ids.txt" \;
