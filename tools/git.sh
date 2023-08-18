#! /bin/bash

git branch -f dev origin/dev
git checkout dev
# Make sure you are in dev
git branch 

# push
git add *
git commit -m "commit message"
git push
