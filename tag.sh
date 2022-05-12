#!/bin/bash

if git tag | grep $1 > /dev/null; then
    git tag -d $1
    git push --delete origin $1
fi

if [ -z $2 ]; then
    git tag $1
else
    git tag -a $1 -m "$2"
fi

git push origin $1
