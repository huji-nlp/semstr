#!/usr/bin/env bash

ACTION=${TEST_SUITE%-*}
FORMAT=${TEST_SUITE#*-}
if [[ "$FORMAT" == ucca ]]; then
    SUFFIX="xml"
else
    SUFFIX="$FORMAT"
fi

# download data
if ! [[ "$ACTION" =~ ^(toy|unit)$ ]]; then
    case "$FORMAT" in
    ucca)
        mkdir pickle
        curl -L http://www.cs.huji.ac.il/~danielh/ucca/ucca-sample.tar.gz | tar xz -C pickle
        TRAIN_DATA="pickle/train/*"
        DEV_DATA="pickle/dev/*"
        ;;
    amr)
        curl --remote-name-all https://amr.isi.edu/download/2016-03-14/alignment-release-{training,dev,test}-bio.txt
        rename 's/.txt/.amr/' alignment-release-*-bio.txt
        python -m semstr.scripts.split -q alignment-release-training-bio.amr alignment-release-training-bio
        CONVERT_DATA=alignment-release-dev-bio.amr
        TRAIN_DATA=alignment-release-training-bio
        DEV_DATA=alignment-release-dev-bio.amr
        ;;
    sdp)
        mkdir data
        curl -L http://svn.delph-in.net/sdp/public/2015/trial/current.tgz | tar xz -C data
        python -m semstr.scripts.split -q data/sdp/trial/dm.sdp data/sdp/trial/dm
        python -m scripts.split_corpus -q data/sdp/trial/dm -t 120 -d 36 -l
        CONVERT_DATA=data/sdp/trial/*.sdp
        TRAIN_DATA=data/sdp/trial/dm/train
        DEV_DATA=data/sdp/trial/dm/dev
        ;;
    esac
fi

case "$TEST_SUITE" in
unit)  # unit tests
    pytest --durations=0 -v tests || exit 1
    python -m semstr.scripts.parse_ud test_files/*.xml -We
    ;;
convert-*)
    python -m semstr.scripts.convert_and_evaluate "$CONVERT_DATA" -v
    ;;
parse-*)
    python -m semstr.scripts.parse_ud $DEV_DATA -We
    ;;
esac
