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
        curl -L https://github.com/UniversalConceptualCognitiveAnnotation/UCCA_English-Wiki/releases/download/v1.2.4/ucca-sample.tar.gz | tar xz -C pickle
        TRAIN_DATA="pickle/train/*"
        DEV_DATA="pickle/dev/*"
        ;;
    amr)
        curl --remote-name-all https://amr.isi.edu/download/2016-03-14/alignment-release-{training,dev,test}-bio.txt
        rename 's/.txt/.amr/' alignment-release-*-bio.txt
        python -m semstr.scripts.split -q alignment-release-training-bio.amr -o alignment-release-training-bio
        CONVERT_DATA=alignment-release-dev-bio.amr
        TRAIN_DATA=alignment-release-training-bio
        DEV_DATA=alignment-release-dev-bio.amr
        ;;
    sdp)
        mkdir data
        curl -L http://svn.delph-in.net/sdp/public/2015/trial/current.tgz | tar xz -C data
        python -m semstr.scripts.split -q data/sdp/trial/dm.sdp -o data/sdp/trial/dm
        python -m scripts.split_corpus -q data/sdp/trial/dm -t 120 -d 36 -l
        CONVERT_DATA=data/sdp/trial/*.sdp
        TRAIN_DATA=data/sdp/trial/dm/train
        DEV_DATA=data/sdp/trial/dm/dev
        ;;
    esac
fi

case "$ACTION" in
unit)  # unit tests
    pytest --durations=0 -v tests || exit 1
    python -m semstr.scripts.parse_ud test_files/*.xml -We
    python -m semstr.scripts.validate test_files/*.* --strict -s
    ;;
evaluate)
    python -m semstr.evaluate test_files/conversion/120.{sdp,xml}
    python -m semstr.evaluate test_files/conversion/120.{conll,xml}
    python -m semstr.evaluate test_files/conversion/120.{export,xml}
    python -m semstr.evaluate test_files/conversion/120.{xml,sdp}
    ;;
convert)
    python -m semstr.scripts.convert_and_evaluate "$CONVERT_DATA" -v
    ;;
parse)
    python -m semstr.scripts.parse_ud "$DEV_DATA" -We
    ;;
parse_udpipe)
    curl -O https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2364/udpipe-ud-2.0-170801.zip
    unzip udpipe-ud-2.0-170801.zip
    python -m semstr.scripts.parse_ud "$DEV_DATA" -We --udpipe udpipe-ud-2.0-170801/english-ud-2.0-170801.udpipe \
        --label-map=semstr/util/resources/ud_ucca_label_map.csv
    ;;
tupa)
    pip install -U --upgrade-strategy=only-if-needed tupa
    python -m spacy download en_core_web_md -q
    python -m tupa test_files/504.xml -t test_files/504.xml -I 1 --max-words-external=50 --word-dim=10 --lstm-layer-dim=10 --embedding-layer-dim=10
esac
