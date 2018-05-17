import os
from glob import glob

import pytest
from ucca import layer0, layer1
from ucca.ioutil import read_files_and_dirs

from semstr.convert import FROM_FORMAT


@pytest.mark.parametrize("suffix", ("xml", "amr", "sdp", "conllu"))
def test_read_files_and_dirs(suffix):
    for passage in read_files_and_dirs(glob(os.path.join("test_files", "*." + suffix)), converters=FROM_FORMAT):
        assert passage.layer(layer0.LAYER_ID).all, "No terminals in passage " + passage.ID
        assert len(passage.layer(layer1.LAYER_ID).all), "No non-terminals but the root in passage " + passage.ID
