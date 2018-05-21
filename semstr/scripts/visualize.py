import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
from tqdm import tqdm
from ucca import visualization
from ucca.ioutil import get_passages_with_progress_bar
from ucca.normalization import normalize

from semstr.convert import FROM_FORMAT

if __name__ == "__main__":
    argparser = ArgumentParser(description="Visualize the given passages as graphs.")
    argparser.add_argument("passages", nargs="+", help="Passages in any format")
    argparser.add_argument("--tikz", action="store_true", help="print tikz code rather than showing plots")
    argparser.add_argument("--out-dir", help="directory to save figures in (otherwise displayed immediately)")
    argparser.add_argument("--no-normalize", action="store_false", dest="normalize", help="normalize passage")
    args = argparser.parse_args()

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
    for passage in get_passages_with_progress_bar(args.passages, desc="Visualizing", converters=FROM_FORMAT):
        if args.normalize:
            normalize(passage)
        if args.tikz:
            tikz = visualization.tikz(passage)
            if args.out_dir:
                with open(os.path.join(args.out_dir, passage.ID + ".tikz.txt"), "w") as f:
                    print(tikz, file=f)
            else:
                with tqdm.external_write_mode():
                    print(tikz)
        else:
            plt.figure(figsize=(19, 10))
            visualization.draw(passage)
            if args.out_dir:
                plt.savefig(os.path.join(args.out_dir, passage.ID + ".png"))
            else:
                mng = plt.get_current_fig_manager()
                mng.full_screen_toggle()
                plt.show()
