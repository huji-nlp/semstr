import matplotlib.pyplot as plt
import os
from argparse import ArgumentParser
from functools import partial
from ucca import visualization
from ucca.ioutil import get_passages_with_progress_bar, external_write_mode
from ucca.normalization import normalize

from semstr.cfgutil import add_boolean_option
from semstr.convert import FROM_FORMAT, map_labels

FROM_FORMAT["txt"] = FROM_FORMAT["amr"]

if __name__ == "__main__":
    argparser = ArgumentParser(description="Visualize the given passages as graphs.")
    argparser.add_argument("passages", nargs="+", help="Passages in any format")
    add_boolean_option(argparser, "tikz", "print tikz code rather than showing plots")
    argparser.add_argument("--out-dir", help="directory to save figures in (otherwise displayed immediately)")
    add_boolean_option(argparser, "normalize", "normalize passage", default=True)
    add_boolean_option(argparser, "extra-normalization", "more normalization rules")
    add_boolean_option(argparser, "enhanced", "read enhanced dependencies", default=True)
    argparser.add_argument("--label-map", help="CSV file specifying mapping of input edge labels to output edge labels")
    add_boolean_option(argparser, "node-ids", "print tikz code rather than showing plots", short="i")
    argparser.add_argument("-f", "--format", choices=("png", "svg"), default="png", help="image format")
    args = argparser.parse_args()

    FROM_FORMAT = {k: partial(v, **vars(args)) for k, v in FROM_FORMAT.items()}
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
    for passage in get_passages_with_progress_bar(args.passages, desc="Visualizing", converters=FROM_FORMAT):
        map_labels(passage, args.label_map)
        if args.normalize:
            normalize(passage, extra=args.extra_normalization)
        if args.tikz:
            tikz = visualization.tikz(passage, node_ids=args.node_ids)
            if args.out_dir:
                with open(os.path.join(args.out_dir, passage.ID + ".tikz.txt"), "w") as f:
                    print(tikz, file=f)
            else:
                with external_write_mode():
                    print(tikz)
        else:
            plt.figure(figsize=(19, 10))
            plt.title(" ".join(filter(None, (passage.extra.get("format"), passage.ID))))
            visualization.draw(passage, node_ids=args.node_ids)
            if args.out_dir:
                plt.savefig(os.path.join(args.out_dir, passage.ID + "." + args.format))
            else:
                mng = plt.get_current_fig_manager()
                mng.full_screen_toggle()
                plt.show()
