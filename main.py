
""" 
Main script 
"""

import os 
import cv2
import argparse 
import process 
from tqdm import tqdm


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-r", "--root", required=True, type=str, help='Path to directory containing the input images')
    ap.add_argument("-p", "--save-plots", default=False, type=bool, help='If set to true, plots of extracted data will be saved to output')
    args = vars(ap.parse_args())
    root = args['root']

    if not os.path.exists('outputs/'):
        os.makedirs('outputs/plots', exist_ok=True)
        os.makedirs('outputs/data', exist_ok=True)

    warns = []
    for f in tqdm(os.listdir(root), desc='Progress'):
        path = os.path.join(root, f)
        img = cv2.imread(path)  
        decon = process.DeconstructGraph(img, os.path.splitext(f)[0])
        output, w = decon.run()
        warns.extend(w)

        if args['save_plots']:
            decon.plot_outputs(output)

    print("\n\n[INFO] Some warnings have been raised.")
    for w in warns:
        print(w)

    print("\n\n[INFO] Process compeleted successfuly!")