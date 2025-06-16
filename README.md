# Moseq-Syllable-Clips
Package to generate Syllable Clips from moseq data. Syllable Clips are short videos of a single mouse performing a single syllable, and can be generated from extracted depth data, RGB video data, or IR video data (the latter two possible if you recorded these modalities along with your depth data).

## Install
`moseq-syllable-clips` can be installed into a `moseq2-app` environment, or you could make your own virtual environment:
```sh
conda create -n moseq-syllable-clips python=3.7
conda activate moseq-syllable-clips
```

Please ensure the following dependencies are installed:
```sh
conda install -c conda-forge ffmpeg=4.2.0
pip install git+https://github.com/dattalab/moseq2-viz.git
```

Then install this package:
```sh
pip install git+https://github.com/tischfieldlab/moseq-syllable-clips.git
```


## Usage

Running the following command will generate a usage summary:
```bash
syllable-clips --help
```

There are a few sub commands depending on the extent you want to generate videos. These all mostly share the same options.
```
single              Render just one example of one given syllable
single-multiple     Render multiple examples of one given syllable
corpus              Render just one example of each syllable
corpus-multiple     Render multiple examples of each syllable
```

You may need to prepare a manifest file for the program to find your RGB/IR videos. This is a simple tabular data format with each row corresponding to a moseq session, with a column containing the session UUID and another column containing the session ID (i.e. `session_12345678910`). specify this information to the program via the `--manifest`, `--man-uuid-col` and `--man-session-id-col` parameters

You need to specify the directory path containing your RGB or IR videos using the `--raw-path` parameter. Your videos should be contained within directories named as the `session_id`, as subfolders within the `--raw-path` folder. It is also expected that RGB videos are named `rgb.mp4` and have a corresponding `rgb_ts.txt` file containing timestamps for each video frame. Similarly, for IR video data, it is expected the videos are named `ir.mp4` and have a corresponding `ir_ts.txt` file containing frame timestamps. If your sessions are compressed (as `*.tar.gz` files), the program will extract the necessary files before generating syllable clips. The `--scratch` parameter controls the location of the extracted files. Which streams are produced is controlled by the `--streams` parameter. This also accepts a `composed` value, which will result in a video with all streams stacked horizontally.

