# Diagnostic Tools

## Demonstration on an image

```bash
python demo.py --model-path /path/to/checkpoint --box-score-thresh 0.5 --index 3575
```
The script visualises detected boxes in an image with indices overlayed, and outputs the predicted scores for each action and each box pair with a valid object. Use the argument `--index` to change images and `--detection-dir` to change the source of detections. For more details of the use of arguments, run `python demo.py --help`.

## Plot learning curves

```bash
python learning_curve.py --source /path/to/training/log1 name1 /path/to/training/log2 name2
```

The script plots the action classification mAP logged during training and validation. The argument `--source` takes an even number of values representing the paths to log files and descriptive names of your choice. The chosen names will appear on the legend.

## Class-wise human-object pair visualisation

```bash
python visualise_and_cache.py --dir /path/to/mat/files 
```

The script generates human-object pairs organised by interaction classes, using cached `.mat` files. To generate these `.mat` files, refer to the [testing section](https://github.com/fredzzhang/spatio-attentive-graphs#testing) in the home page of the repo. Use argument `--dir` to specify the directory where the `.mat` files are stored. By default, visualisations are cached under `./cache` in the current directory. To change the directory, use the argument `--cache-dir`. For more details of the use of arguments, run `python visualise_and_cache.py --help`.

It takes a considerable amount of time (~2h) and space (~30G) to generate and store the visualisations for all classes. To display visualisations from the same interaction class, run the following script

```bash
python generate_html_page.py --image-dir ./cache/class_000/examples
```

This will create a file called `table.html` in your current directory. Open the file to visualise box pairs in a web page. Use the argument `--image-dir` to select different interaction classes. For more layout options, refer to the documentation for class [_pocket.utils.ImageHTMLTable_](https://github.com/fredzzhang/pocket/tree/master/pocket/utils) and make corresponding changes in _generate_html_page.py_