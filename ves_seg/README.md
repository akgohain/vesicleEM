# Large vesicle segmentation

This module contains code for training and infering on a new dataset, as well as pretrained model checkpoints and a sample for validation. For setup, install [Pytorch Connectomics](https://connectomics.readthedocs.io/en/latest/notes/installation.html), [em_util](https://github.com/PytorchConnectomics/em_util), [neuroglancer](https://pypi.org/project/neuroglancer), and [cloud-volume](https://pypi.org/project/cloud-volume).

## Training
Download your data. This model was built using the Pytorch Connectomics package, and thus all configuration for training is done within the yaml files within `configs`. Refer to the file `configs/00_base.yaml` as a template. Most fields can be left untouched, but to adapt to your dataset, change the following parameters at the very minimum:

* `DATASET.INPUT_PATH`: points towards directory containing training data. PyTC's dataloader will automatically seperate this into training/validation partitions.
* `DATASET.IMAGE_NAME`: points to a single image or a list of images. Lists can be represented using standard YAML notation or by concatenating lists of images with `@` as a delimiter
* `DATASET.OUTPUT_PATH`: points to an output directory that will contain model checkpoints

After editing the config file, you can execute training with:

`python scripts/main.py --config-base configs/<base.yaml> --config-file configs/bcd_config.yaml`

To continue retraining from a checkpoint, use:

`python scripts/main.py --config-base configs/<base.yaml> --config-file configs/bcd_config.yaml --checkpoint <path-to-checkpoint> SOLVER.ITERATION_RESTART True`

## Inference

Inference tools are available in `tools/process.py`. To do inference, in a Python shell or file, run:

`do_inference(<im_path>, <pred_path>, [<base_config.yaml>, <bcd_config.yaml>], <checkpoint_path>)`

To also get adapted rand, precison, and recall metrics, run:

`infer_and_stats([<im_path1>, ...], [<pred_path1>, ...], [<mask_path1, ...>], [<base_config.yaml>, <bcd_config.yaml>], <checkpoint_path>), 
`

Our final model checkpoint is provided at `outputs/checkpoint_1000000.pth.tar`.

## Visualiztion

Visualing predictions is done with neuroglancer. To load a volume and its segmentation, modify the file `scripts/ng.py`, then run `python -i scripts/ng.py`. A link to view neuroglancer will open. If running on a remote cluster, please note that port forwarding will be necessary to view on your machine. The `screenshot()` function can be used to take high-resolution screenshots of the neuroglancer representation.

Further documentation for both functions is contained in `tools/process.py`

## Sample

A small sample is prepared in the `sample` directory. `7-13` shows a small soma region with large vesicles. To generate large vesicle predictions in the regionm, run the following command from within the root directory:

`python tools/process.py`

After inference, a neuroglancer script that display the CLAHE-enhanced images, neuron mask, ground truth, and predictions with the final model checkpoints can be viewed by running `sample/ng.py` from within the root directory.
