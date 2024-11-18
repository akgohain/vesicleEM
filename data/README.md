# Data processing scripts



## Notations
- 
- For vesicle prediction, each `tile` is cropped into a smaller `crop` that contains needed neuron regions.

- For large neurons, we internally divide them into `chunks` during the processing.

## Examples
- Data processing
    - Folder of images -> h5 volume
        ```
        python run_local.py -t im-to-h5 -i "/data/projects/weilab/dataset/hydra/vesicle_pf/KR4 proofread/*.tif" -o /data/projects/weilab/dataset/hydra/vesicle_pf/KR4.h5
        ```
    - Print the h5 volume size
        ```
        python run_local.py -t vol-shape -i /data/projects/weilab/dataset/hydra/vesicle_pf/KR4.h5
        ```
- Neuron mask
    - Neuron mask tiles (VAST export the whole dataset into tiles of the same size) -> Neuron bounding boxes
        ```
        # find all tile names
        python neuron_mask.py tile-names

        # map: compute neuron bounding box for each tile
        # run the generated slurm files for parallel computation
        python slurm.py neuron_mask.py tile-bbox {NUM_JOBS}

        # reduce: result integration
        python neuron_mask.py neuron-bbox
        ```
    - Neuron id -> Bounding box
        ```
        python neuron_mask.py neuron-bbox-print {NEURON_ID}
        ```
    - Neuron id -> Neuron mask within its bounding box
        ```
        python neuron_mask.py neuron-mask {NEURON_ID}
        ```

- Vesicle mask
    - Tile name -> Vesicle instances (im, seg)
        ```
        python vesicle_mask.py chunk {CHUNK_NAME}
        ```