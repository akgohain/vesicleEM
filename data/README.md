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
        python neuron_mask.py -t tile-names

        # map: compute neuron bounding box for each tile
        # run the generated slurm files for parallel computation
        python run_slurm.py neuron_mask.py "-t tile-bbox" {NUM_JOBS}

        # reduce: result integration
        python neuron_mask.py -t neuron-bbox
        ```
    - Neuron id or name -> Bounding box
        ```
        python neuron_mask.py -t neuron-bbox-print -n {NEURON}
        ```
    - Neuron id or name -> Neuron mask within its bounding box
        ```
        python neuron_mask.py -t neuron-mask -n {NEURON}
        ```

- Vesicle mask
    - Neuron id or name -> Vesicle instance seg or image within the neuron
        ```
        # for big vesicle seg
        python vesicle_mask.py -t neuron-vesicle -n {NEURON} -v big
        # for image
        python vesicle_mask.py -t neuron-vesicle -n {NEURON} -v im -p "file_type:h5"
        
        ```
