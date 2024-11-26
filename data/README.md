# Data processing scripts



## Notations
- 
- For vesicle prediction, each `tile` is cropped into a smaller `crop` that contains needed neuron regions.

- For large neurons, we internally divide them into `chunks` during the processing.

## Examples
- Compute the bounding boxes of neurons
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
<<<<<<< HEAD
- Generate the neuron mask within its bounding box
        ```
        python neuron_mask.py -t neuron-mask -n {NEURON_IDs_or_NAMEs}
        ```
- Generate the vesicle prediction within each neuron mask
        ```
        python vesicle_mask.py -t neuron-vesicle -n {NEURON_ID_or_NAME}
        ```
- Convert VAST proofreading results into one h5 volume
        ```
        python run_local.py -t im-to-h5 -p "image_type:seg" -ir "{VAST_EXPORT_FOLDER}/*.png" -o {OUTPUT_FILE}
        ```
- Generate the instance segmentation for big and small vesicles
        ```
        # for memory-efficient version, add the flag: -jn {NUM_CHUNK}
        python vesicle_mask.py -t neuron-vesicle-proofread -ir /data/projects/weilab/dataset/hydra/vesicle_pf/ -i KR10_8nm.h5,VAST_segmentation_metadata_KR10.txt -n KR10 -r 1,4,4 
        ```
- Generate the image volume within each neuron mask
        ```
        python vesicle_mask.py -t neuron-vesicle -v im -p "file_type:h5" -n {NEURON_ID_or_NAME}
        ```
- Data processing
    - Print the h5 volume size
        ```
        python run_local.py -t vol-shape -i /data/projects/weilab/dataset/hydra/vesicle_pf/KR4.h5
        ```
- Neuron mask
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
