# Data processing scripts

## Neuron segmentation
- Compute the bounding boxes of neurons from VAST export (in tiles)
    - Neuron mask tiles -> Neuron bounding boxes
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
- Generate the neuron mask within its bounding box  
    ```
    python neuron_mask.py -t neuron-mask -n {NEURON_ID_or_NAME}
    ```
## Vesicle prediction
- Vesicle prediction (in chunks) -> PNGs for VAST import within one neuron mask
    ```
    python vesicle_mask.py -t neuron-vesicle -n {NEURON_ID_or_NAME}
    ```
- VAST export proofread results -> Vesicle proofread h5 volume
    ```
    python run_local.py -t im-to-h5 -p "image_type:seg" -ir "{VAST_EXPORT_FOLDER}/*.png" -o {OUTPUT_FILE}
    ```
- Vesicle proofread h5 volume -> Vesicle instance segmentation within one neuron mask
    ```
    # for memory-efficient version, add the flag: -jn {NUM_CHUNK}
    python vesicle_mask.py -t neuron-vesicle-proofread -ir {VAST_EXPORT_FOLDER} -n {NEURON_ID_or_NAME} -r 1,4,4 
    ```
- Image tiles -> Image h5 volume within one neuron mask
    ```
    python vesicle_mask.py -t neuron-vesicle -v im -p "file_type:h5" -n {NEURON_ID_or_NAME}
    ```
- Image and vesicle instance segmentation -> Image and vesicle segmentation patches
    ```
    python vesicle_mask.py -t neuron-vesicle-patch -ir {RESULT_PATH} -n {NEURON_ID_or_NAME} -v big
    ```

## Misc.
- Print the h5 volume size
    ```
    python run_local.py -t vol-shape -i {H5_PATH}
    ```
