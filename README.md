# BayeSN-numpyro
This is a WIP numpyro implementation of BayeSN

## Dependencies

### Python packages:

- arviz
- astropy
- sncosmo
- extinction
- h5py
- jax (see note below)
- scipy
- ruamel.yaml
- tqdm
- pandas
- numpyro

All of these can just be pip installed to the latest version with the exception of jax. If you want to use GPUs, you must take care to install the correct version of jax following instructions below. If you just want to run on CPU, pip installing jax will be fine.

I used Python 3.10, if you use something different let me know how you get on.

### Requirements for GPU:

- cudatoolkit > 11.4
- cudnn > 8.2
- jax version which matches cudatoolkit/cudnn version, instructions below

## Guide to getting started using GPUs on CSD3

To use GPUs for numpyro, we need to have a version of jax installed designed for GPU - the default pip install will not work. In addition, we need to have cudatoolkit and cudnn installed. CSD3 does provide some versions of cuda and cudnn which can be loaded using "module load" but these are out of date and will not work. I have queried this and this should be fixed soon but in the meantime you can install both cudatoolkit and cudnn in a conda environment. To get GPUs up and running on CSD3, please follow these instructions:

1) Install your favourite conda in your home directory on CSD3. I personally recommend mambaforge, a rewrite of conda which is faster, but any up to date version should do. Note that CSD3 does provide conda but it's a really old version and won't install the right cudatoolkit and cudnn versions.

2) Create a new environment with the Python version of your choice (I used 3.10), via the command "conda create -n yourenvname python=yourpythonversion" ("mamba create ..." if using mamba)

3) Switch to that environment then install cudnn using "conda install -c conda-forge cudnn". This will also install the correct version of cudatoolkit for that cudnn version. The latest cudnn version on conda-forge is currently 8.4.

4) Now install jax using the command "pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html". Make sure you don't install jax for cudnn 8.6 as this will not work - 8.4 is the most recent version on conda-forge.

5) Now pip install everything else.

You should now have a Python environment which will work.


## Running GPU scripts on CSD3

Now you have set up the environment, you can deploy jobs using SLURM scripts. For full details see the CSD3 documentation, but I have included a script called "bayesn.slurm" which you can use. Edit this file to ensure that you're charging to the right account, are running the right script (out of train.py, fit.py etc.) and also edit the line starting with "conda activate" to make sure you use your new conda environment. Once you're happy with that, just run "sbatch bayesn.slurm" to put the job in the queue.

