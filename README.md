<div align="center">

<h1>DiffMimic: <br> Efficient Motion Mimicking with Differentiable Physics</h1>

<div>
Jiawei Ren&emsp;Cunjun Yu&emsp;Siwei Chen&emsp;Xiao Ma&emsp;Liang Pan</a>&emsp;Ziwei Liu<sup>*</sup>
</div>
<div>
    S-Lab, Nanyang Technological University&emsp; 
    National University of Singapore &emsp;<sup>*</sup>corresponding author
</div>

<div>
   <strong>ICLR 2023</strong>
</div>
<div>
<img src="asset/teaser.gif" width="80%"/>
</div>

---

<h4 align="center">
  <a href="https://diffmimic.github.io/" target='_blank'>[Project Page]</a> •
  <a href="https://openreview.net/forum?id=06mk-epSwZ" target='_blank'>[Paper]</a> •
<a href="https://diffmimic-demo-main-g7h0i8.streamlit.app/" target='_blank'>[Demo]</a>
</h4>

</div>


## Installation

```
conda create -n diffmimic python==3.9
conda activate diffmimic

pip install --upgrade pip
pip install --upgrade "jax[cuda]==0.3.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install brax==0.0.15
pip install flax==0.6.0
pip install streamlit  
pip install tensorflow
```

## Get Started
```shell
python mimic.py --config configs/AMP/backflip.yaml
```

## Visualize
```shell
streamlit run visualize.py
```


## Citation
If you find our work useful for your research, please consider citing the paper:
```
@inproceedings{ren2022diffmimic,
  author    = {Ren, Jiawei and Yu, Cunjun and Chen, Siwei and Ma, Xiao and Pan, Liang and Liu, Ziwei},
  title     = {DiffMimic: Efficient Motion Mimicking with Differentiable Physics},
  journal   = {ICLR},
  year      = {2022},
}
```
