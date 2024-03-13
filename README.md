# MUCRP

Code for the paper submitted to TKDE 2024 (under review).



# Dataset

* Due to file size limitations, we have not uploaded all of the data. The Amazon data can be obtained from [this website](https://jmcauley.ucsd.edu/data/amazon/), and the Douban data can be obtained from [this website](https://github.com/fengzhu1/GA-DTCDR/tree/main).

# Requirements

- Python == 3.8

- Pytorch == 1.11.0

  

# Run

```python
# put the datasets into the data directory
python main.py --dataset YOUR_DATASET --K 30 --opk 0.9 --lam_vl 0.7 --lam_vg 1.0 --lam_vc 0.3 
```
