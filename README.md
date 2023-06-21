# Spatial-Spectral ConvNeXt for Hyperspectral Image Classification

This is a code demo for the paper "Y. Zhu, K. Yuan, W. Zhong and L. Xu, "Spatial-Spectral ConvNeXt for Hyperspectral Image Classification," in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, doi: 10.1109/JSTARS.2023.3282975.". More specifically, it is detailed as follow.

<img src="https://i.328888.xyz/2022/12/25/DFBPw.png" alt="DFBPw.png" border="0" />


## environment we use

python = 3.9

pytorch = 1.10.2

cuda = 11.3

## dataset we use
Indian Pine and pavia university can be downloded at https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes

WHU-Hi-HanChuan and WHU-Hi-HongHu can be downloded at http://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm

You should put the HSI data and the corresponding target under the directory "./HSIdata"

An example dataset folder has the following structure:

```
HSI_data
└───IN
│   │——Indian_pines_corrected.mat
│   │——Indian_pines_gt.mat
└───PU
    │——PaviaU.mat
    │——PaviaU_gt.mat
```

## Usage
Take SS-ConvNeXt method on the IN dataset as an example:

1. Download the required data set and move to folder **./HSIdata**.
2. Modify the file **config.josn**. If you choose In dataset, you should set patch size = 9, batch size = 16.
3. create training, validation and test mask. Modify the corresponding paramater in the **config.josn**(i.e. **mask_para**)
4. run main.py
5. the result will be saved under the directory **./Indian pines_result_Fixed**
## Results
### Numerical evaluation
<img src="https://img1.imgtp.com/2023/06/21/UGp87PAX.png" alt="Table.png" title="Table.png" />
### Visual evaluation
<img src="https://img1.imgtp.com/2023/06/21/ILrdeTsy.png" alt="ILrdeTsy.png" border="0" />
