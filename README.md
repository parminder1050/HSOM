# Hybrid SOM based cross-modal retrieval exploiting Hebbian learning
This repository includes the MATLAB file "hebb.m" comprising the implementation of the HSOM approach for cross-modal information retrieval. The images present in the repository helps in understanding the technique and the process followed.

It is an unsupervised cross-modal retrieval framework based on associative learning where two traditional SOMs are trained separately for images and collateral text and then they are associated together using the Hebbian learning network to facilitate the cross-modal retrieval process.

HSOM technique has been tested on popular Wikipedia dataset. To know more about the data and to download it, refer to the link: http://www.svcl.ucsd.edu/projects/crossmodal/

Raw features (comprising SIFT features for images and LDA for text) has been included in the "Wikipedia dataset raw features" folder. 

If you find this repository useful, please cite the below article. 

@article{kaur2022hybrid,
  title={Hybrid SOM based cross-modal retrieval exploiting Hebbian learning},
  author={Kaur, Parminder and Malhi, Avleen Kaur and Pannu, Husanbir Singh},
  journal={Knowledge-Based Systems},
  volume={239},
  pages={108014},
  year={2022},
  publisher={Elsevier}
}
