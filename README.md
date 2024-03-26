# AOSDGN
[RESS 2022] Adaptive open set domain generalization network: Learning to diagnose unknown faults under unknown working conditions


## Paper

Paper link: [Adaptive open set domain generalization network: Learning to diagnose unknown faults under unknown working conditions](https://www.sciencedirect.com/science/article/pii/S0951832022003064)

## Abstract

Recently, domain generalization techniques have been introduced to enhance the generalization capacity of fault diagnostic models under unknown working conditions. Most existing studies assume consistent machine health states between the training and testing data. However, fault modes in the testing phase are unpredictable, and unknown fault modes usually occur, hindering the wide applications of domain generalization-based fault diagnosis methods in industries. To address such problems, this paper proposes an adaptive open set domain generalization network to diagnose unknown faults under unknown working conditions. A local class cluster module is implemented to explore domain-invariant representation space and obtain discriminative representation structures by minimizing triplet loss. An outlier detection module learns optimal decision boundaries for individual class representation spaces to classify known fault modes and recognize unknown fault modes. Extensive experimental results on two test rigs demonstrated the effectiveness and superiority of the proposed method.

##  Proposed Network 


![image](https://github.com/CHAOZHAO-1/AOSDGN/blob/main/IMG/F1.png)

##  BibTex Citation


If you like our paper or code, please use the following BibTex:

```
@article{zhao2022adaptive,
  title={Adaptive open set domain generalization network: Learning to diagnose unknown faults under unknown working conditions},
  author={Zhao, Chao and Shen, Weiming},
  journal={Reliability Engineering \& System Safety},
  volume={226},
  pages={108672},
  year={2022},
  publisher={Elsevier}
}
```
