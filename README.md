# Concept Bottleneck Models (under construction)

![teaser](https://github.com/yewsiang/ConceptBottleneck/blob/master/figures/teaser_landscape.png)

This repository contains code and scripts for the following paper:

> Concept Bottleneck Models
>
> Pang Wei Koh\*, Thao Nguyen\*, Yew Siang Tang\*, Stephen Mussmann, Emma Pierson, Been Kim, and Percy Liang
>
> ICML 2020

The experiments use the following datasets:
- [NIH Osteoarthritis Initiative (OAI)](https://nda.nih.gov/oai/)
- [Caltech-UCSD Birds 200 (CUB)](http://www.vision.caltech.edu/visipedia/CUB-200.html)

The NIH Osteoarthritis Initiative (OAI) dataset requires an application for data access, so we are unable to provide the raw data here. 
We focus instead on scripts replicating our results on CUB, which is a public dataset.

## Abstract

![teaser](https://github.com/yewsiang/ConceptBottleneck/blob/master/figures/tti_qual_examples.png)

We seek to learn models that we can interact with using high-level concepts:
would the model predict severe arthritis if it thinks there is a bone spur in the x-ray?
State-of-the-art models today do not typically support the manipulation of concepts like "the existence of bone spurs",
as they are trained end-to-end to go directly from raw input (e.g., pixels) to output (e.g., arthritis severity).
We revisit the classic idea of first predicting concepts that are provided at training time,
and then using these concepts to predict the label.
By construction, we can intervene on these _concept bottleneck models_
by editing their predicted concept values and propagating these changes to the final prediction.
On x-ray grading and bird identification, concept bottleneck models achieve competitive accuracy with standard end-to-end models,
while enabling interpretation in terms of high-level clinical concepts ("bone spurs") or bird attributes ("wing color").
These models also allow for richer human-model interaction: accuracy improves significantly if we can correct model mistakes on concepts at test time.

## Prerequisites
We used the same environment as Codalab's default gpu setting, please run `pip install -r requirements.txt`. Main packages are:
- matplotlib 3.1.1
- numpy 1.17.1
- pandas 0.25.1
- Pillow 6.1.0
- scipy 1.3.1
- scikit-learn 0.21.3
- torch 1.1.0
- torchvision 0.4.0

### Docker
You can pull the Docker image directly from Docker Hub.
```
docker pull codalab/default-gpu
```

## Usage
Standard task training for CUB can be run using the script in ```scripts/```. More information about how to perform data processing and other evaluations can be found in the README in ```CUB/```.

## Codalab
You can visit our [Codalab page](https://worksheets.codalab.org/worksheets/0x362911581fcd4e048ddfd84f47203fd2) to follow our experiments and retrieve the figures in our paper. 
