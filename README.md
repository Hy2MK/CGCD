
# Proxy Anchor-based Unsupervised Learning for Continuous Generalized Category Discovery
![teaser](assets/teaser.jpg)
Official PyTorch implementation of ICCV 2023 paper [**Proxy Anchor-based Unsupervised Learning for Continuous Generalized Category Discovery**](https://arxiv.org/abs/2307.10943).

Code will be available soon.

## Requirements
- Python3
- PyTorch (> 1.0)
- NumPy
- tqdm

## Datasets
1. Download four public benchmarks for fine-grained dataset
   - [CUB-200-2011](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz)
   - [MIT-67: Indoor Scene Recognition](http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar)
   - [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar)
   - [FGVC-Aircraft Benchmark](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz)

2. Extract the tgz or zip file into `./data/` (Exceptionally, for CUB-200-2011, put the files in a `./data/CUB200`)

## Acknowledgements
Our code is modified and adapted on these great repositories:

- [No Fuss Distance Metric Learning using Proxies](https://github.com/dichotomies/proxy-nca)
- [PyTorch Metric learning](https://github.com/KevinMusgrave/pytorch-metric-learning)

## New Method for Further Improvement
Recently, our paper **Embedding Transfer with Label Relaxation for Improved Metric Learning** which presents the new knowledge distillation method for metric learning is accepted and will be presented at CVPR21.
Our new method can greatly improve the performance, or reduce sizes and output dimensions of embedding networks with negligible performance degradation.
If you are also interested in new knowlege distillation method for metric learning, please check the following arxiv and repository links.

## Citation
If you use this method or this code in your research, please cite as:

   @misc{kim2023proxy,
      title={Proxy Anchor-based Unsupervised Learning for Continuous Generalized Category Discovery}, 
      author={Hyungmin Kim and Sungho Suh and Daehwan Kim and Daun Jeong and Hansang Cho and Junmo Kim},
      year={2023},
      eprint={2307.10943},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
   }

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
