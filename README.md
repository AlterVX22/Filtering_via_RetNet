# Collaborative Filtering with RetNet

## Implementation of collaborative filtering using the RetNet neural network.

**Architecture paper:** [Retentive Network: A Successor to Transformer for Large Language Models](https://arxiv.org/abs/2307.08621)

**RetNet repository:** [TorchScale - A Library of Foundation Architectures](https://github.com/microsoft/torchscale)

Contains 2 classes:
1. **DatasetBatchIterator()** - processes data to form batches.
   Used in:

   Video: [YouTube](https://www.youtube.com/watch?v=dN8U0GNKCcc)

   Slides: [Google Drive](https://drive.google.com/file/d/19oaf9RaS9QqNLgyxvYTqgHoiRTIcMFf4/view)

   Colab Notebook: [Google Colab](https://colab.research.google.com/drive/1cVp1LfjCXtXmRYmOnnb3pD9STUH0tlzl?usp=sharing#scrollTo=Y3d9kwzvvhmp)

3. **NeuralColabFilteringRetNet()** - utilizes input data about users and movies along with their embeddings to predict ratings.
   Additional factors can be incorporated into the code with minor modifications.

An example of working with the dataset is provided in the file example.ipynb.
